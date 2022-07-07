#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

from nlf.embedding import WindowedPE
from nlf.activations import get_activation


class BaseNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
    ):
        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = cfg.skips if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'

        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels, self.W)
            else:
                layer = nn.Linear(self.W, self.W)

            layer = nn.Sequential(layer, get_activation(self.layer_activation))
            setattr(self, f'encoding{i+1}', layer)

        self.encoding_final = nn.Linear(self.W, self.W)

        # Output
        self.out_layer = nn.Sequential(
            nn.Linear(self.W, self.out_channels),
            get_activation(self.activation)
        )

    def forward(self, x, sigma_only=False):
        input_x = x

        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f'encoding{i+1}')(x)

        encoding_final = self.encoding_final(x)
        return self.out_layer(encoding_final)

    def set_iter(self, i):
        pass

class TensoRFBase(nn.Module):
    # TensoRF base model for share function
    def __init__(self):
        super().__init__()

    def get_stepsize(self, init, final, step):
        #linear in log space
        return (torch.round(torch.exp(torch.linspace(np.log(init), np.log(final), step))).long()).tolist()

    def grid_normalize(self, x):
        # normalzie value
        bounds = self.bounds.to(x.device)
        lower = bounds[0:1].expand(x.shape[0],-1)
        upper = bounds[1:2].expand(x.shape[0],-1)

        norm_x = (x - lower) / (upper - lower) #normalize 
        #norm_x = torch.clip(norm_x,0.0,1.0) #clip over/under 1.0
        norm_x = (norm_x * 2.0) - 1.0
        return norm_x

class CPdecomposition(TensoRFBase):
    # CANDECOMP/PARAFAC decompsotion proposed in tensoRF

    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels        

        self.n_comp = cfg.n_comp
        self.current_level = 0
        self.upsampling_epoch = cfg.upsampling_epoch
        self.bounds = torch.tensor(cfg.bounds)
        self.width_size = self.get_stepsize(cfg.plane_init[0], cfg.plane_final[0], len(self.upsampling_epoch) + 1)
        scale = cfg.initial_scale
        self.activation = get_activation(cfg.activation)
        print("Expected width")
        print(self.width_size)
        
        # planes contain level-of-plane while each plane contain height as input channel and width
        self.planes = []
        for i in range(len(self.upsampling_epoch) + 1):
            plane_shape = (1, self.n_comp * out_channels, in_channels, self.width_size[i])
            self.planes.append(torch.nn.Parameter(scale * (2 * torch.randn(plane_shape) - 1)))
        self.planes = torch.nn.ParameterList(self.planes)
        self.in_channels_loc = torch.linspace(-1, 1, in_channels)
    
    def set_level(self,level):
        self.current_level = level
        prev = level-1
        if prev >= 0:
            with torch.no_grad():
                H = self.in_channels
                W = self.width_size[level]
                plane = F.interpolate(self.planes[prev].data, size=(H, W), mode='bilinear',  align_corners=True)
                self.planes[level].data = (plane)
    
    def forward(self, x, sigma_only=False):
        grid = self.grid_normalize(x) #(num_ray, in_channel)
        h_loc = self.in_channels_loc.to(grid.device).view(1,-1).expand(grid.shape[0],-1)
        grid = torch.cat([h_loc[...,None], grid[..., None]], dim=-1)[None,:] #(1,  num_ray, in_channel, 2)
        feature = torch.nn.functional.grid_sample(self.planes[self.current_level], grid, mode='bilinear',align_corners=True)[0] #[outchannel*component, num_ray, in_channel]
        feature = torch.prod(feature, dim=-1) #[out_channel*component, num_ray]: combine (outer product) across each input
        feature = feature.permute(1,0).view(-1,self.n_comp,self.out_channels) #[num_ray, component, out_channel]
        feature = torch.sum(feature,dim=1) #combine product across componenet #[num_ray, out_channel]
        output = self.activation(feature)
        return output

class PlaneDecomposition(TensoRFBase):
    # decomposition as vector of plane
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels     

        if in_channels % 2 != 0:
            raise Exception("Vector decomposition is require to an even input channel (%2==0)")   

        self.n_comp = cfg.n_comp
        self.current_level = 0
        self.upsampling_epoch = cfg.upsampling_epoch
        self.bounds = torch.tensor(cfg.bounds)
        self.height_size = self.get_stepsize(cfg.plane_init[0], cfg.plane_final[0], len(self.upsampling_epoch) + 1)
        self.width_size = self.get_stepsize(cfg.plane_init[1], cfg.plane_final[1], len(self.upsampling_epoch) + 1)
        scale = cfg.initial_scale
        self.activation = get_activation(cfg.activation)
        print("Expected width")
        print(self.width_size)
        
        # planes contain level-of-plane while each plane contain height as input channel and width

        # Pytorch lightning prefer to use register_buffer instead 
        # @see https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/2
        # @see https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html
       
        self.planes = []
        for i in range(len(self.upsampling_epoch) + 1):
            plane_shape = (1, self.n_comp * out_channels, in_channels // 2, self.height_size[i], self.width_size[i])
            self.planes.append(torch.nn.Parameter(scale * (2 * torch.randn(plane_shape) - 1)))
        self.planes = torch.nn.ParameterList(self.planes)

        self.in_channels_loc = torch.linspace(-1, 1, in_channels // 2)
    

    
    def set_level(self,level):
        self.current_level = level
        prev = level-1
        if prev >= 0:
            with torch.no_grad():
                D = self.in_channels // 2
                H = self.height_size[level]
                W = self.width_size[level]
                print("Set plane resolution to: (",D,",",H,",",W,")")
                
                plane = F.interpolate(self.planes[prev].data, size=(D, H, W), mode='trilinear',  align_corners=True)
                self.planes[level].data = (plane)

    
    def forward(self, x, sigma_only=False):
        grid = self.grid_normalize(x) #(num_ray, in_channel)
        grid = grid.view(grid.shape[0],-1,2) #(num_ray, in_channel // 2, 2)
        h_loc = self.in_channels_loc.to(grid.device).view(1,-1,1).expand(grid.shape[0],-1,-1) #(num_ray, in_channel // 2, 1)
        grid = torch.cat([grid, h_loc], dim=-1) #(num_ray, in_channel//2, 3) h_loc need to be last due to x,y,z format [32768, 2, 3])
        grid = grid.permute(1,0,2)[None]  #(1, in_channel//2, num_ray, 1, 3) #NDH3
        grid = grid[...,None,:]  #NDHW3
        feature = torch.nn.functional.grid_sample(self.planes[self.current_level], grid, mode='bilinear',align_corners=True)[0][...,0] #[outchannel*component, in_channel//2, num_ray]
        #feature = torch.nn.functional.grid_sample(self.planes(self.current_level), grid, mode='bilinear',align_corners=True)[0][...,0] #[outchannel*component, in_channel//2, num_ray]
        feature = torch.prod(feature, dim=-2) #[out_channel*component, num_ray]: combine (outer product) across each input
        feature = feature.permute(1,0).view(-1,self.n_comp,self.out_channels) #[num_ray, component, out_channel]
        feature = torch.sum(feature,dim=1) #combine product across componenet #[num_ray, out_channel]
        output = self.activation(feature)
        return output


class TwoPlaneCoarse2FineTensorRF(TensoRFBase):
    #this model will keep "all" planes size which eatting up the memory.
    #TODO: instead of use multiple level like this, we must has BIG tensor but use partial of it to train until reach final state

    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()
        if in_channels != 4:
            raise Exception("TwoPlaneCoarse2FineTensorRF only lightfield location (4 inputs)")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.n_comp = cfg.n_comp
        self.current_level = 0
        self.upsampling_epoch = cfg.upsampling_epoch
        self.bounds = torch.tensor(cfg.bounds)

        self.height_step = self.get_stepsize(cfg.plane_init[0], cfg.plane_final[0], len(self.upsampling_epoch) + 1)
        self.width_step = self.get_stepsize(cfg.plane_init[1], cfg.plane_final[1], len(self.upsampling_epoch) + 1)
        scale = cfg.initial_scale
        self.activation = get_activation(cfg.activation)
        print("Expected width")
        print(self.width_step)
        self.uv_planes = []
        self.st_planes = []


        for i in range(len(self.upsampling_epoch) + 1):
            plane_shape = (1, self.n_comp * out_channels, self.height_step[i], self.width_step[i])
            self.uv_planes.append(torch.nn.Parameter(scale * (2 * torch.randn(plane_shape) - 1)))
            self.st_planes.append(torch.nn.Parameter(scale * (2 * torch.randn(plane_shape) - 1)))
        
        self.uv_planes = torch.nn.ParameterList(self.uv_planes)
        self.st_planes = torch.nn.ParameterList(self.st_planes)

    def set_level(self, level):
        self.current_level = level
        prev = level-1
        if prev >= 0:
            #interpolate value from previous level
            with torch.no_grad():
                H = self.height_step[level]
                W = self.width_step[level]
                print("Set plane resolution to: (",H,",",W,")")
                uv_plane = F.interpolate(self.uv_planes[prev].data, size=(H, W), mode='bilinear',  align_corners=True)
                st_plane = F.interpolate(self.st_planes[prev].data, size=(H, W), mode='bilinear',  align_corners=True)
                self.uv_planes[level].data = (uv_plane)
                self.st_planes[level].data = (st_plane)
            
    def forward(self, x, sigma_only=False):
        uvst_grid = self.grid_normalize(x)[None,None]
        uv = uvst_grid[...,:2]
        st = uvst_grid[...,2:]
        # uv_planes #torch.Size([1, 48, 300, 300])
        uv_feature = torch.nn.functional.grid_sample(self.uv_planes[self.current_level], uv, mode='bilinear',align_corners=True)[0,:,0]
        st_feature = torch.nn.functional.grid_sample(self.st_planes[self.current_level], st, mode='bilinear',align_corners=True)[0,:,0]
        feature = uv_feature * st_feature #outer product in TensoRF
        feature = feature.permute(1,0).view(-1,self.n_comp,self.out_channels)
        feature = torch.sum(feature,dim=1) #combine product across componenet
        output = self.activation(feature)
        return output

class TwoPlaneTensoRF(nn.Module):
    # for coarse to fine use TwoPlaneCoarse2FineTensorRF instead
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        if in_channels != 4:
            raise Exception("TwoPlaneTensoRF only lightfield location (4 inputs)")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.n_comp = cfg.n_comp
        scale = cfg.initial_scale
        plane_width = cfg.plane_init[1]
        plane_heigth = cfg.plane_init[0]
        self.activation = get_activation(cfg.activation)

        # uv plane and st plane is a 2 slab represent in light field 
        # however, output channel is design for multiple output 
        # in case of RGB (out_channel = 3) each uv_plane contain n_comp*3, you have sum the result seperately 
        # this can be done by permute to the last channel and then reshape to (n_comp,3) and sum over n_comp
        self.uv_planes = torch.nn.Parameter(scale * (2 * torch.randn((1, self.n_comp * out_channels, plane_heigth, plane_width)) - 1))
        self.st_planes = torch.nn.Parameter(scale * (2 * torch.randn((1, self.n_comp * out_channels, plane_heigth, plane_width)) - 1))
        self.bounds = torch.tensor(cfg.bounds)

    def grid_normalize(self, x):
        # normalzie value
        bounds = self.bounds.to(x.device)
        lower = bounds[0:1].expand(x.shape[0],-1)
        upper = bounds[1:2].expand(x.shape[0],-1)

        norm_x = (x - lower) / (upper - lower) #normalize 
        #norm_x = torch.clip(norm_x,0.0,1.0) #clip over/under 1.0
        norm_x = (norm_x * 2.0) - 1.0
        return norm_x
        
    
    def forward(self, x, sigma_only=False):
        uvst_grid = self.grid_normalize(x)[None,None]
        assert torch.min(uvst_grid) > -1.0
        assert torch.max(uvst_grid) < 1.0
        uv = uvst_grid[...,:2]
        st = uvst_grid[...,2:]
        # uv_planes #torch.Size([1, 48, 300, 300])
        uv_feature = torch.nn.functional.grid_sample(self.uv_planes, uv, mode='bilinear',align_corners=True)[0,:,0]
        st_feature = torch.nn.functional.grid_sample(self.st_planes, st, mode='bilinear',align_corners=True)[0,:,0]
        feature = uv_feature * st_feature #outer product in TensoRF
        feature = feature.permute(1,0).view(-1,self.n_comp,self.out_channels)
        feature = torch.sum(feature,dim=1) #combine product across componenet
        # TODO: lookup code
        output = self.activation(feature)
        return output

class MixedTwoPlaneRF(PlaneDecomposition):
    def __init__(self, in_channels, out_channels, cfg):
        # we intensionly increase component 3 times for component pair [uv,st], [us,vt] and [uv,st]
        self.e_comp = cfg.n_comp 
        cfg.n_comp = cfg.n_comp * 3  
        super().__init__(in_channels, out_channels, cfg)

    def get_feature(self, planes, pos_plane1, pos_plane2):
        # pos_plane1 and pos_p2 is the point on the plane shape [batch, 2]
        grid = torch.cat([pos_plane1[:,None,:], pos_plane2[:,None,:]],dim=-2) #(num_ray, 2, 2)
        h_loc = self.in_channels_loc.to(grid.device).view(1,-1,1).expand(grid.shape[0],-1,-1) #(num_ray, 2, 1)
        grid = torch.cat([grid, h_loc], dim=-1) #(num_ray, in_channel//2, 3) h_loc need to be last due to x,y,z format [32768, 2, 3])
        grid = grid.permute(1,0,2)[None]  #(1, in_channel//2, num_ray, 1, 3) #NDH3
        grid = grid[...,None,:]  #NDHW3
        feature = torch.nn.functional.grid_sample(planes, grid, mode='bilinear',align_corners=True)[0][...,0] #[outchannel*component, in_channel//2, num_ray]
        feature = torch.prod(feature, dim=-2) #[out_channel*component, num_ray]: combine (outer product) across each input
        feature = feature.permute(1,0).view(-1,self.e_comp,self.out_channels) #[num_ray, component, out_channel]
        feature = torch.sum(feature,dim=1) #combine product across componenet #[num_ray, out_channel]
        return feature

    def forward(self, x, sigma_only=False):
        grid = self.grid_normalize(x) #(num_ray, in_channel)
        #there has 3 set of planes uv_st, us_vt, ut_vs
        comp_cnt = self.e_comp * self.out_channels
        uv = torch.cat([grid[...,0:1], grid[...,1:2]],dim=-1) #(num_ray, 1, 2)
        st = torch.cat([grid[...,2:3], grid[...,3:4]],dim=-1)
        us = torch.cat([grid[...,0:1], grid[...,2:3]],dim=-1)
        vt = torch.cat([grid[...,1:2], grid[...,3:4]],dim=-1)
        ut = torch.cat([grid[...,0:1], grid[...,3:4]],dim=-1)
        vs = torch.cat([grid[...,1:2], grid[...,3:4]],dim=-1)

        
        uv_st = self.get_feature(self.planes[self.current_level][:,comp_cnt*0:comp_cnt*1], uv, st)
        us_vt = self.get_feature(self.planes[self.current_level][:,comp_cnt*1:comp_cnt*2], us, vt)
        ut_vs = self.get_feature(self.planes[self.current_level][:,comp_cnt*2:comp_cnt*3], ut, vs) 

        feature = torch.cat([uv_st[...,None], us_vt[...,None], ut_vs[...,None]],dim=-1)
        feature = torch.sum(feature,dim=-1)
        output = self.activation(feature)
        return output

  

class NeRFNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
    ):
        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = cfg.skips if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'

        self.pos_pe = WindowedPE(
            3,
            cfg.pos_pe
        )
        self.dir_pe = WindowedPE(
            3,
            cfg.dir_pe
        )

        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.pos_pe.out_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.pos_pe.out_channels, self.W)
            else:
                layer = nn.Linear(self.W, self.W)

            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'encoding{i+1}', layer)

        self.encoding_dir = nn.Sequential(
            nn.Linear(self.W + self.dir_pe.out_channels, self.W // 2),
            nn.ReLU(True)
        )

        self.encoding_final = nn.Linear(self.W, self.W)

        # Output
        self.sigma = nn.Sequential(
            nn.Linear(self.W, 1),
            nn.Sigmoid(),
        )

        self.rgb = nn.Sequential(
            nn.Linear(self.W // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, sigma_only=False):
        input_pos, input_dir = torch.split(x, [3, 3], -1)

        input_pos = self.pos_pe(input_pos)
        input_dir = self.dir_pe(input_dir)

        pos = input_pos

        for i in range(self.D):
            if i in self.skips:
                pos = torch.cat([input_pos, pos], -1)

            pos = getattr(self, f'encoding{i+1}')(pos)

        encoding_final = self.encoding_final(x)
        return self.out_layer(encoding_final)

    def set_iter(self, i):
        pass

class TwoPlaneMLP(nn.Module):
    # split mlp to two plane
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()
        self.net_depth = cfg.depth
        self.net_width = cfg.hidden_channels

        self.in_channels = in_channels
        #self.out_channels = out_channels
        self.n_comp = cfg.n_comp
        self.out_channels = out_channels
        self.skips = cfg.skips if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'
        self.activation = get_activation(self.activation)
        self.positional_encoding = WindowedPE(
            self.in_channels,
            cfg.color_pe
        )
        net_in_ch = self.positional_encoding.out_channels // 2
        for i in range(self.net_depth):
            if i == 0:
                uv_p = nn.Linear(net_in_ch, self.net_width)
                st_p = nn.Linear(net_in_ch, self.net_width)
            elif i in self.skips:
                uv_p = nn.Linear(self.net_width + net_in_ch, self.net_width)
                st_p = nn.Linear(self.net_width + net_in_ch, self.net_width)
            else:
                uv_p = nn.Linear(self.net_width, self.net_width)
                st_p = nn.Linear(self.net_width, self.net_width)

            uv_p = nn.Sequential(uv_p, nn.ReLU(True))
            st_p = nn.Sequential(st_p, nn.ReLU(True))
            setattr(self, f'net_uv_{i+1}', uv_p)
            setattr(self, f'net_st_{i+1}', st_p)

      
        self.uv_comp = nn.Sequential(
            nn.Linear(self.net_width, self.n_comp * 3),
        )
        self.st_comp = nn.Sequential(
            nn.Linear(self.net_width, self.n_comp * 3),
        )

    def forward(self, x, sigma_only=False):     
        uv, st = torch.split(x, [2, 2], -1)

        # fetch uv and st network
        uv_enc = self.positional_encoding(uv)
        st_enc = self.positional_encoding(st)

        x_uv = uv_enc
        x_st = st_enc

        for i in range(self.net_depth):
            if i in self.skips:
                x_uv = torch.cat([x_uv, uv_enc], -1)
                x_st = torch.cat([x_st, st_enc], -1)
            x_uv = getattr(self, f'net_uv_{i+1}')(x_uv)
            x_st = getattr(self, f'net_st_{i+1}')(x_st)
        n_batch = x.shape[0]
        uv_comp = self.uv_comp(x_uv).view(n_batch, -1, 3)
        st_comp = self.st_comp(x_st).view(n_batch, -1, 3)

        # combine component
        rgb = torch.sum(uv_comp * st_comp,dim=-2)
        rgb = self.activation(rgb)
        return rgb

class CombiPlaneMLP(nn.Module):

    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        # network pre-config
        self.net_depth = cfg.depth
        self.net_width = cfg.hidden_channels
        self.in_channels = in_channels
        self.n_comp = cfg.n_comp
        self.out_channels = out_channels
        self.skips = cfg.skips if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'
        self.activation = get_activation(self.activation)
        # plane combination uv,st / us,vt /  ut,sv
        self.net_names = ['uv', 'st', 'us', 'vt', 'ut', 'sv']
        self.positional_encoding = WindowedPE(self.in_channels // 2, cfg.color_pe)
        for name in self.net_names:
            self.build_net(name, self.positional_encoding.out_channels, self.n_comp * 3)

    def build_net(self, name, in_channels, out_channels):
        in_size = in_channels
        for i in range(self.net_depth):
            out_size = self.net_width
            if i == self.net_depth - 1:
                out_size = out_channels 
            if i in self.skips:
                in_size = self.net_width + in_channels
            layer = nn.Linear(in_size, out_size)
            
            if i != self.net_depth - 1:
                layer = nn.Sequential(layer, nn.ReLU(True))
            else:
                layer = nn.Sequential(layer) #last layer will has no activation

            setattr(self, f'{name}_{i+1}', layer)
            in_size = out_size

    def fetch_net(self, name, x):
        x_in = x
        for i in range(self.net_depth):
            if i in self.skips:
                x = torch.cat([x, x_in], -1)
            x = getattr(self, f'{name}_{i+1}')(x)
        return x

    def get_location(self, plane_name, x):
        lookup_id = {'u': 0, 'v': 1, 's': 2, 't':3}
        output = []
        for i in plane_name:
            output.append(x[...,lookup_id[i],None])
        output = torch.cat(output, dim=-1)
        return output

    def forward(self, x, sigma_only=False): 
        
        plane_comps = []
        buff_comp = None 

        for i,name in enumerate(self.net_names):
            loc = self.get_location(name, x)
            enc_loc = self.positional_encoding(loc)
            comp_rgb = self.fetch_net(name, enc_loc).view(-1, self.n_comp, 3)
            if i % 2 == 0:
                buff_comp = comp_rgb
            else:
                buff_comp = buff_comp * comp_rgb
                buff_comp = torch.sum(buff_comp,dim=1) #ray, 3
                plane_comps.append(buff_comp[None])
        
        rgb = torch.cat(plane_comps, dim=0)
        rgb = torch.sum(rgb,dim=0) #ray, 3
        rgb = self.activation(rgb)
        return rgb



net_dict = {
    'base': BaseNet,
    'nerf': NeRFNet,
    'twoplane_tensorf': TwoPlaneTensoRF,
    'twoplane_mlp': TwoPlaneMLP,
    'combiplane_mlp': CombiPlaneMLP,
    'twoplane_c2f_tensorf': TwoPlaneCoarse2FineTensorRF,
    'mixed_twoplane': MixedTwoPlaneRF,
    'cp_decomposition': CPdecomposition,
    'plane_decomposition': PlaneDecomposition
}
