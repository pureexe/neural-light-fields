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
        with torch.no_grad():
            if torch.min(norm_x) < -1.0: print('caution: norm_x underflow detected')
            if torch.min(norm_x) > 1.0: print('caution: norm_x overflow detected')
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
                D = self.in_channels
                H = self.height_size[level]
                W = self.width_size[level]
                plane = F.interpolate(self.planes[prev].data, size=(D, H, W), mode='trilinear',  align_corners=True)
                self.planes[level].data = (plane)
    
    def forward(self, x, sigma_only=False):
        grid = self.grid_normalize(x) #(num_ray, in_channel)
        grid = grid.view(grid.shape[0],-1,2)
        h_loc = self.in_channels_loc.to(grid.device).view(1,1,-1).expand(grid.shape[0],grid.shape[1],-1)
        grid = torch.cat([h_loc[...,None], grid[..., None]], dim=-1)[None,:] #(1,  num_ray, in_channel//2, 3)
        raise Exception("Require a proper support of F.gridsample")
        feature = torch.nn.functional.grid_sample(self.planes[self.current_level], grid, mode='bilinear',align_corners=True)[0] #[outchannel*component, num_ray, in_channel]
        feature = torch.prod(feature, dim=-1) #[out_channel*component, num_ray]: combine (outer product) across each input
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
        #print("######\n#########\n#####\n start normalize \n #############\n ########## \n ##############")
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

net_dict = {
    'base': BaseNet,
    'nerf': NeRFNet,
    'twoplane_tensorf': TwoPlaneTensoRF,
    'twoplane_c2f_tensorf': TwoPlaneCoarse2FineTensorRF,
    'cp_decomposition': CPdecomposition
}
