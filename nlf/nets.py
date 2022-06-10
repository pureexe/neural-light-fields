#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

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

class TwoPlaneTensoRF(nn.Module):
    #TODO: proper support coarse to fine
    #TODO: support advance intitialize
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        if in_channels != 4:
            raise Exception("TwoPlaneTensoRF only lightfield location (4 inputs)")

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #TODO: need to read from config
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
    'twoplane_tensorf': TwoPlaneTensoRF
}
