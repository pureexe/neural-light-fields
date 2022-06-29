# visualize ST plane which UV stay on some position

import torch

from nlf.visualizers.base import BaseVisualizer
from utils.ray_utils import (
    get_epi_rays,
)

from utils.visualization import (
    get_warp_dimensions,
    visualize_warp
)

def from_uvst(u, v, s, t, near, far):
    rays = torch.stack(
        [
            s,
            t,
            near * torch.ones_like(s),
            u - s,
            v - t,
            (far - near) * torch.ones_like(s),
        ],
        axis=-1
    ).view(-1, 6)
    # direction need to be normalize
    rays = torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2, dim=-1)
        ],
        -1
    )
    return rays

class UV_EPIVisualizer(BaseVisualizer):
    def __init__(self, system, cfg):
        super().__init__(system, cfg)

        self.u = cfg.u if 'u' in cfg else 0 #None
        self.v = cfg.v if 'v' in cfg else 0 #None
        self.s = cfg.s if 's' in cfg else 0 #None
        self.t = cfg.t if 't' in cfg else 0 #None
        self.H = cfg.H if 'H' in cfg else None

        self.near = cfg.near if 'near' in cfg else -1.0
        self.far = cfg.far if 'far' in cfg else 0.0

        self.st_scale = 1.0
        self.uv_scale = 1.0
        self.var_name = {
            'rgb': 'uv_rgb',
            'warp': 'uv_warp',
            'tform': 'uv_tform'
        }
        
    def get_ray(self, H,W, aspect):
        # Coordinates
        u = torch.linspace(-1,1, W)
        v = torch.linspace(-1,1, H) / aspect
        vu = list(torch.meshgrid([v, u]))
        u = vu[1] * self.uv_scale 
        v = vu[0] * self.uv_scale
        s = torch.ones_like(vu[0]) * self.s * self.st_scale
        t = torch.ones_like(vu[0]) * self.t * self.st_scale 
        rays = from_uvst(u, v, s, t, self.near, self.far)
        return rays

    def validation(self, batch, batch_idx):
        if batch_idx > 0:
            return
            
        system = self.get_system()
        W = system.cur_wh[0]
        H = system.cur_wh[1]
        ASPECT = system.trainer.datamodule.train_dataset.aspect
        if self.H is not None:
            H = self.H
        ## Forward
        outputs = {}
        
        rays = self.get_ray(H,W,ASPECT).type_as(batch['rays'])

        # RGB
        rgb = system(rays)['rgb']
        rgb = rgb.view(H, W, 3).cpu()
        rgb = rgb.permute(2, 0, 1)

        outputs[self.var_name['rgb']] = rgb

        # Warp
        if not system.is_subdivided:
            embedding = system.embed(rays)['embedding']
            params = system.embed_params(rays)['params']
        elif system.render_fn.subdivision.max_hits < 4:
            embedding = system.render_fn.embed_vis(rays)['embedding']
            params = system.render_fn.embed_params_vis(rays)['params']

        if not system.is_subdivided or system.render_fn.subdivision.max_hits < 4:
            warp_dims = get_warp_dimensions(
                embedding, W, H, k=min(embedding.shape[-1], 3)
            )
            warp = visualize_warp(
                embedding, warp_dims, W, H, k=min(embedding.shape[-1], 3)
            )
            outputs[self.var_name['warp']] = warp

            warp_dims = get_warp_dimensions(
                params, W, H, k=min(embedding.shape[-1], 3)
            )
            warp = visualize_warp(
                params, warp_dims, W, H, k=min(embedding.shape[-1], 3)
            )
            outputs[self.var_name['tform']] = warp

        return outputs

    def validation_image(self, batch, batch_idx):
        if batch_idx > 0:
            return {}

        # Outputs
        temp_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/epi_{key}'] = temp_outputs[key]

        return outputs

class ST_EPIVisualizer(UV_EPIVisualizer):
    def __init__(self, system, cfg):
        super().__init__(system, cfg)
        self.var_name = {
            'rgb': 'st_rgb',
            'warp': 'st_warp',
            'tform': 'st_tform'
        }

    def get_ray(self, H,W, aspect):
        # Coordinates
        s = torch.linspace(-1,1, W)
        t = torch.linspace(-1,1, H) / aspect
        ts = list(torch.meshgrid([t, s]))
        s = ts[1] * self.st_scale 
        t = ts[0] * self.st_scale
        u = torch.ones_like(ts[0]) * self.u * self.uv_scale
        v = torch.ones_like(ts[0]) * self.v * self.uv_scale 
        rays = from_uvst(u, v, s, t, self.near, self.far)
        return rays