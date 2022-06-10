#TotalVariationRegularizer is designed for 

import torch

from nlf.regularizers.base import BaseRegularizer


class TotalVariationRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        self.tv_weight = cfg.weight
        super().__init__(system, cfg)


    def loss(self, batch, batch_idx):
        #actually, there is nothing to do with batch 
        system = self.get_system()
        color_model = system.render_fn.model.color_model
        color_model_name = type(color_model).__name__
        if color_model_name == "TwoPlaneTensoRF":
            # in UV,ST mode
            loss = self.tv_weight * (plane_tv(color_model.uv_planes) + plane_tv(color_model.st_planes))
        else:
            raise Exception("No totalvariation apply, please remove from config file if not use")
        return loss

def plane_tv(x):
    """
    Total variation for grid_sample plane
    @params x: plane to do a tv #shape[batch,channel,height,width]
    @return tv: value of total variation
    """
    batch_size = x.shape[0]
    count_h = torch.prod(torch.tensor(x[...,1:,:].shape))
    count_w = torch.prod(torch.tensor(x[...,:,1:].shape))
    h_tv = torch.pow((x[...,1:,:]-x[...,:-1,:]),2).sum()
    w_tv = torch.pow((x[...,1:]-x[...,:-1]),2).sum()
    tv = (h_tv / count_h) + (w_tv / count_w) 
    tv = 2*tv*batch_size
    return tv