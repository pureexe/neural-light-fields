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
        elif color_model_name in ["TwoPlaneCoarse2FineTensorRF","AppearanceTensorRF","BrightTensoRF"]:
            lvl = color_model.current_level
            uv_plane = color_model.uv_planes[lvl]
            st_plane = color_model.st_planes[lvl]
            loss = self.tv_weight * (plane_tv(uv_plane) + plane_tv(st_plane))
        elif color_model_name == "PlaneDecomposition" or color_model_name == "MixedTwoPlaneRF":
            lvl = color_model.current_level
            plane = color_model.planes[lvl]
            #plane = color_model.planes(lvl)
            loss = self.tv_weight * (plane_tv(plane))
        elif color_model_name == "PlaneListOfDecomposition":
            lvl = color_model.current_level
            ppl = color_model.plane_per_lvl
            loss = 0.0
            for i in range(ppl):
                plane = color_model.planes[lvl * ppl + i]
                loss = loss +  plane_tv(plane)
            loss = self.tv_weight * loss
        elif color_model_name == "CPdecomposition":
            lvl = color_model.current_level
            plane = color_model.planes[lvl]
            loss = self.tv_weight * line_tv(plane)
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

def plane_tv3d(x):
    batch_size = x.shape[0]
    count_c = x.shape[1]
    count_d = x.shape[2]

    count_h = torch.prod(torch.tensor(x[...,1:,:].shape))
    count_w = torch.prod(torch.tensor(x[...,:,1:].shape))
    h_tv = torch.pow((x[...,1:,:]-x[...,:-1,:]),2).sum()
    w_tv = torch.pow((x[...,1:]-x[...,:-1]),2).sum()
    tv = (h_tv / count_h) + (w_tv / count_w) 
    tv = tv / count_c
    tv = tv / count_d
    
    tv = 2*tv*batch_size
    return tv


def line_tv(x):
    """
    Total variation for grid_sample plane in line(last dimension)
    @params x: plane to do a tv #shape[batch,channel,height,width]
    @return tv: value of total variation
    """
    batch_size = x.shape[0]
    count_w = torch.prod(torch.tensor(x[...,:,1:].shape))
    w_tv = torch.pow((x[...,1:]-x[...,:-1]),2).sum()
    tv = (w_tv / count_w) 
    tv = 2*tv*batch_size
    return tv