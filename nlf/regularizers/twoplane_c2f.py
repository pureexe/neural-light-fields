#TotalVariationRegularizer is designed for 

import torch

from nlf.regularizers.base import BaseRegularizer


class TwoPlaneC2F(BaseRegularizer):
    # upscale
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)
        self.epoch_counter = 0
        self.compatible_with_set_level = ["TwoPlaneCoarse2FineTensorRF", "CPdecomposition", "PlaneDecomposition", "MixedTwoPlaneRF"]

    def loss(self, batch, batch_idx):
        #actually, there is nothing to do with batch 
        system = self.get_system()
        epoch = system.current_epoch
        require_check = False
        color_model = system.render_fn.model.color_model
        if epoch != self.epoch_counter:
            self.epoch_counter = epoch
            require_check = True
        if not require_check:
            return 0.0
        color_model_name = type(color_model).__name__
        if color_model_name in self.compatible_with_set_level:
            if epoch in color_model.upsampling_epoch:
                for idx, epoch_id in enumerate(color_model.upsampling_epoch):                    
                    if epoch_id == epoch:
                        color_model.set_level(idx+1)
        else:
            raise Exception("No totalvariation apply, please remove from config file if not use")
        return 0.0

