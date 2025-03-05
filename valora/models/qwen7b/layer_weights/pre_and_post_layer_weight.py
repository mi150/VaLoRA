import torch
import numpy as np
from valora.common.basemodel import PreAndPostLayerWeight

class QwenPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights, dummy=False):
        if dummy:
            self._load_dummy_weights()
            return

        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.world_size_

        if "transformer.wte.weight" in weights:
            self.wte_weight_ = self._cuda(weights['transformer.wte.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :])
        if 'lm_head.weight' in weights:
            self.lm_head_weight_ = self._cuda(weights['lm_head.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :])
        if 'transformer.ln_f.weight' in weights:
            self.final_norm_weight_ = self._cuda(weights['transformer.ln_f.weight'])

        return

    def _load_dummy_weights(self):
        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.world_size_
        n_embed = self.network_config_["hidden_size"]
        
        # Dummy weights for embedding and output layers
        self.wte_weight_ = (torch.rand((split_vob_size, n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
        self.lm_head_weight_ = (torch.rand((split_vob_size, n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
        self.final_norm_weight_ = (torch.rand((n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.wte_weight_, self.lm_head_weight_, self.final_norm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return