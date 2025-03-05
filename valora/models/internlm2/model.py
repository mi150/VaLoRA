import os
import json
import torch

from valora.models.internlm2.layer_weights.transformer_layer_weight import Internlm2TransformerLayerWeight
from valora.models.internlm2.layer_weights.pre_and_post_layer_weight import Internlm2PreAndPostLayerWeight
from valora.models.internlm.model import InternlmTpPartModel


class Internlm2TpPartModel(InternlmTpPartModel):
    # weight class
    pre_and_post_weight_class = Internlm2PreAndPostLayerWeight 
    transformer_weight_class = Internlm2TransformerLayerWeight

    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num,mem_adapter_size, load_way="HF", mode=[], dummy=False):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size,load_way, mode, dummy=dummy)
        return
    