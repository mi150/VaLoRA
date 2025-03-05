import os
import json
import torch

from valora.models.internlm.layer_infer.transformer_layer_infer import InternlmTransformerLayerInfer
from valora.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from valora.models.llama.model import LlamaTpPartModel


class InternlmTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = InternlmTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = InternlmTransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num,mem_adapter_size, load_way="HF", mode=[], dummy=False):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size,load_way, mode, dummy=dummy)
        return
    