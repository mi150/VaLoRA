import torch
import torch.distributed as dist

from valora.models.qwen.layer_weights.pre_and_post_layer_weight import QwenPreAndPostLayerWeight
from valora.models.qwen.infer_struct import QwenInferStateInfo
from valora.common.basemodel import PreLayerInferTpl
from valora.utils.infer_utils import mark_cost_time


class QwenPreLayerInfer(PreLayerInferTpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: QwenInferStateInfo, layer_weight: QwenPreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embeddings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embeddings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embeddings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embeddings

    def token_forward(self, input_ids, infer_state: QwenInferStateInfo, layer_weight: QwenPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embeddings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embeddings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embeddings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embeddings
