import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from valora.models.qwen7b.triton_kernel.context_flashattention_nopad import context_attention_fwd
from valora.models.qwen7b.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from valora.models.qwen7b.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from valora.models.qwen7b.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2
from valora.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from valora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from valora.models.qwen7b.layer_weights.transformer_layer_weight import QwenTransformerLayerWeight
from valora.models.qwen7b.infer_struct import QwenInferStateInfo

class QwenTransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    Qwen-specific implementation of Transformer layer inference,
    extended from Llama's implementation with additional modifications.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        key_value_head_num_ = network_config["num_key_value_heads"]
        assert key_value_head_num_ % self.world_size_ == 0
        self.tp_k_head_num_ = key_value_head_num_ // self.world_size_
        self.tp_v_head_num_ = key_value_head_num_ // self.world_size_

        return
    
    # gqa attention
    def _context_attention_kernel(self, q, k, v, infer_state: QwenInferStateInfo, layer_weight:QwenTransformerLayerWeight) -> torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor
    
    # gqa attention
    def _token_decode_attention_normal(self, q, infer_state: QwenInferStateInfo):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.b_loc,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)
        
        if triton.__version__ == "2.0.0":
            prob = torch.empty_like(att_m_tensor)
            token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
            att_m_tensor = None

            o_tensor = torch.empty_like(q)

            token_att_fwd2(prob,
                        infer_state.mem_manager.value_buffer[self.layer_num_],
                        o_tensor.view(calcu_shape1),
                        infer_state.b_loc,
                        infer_state.b_start_loc,
                        infer_state.b_seq_len,
                        infer_state.max_len_in_batch)
            prob = None
            return o_tensor
        elif triton.__version__ >= "2.1.0":
            o_tensor = torch.empty_like(q)
            from valora.models.llama2.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
            token_softmax_reducev_fwd(att_m_tensor, 
                                      infer_state.mem_manager.value_buffer[self.layer_num_],
                                      o_tensor.view(calcu_shape1),
                                      infer_state.b_loc,
                                      infer_state.b_start_loc,
                                      infer_state.b_seq_len,
                                      infer_state.max_len_in_batch,
                                      infer_state.other_kv_index)
            return o_tensor
        else:
            raise Exception("not support triton version")
