# import torch
# import numpy as np
# from valora.models.llama.infer_struct import LlamaInferStateInfo

# class QwenInferStateInfo(LlamaInferStateInfo):
#     def __init__(self):
#         super().__init__()
#         self.position_cos = None
#         self.position_sin = None
#         self.other_kv_index = None
#         self.logn_values = None

#     def init_some_extra_state(self, 
#             model, 
#             batch_size, 
#             total_token_num,
#             max_len_in_batch,
#             input_ids : torch.Tensor,
#             b_loc : torch.Tensor,
#             b_start_loc : torch.Tensor,
#             b_seq_len : torch.Tensor,
#             is_prefill):
#         use_dynamic_ntk = model.config.get("use_dynamic_ntk", False)
#         if not use_dynamic_ntk:
#             super().init_some_extra_state(model, input_ids=input_ids)
#             return
        
#         if self.is_prefill:
#             b_start_loc_numpy = self.b_start_loc.cpu().numpy()
#             b_seq_len_numpy = self.b_seq_len.cpu().numpy()
#             position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
#                                             for i in range(len(b_seq_len_numpy))], axis=0)).cuda()

#             # Prepare position embeddings
#             self.position_sin = []
#             self.position_cos = []

#             # Calculate NTK indices based on sequence length
#             infer_ntk_id = torch.clamp(
#                 torch.ceil(torch.log2(self.b_seq_len / model.config.get("seq_length", 2048)) + 1), 
#                 0, model.max_ntk_alpha
#             ).long()

#             for i in range(len(infer_ntk_id)):
#                 sin_cached = model._sin_cached[infer_ntk_id[i]]
#                 cos_cached = model._cos_cached[infer_ntk_id[i]]
#                 start = b_start_loc_numpy[i]
#                 end = start + b_seq_len_numpy[i]

#                 self.position_sin.append(sin_cached[position_ids[start:end]])
#                 self.position_cos.append(cos_cached[position_ids[start:end]])

#             self.position_sin = torch.cat(self.position_sin, dim=0)
#             self.position_cos = torch.cat(self.position_cos, dim=0)

#             if model.logn_tensor is not None:
#                 self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
#             position_ids = None
#         else:
#             infer_ntk_id = torch.clamp(
#                 torch.ceil(torch.log2(self.b_seq_len / model.config.get("seq_length", 2048)) + 1), 
#                 0, model.max_ntk_alpha
#             ).long()
#             position_ids = (self.b_seq_len - 1).long()
#             self.position_cos = model._cos_cached[infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
#             self.position_sin = model._sin_cached[infer_ntk_id, position_ids].view(position_ids.shape[0], -1)

#             if model.logn_tensor is not None:
#                 self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)

#             self.other_kv_index = b_loc[0, max_len_in_batch - 1].item()
#             position_ids = None
#         return

import torch
import numpy as np
from valora.common.basemodel import InferStateInfo

class QwenInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None
    
    def init_some_extra_state(self, 
            model, 
            batch_size, 
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_loc : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill):
        if is_prefill:
            b_seq_len_numpy = b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            position_ids = None
        else:
            self.position_cos = torch.index_select(model._cos_cached, 0, b_seq_len - 1).view(b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, b_seq_len - 1).view(b_seq_len.shape[0], -1)
            self.other_kv_index = b_loc[0, max_len_in_batch - 1].item()
        return
