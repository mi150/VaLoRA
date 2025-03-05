import numpy as np
import torch
import torch.nn as nn
from typing import final

from valora.common.infer_utils import init_bloc
from valora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from valora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from valora.utils.infer_utils import mark_cost_time
from valora.utils.infer_utils import calculate_time, mark_start, mark_end
from valora.server.router.req_queue import weight_output_counts, weight_rank_counts, weight_lora_ids, \
    weight_start_ids, weight_tmp_d
from atmm_ops import dispatch_bgmv as dispatch_sgmm
import time
import logging

batch_req_bins = torch.arange(128 * 4096, dtype=torch.long, device="cuda")
batch_req_bins = batch_req_bins // 4096


class LoraPEFTBatchInfer:

    def __init__(self, base_model, infer_adapter=None):
        t1 = time.time()
        self.base_model = base_model
        self.is_sgmm = False
        self.max_lora_dim = a_len = int(max(infer_adapter.a_len)) // 4
        emb_dim = self.base_model.layers_infer[0].embed_dim_

        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            self.key_buffer = infer_adapter.mem_manager.key_buffer
            self.value_buffer = infer_adapter.mem_manager.value_buffer
            try:
                self.adapter_idx = self.infer_adapter.adapter_dirs.index(self.infer_adapter.merged_adapter_dir)
                print("merged_adapter_dir:", self.infer_adapter.merged_adapter_dir, " index:", self.adapter_idx,
                      "rank:", self.infer_adapter.a_len[self.adapter_idx], "max_rank:", self.max_lora_dim,
                      infer_adapter.a_start)
            except ValueError:
                self.adapter_idx = -1
                return
            self.scaling = infer_adapter.a_scaling[self.adapter_idx]
            start = int(infer_adapter.a_start[self.adapter_idx])
            a_len = int(infer_adapter.a_len[self.adapter_idx])
            loc = infer_adapter.a_loc[start:start + a_len]
            self.r = r = a_len // 4
            key_buffer = infer_adapter.mem_manager.key_buffer
            value_buffer = infer_adapter.mem_manager.value_buffer

            if self.is_sgmm:
                # for sgmm
                dtype = torch.float16
                device = torch.device("cuda")
                x_list = [key_buffer[layer_id // 4][loc[self.r * (layer_id % 4):self.r * (layer_id % 4 + 1)]]
                              .reshape(self.r, emb_dim).transpose(0, 1) for layer_id in
                          range(self.base_model.layers_num * 4)]
                x_ptr_l = [t.data_ptr() for t in x_list]
                self.x_ptr = torch.tensor(x_ptr_l, dtype=torch.int64, device=device)
                w_list = [value_buffer[layer_id // 4][loc[self.r * (layer_id % 4):self.r * (layer_id % 4 + 1)]]
                              .reshape(self.r, emb_dim) for layer_id in
                          range(self.base_model.layers_num * 4)]
                w_ptr_l = [t.data_ptr() for t in w_list]
                self.w_ptr = torch.tensor(w_ptr_l, dtype=torch.int64, device=device)
                self.y_list = y_list = [
                    getattr(base_model.trans_layers_weight[layer_id // 4], f"{suffix}_weight_")
                    for layer_id in range(4 * self.base_model.layers_num)
                    for suffix in ['q', 'k', 'v', 'o'][layer_id % 4: layer_id % 4 + 1]
                ]
                y_ptr_l = [t.data_ptr() for t in y_list]
                self.y_ptr = torch.tensor(y_ptr_l, dtype=torch.int64, device=device)
                s_list = [self.r for _ in range(self.base_model.layers_num * 4)]
                self.s = torch.tensor(s_list, dtype=torch.int32, device=device)
                self.type_slice = torch.randn((2, 2), dtype=dtype, device=device)

            else:
                self.batch_lora_A = [torch.zeros((4, emb_dim, self.max_lora_dim), dtype=torch.float16, device="cuda")
                                     for _ in range(self.base_model.layers_num)]
                self.batch_lora_B = [torch.zeros((4, self.max_lora_dim, emb_dim), dtype=torch.float16, device="cuda")
                                     for _ in range(self.base_model.layers_num)]

                for layer_id in range(self.base_model.layers_num):
                    self.batch_lora_A[layer_id][0, :, :r].copy_(
                        key_buffer[layer_id][loc[:r]].reshape(r, emb_dim).transpose(0, 1))
                    self.batch_lora_A[layer_id][1, :, :r].copy_(
                        key_buffer[layer_id][loc[r:r * 2]].reshape(r, emb_dim).transpose(0, 1))
                    self.batch_lora_A[layer_id][2, :, :r].copy_(
                        key_buffer[layer_id][loc[r * 2:r * 3]].reshape(r, emb_dim).transpose(0, 1))
                    self.batch_lora_A[layer_id][3, :, :r].copy_(
                        key_buffer[layer_id][loc[r * 3:r * 4]].reshape(r, emb_dim).transpose(0, 1))

                    self.batch_lora_B[layer_id][0, :r, :].copy_(
                        value_buffer[layer_id][loc[:r]].reshape(emb_dim, r).transpose(0, 1))
                    self.batch_lora_B[layer_id][1, :r, :].copy_(
                        value_buffer[layer_id][loc[r:r * 2]].reshape(emb_dim, r).transpose(0, 1))
                    self.batch_lora_B[layer_id][2, :r, :].copy_(
                        value_buffer[layer_id][loc[r * 2:r * 3]].reshape(emb_dim, r).transpose(0, 1))
                    self.batch_lora_B[layer_id][3, :r, :].copy_(
                        value_buffer[layer_id][loc[r * 3:r * 4]].reshape(emb_dim, r).transpose(0, 1))
            print("Init time:", time.time() - t1)
            torch.cuda.synchronize()

    @torch.inference_mode()
    def merge_adapter(self):
        
        if self.adapter_idx == -1:
            return
        base_model = self.base_model
        st = time.time()
        for layer_id in range(self.base_model.layers_num):
            base_layer_weight = base_model.trans_layers_weight[layer_id]
            base_layer_infer = base_model.layers_infer[layer_id]
            # AxB
            r = self.infer_adapter.a_len[self.adapter_idx] // 4
            a = self.batch_lora_A[layer_id][:4, :, :r]
            b = self.batch_lora_B[layer_id][:4, :r, :] * self.scaling
            ab = torch.bmm(a.view(4, -1, r), b.view(4, r, -1))
            assert ab.shape == (4, base_layer_infer.embed_dim_, base_layer_infer.embed_dim_)
            # W+AB
            base_layer_weight.q_weight_.add_(ab[0])
            base_layer_weight.k_weight_.add_(ab[1])
            base_layer_weight.v_weight_.add_(ab[2])
            base_layer_weight.o_weight_.add_(ab[3])
        torch.cuda.synchronize()
        logging.info(f'merge rank {r} time cost {time.time() - st :.6f} seconds')
        print(f'merge time cost {time.time() - st :.6f} seconds')

    @torch.inference_mode()
    def unmerge_adapter(self):
        if self.adapter_idx == -1:
            return
        base_model = self.base_model
        st = time.time()
        for layer_id in range(self.base_model.layers_num):
            base_layer_weight = base_model.trans_layers_weight[layer_id]
            base_layer_infer = base_model.layers_infer[layer_id]
            # AxB
            r = self.infer_adapter.a_len[self.adapter_idx] // 4
            a = self.batch_lora_A[layer_id][:4, :, :r]
            b = self.batch_lora_B[layer_id][:4, :r, :] * self.scaling
            ab = torch.bmm(a.view(4, -1, r), b.view(4, r, -1))
            assert ab.shape == (4, base_layer_infer.embed_dim_, base_layer_infer.embed_dim_)
            # W-AB
            base_layer_weight.q_weight_.sub_(ab[0])
            base_layer_weight.k_weight_.sub_(ab[1])
            base_layer_weight.v_weight_.sub_(ab[2])
            base_layer_weight.o_weight_.sub_(ab[3])
        torch.cuda.synchronize()

        logging.info(f'unmerge rank {r} time cost {time.time() - st :.6f} seconds')




   