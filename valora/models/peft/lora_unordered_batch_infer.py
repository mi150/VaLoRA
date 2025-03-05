import numpy as np
import torch
import torch.nn as nn
from typing import final
import logging
from valora.common.infer_utils import init_bloc
from valora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from valora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from valora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from valora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from valora.utils.infer_utils import mark_cost_time
from valora.utils.infer_utils import calculate_time, mark_start, mark_end
from valora._kernels import dispatch_bgmv
from atmm_ops import dispatch_bgmv as dispatch_sgmm
from valora.server.router.req_queue import rank_counts, tmp_d, no_lora_req
# from punica_copy import dispatch_bgmv as dispatch_sgmm
# from .test_sgmm_value import y_ptr, x_ptr, k_ptr, v_ptr, o_ptr, s_0, s_1
from valora.common.configs.config import setting
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark=False
import time

shape_to_config = {
    (1, 256, 4096, 32): (64, 128, 64, 32, 64),
    (1, 512, 4096, 32): (128, 32, 64, 32, 32),
    (1, 1024, 4096, 32): (64, 32, 64, 64, 32),
    (1, 2048, 4096, 32): (128, 32, 64, 32, 32),
    (1, 256, 4096, 64): (64, 64, 32, 64, 32),
    (1, 1024, 4096, 64): (128, 32, 32, 32, 32),
    (1, 2048, 4096, 64): (32, 32, 64, 32, 32),
    (1, 256, 4096, 128): (128, 32, 64, 32, 32),
    (1, 512, 4096, 128): (32, 128, 64, 32, 32),
    (1, 1024, 4096, 128): (128, 32, 64, 32, 32),
    (1, 2048, 4096, 128): (64, 64, 32, 32, 32),
    (1, 256, 4096, 32): (32, 32, 64, 32, 32),
    (1, 512, 4096, 32): (64, 64, 64, 32, 32),
    (1, 1024, 4096, 32): (64, 32, 64, 64, 32),
    (1, 2048, 4096, 32): (128, 32, 64, 32, 32),
    (1, 256, 4096, 64): (64, 32, 64, 32, 32),
    (1, 512, 4096, 64): (32, 32, 64, 32, 32),
    (1, 1024, 4096, 64): (128, 32, 32, 32, 32),
    (1, 2048, 4096, 64): (32, 32, 64, 32, 32),
    (1, 256, 4096, 128): (128, 32, 64, 32, 32),
    (1, 512, 4096, 128): (64, 32, 64, 64, 32),
    (1, 1024, 4096, 128): (32, 64, 64, 32, 64),
    (1, 2048, 4096, 128): (128, 32, 64, 32, 32),
    (1, 256, 4096, 32): (64, 64, 32, 32, 64),
    (1, 512, 4096, 32): (64, 64, 32, 64, 32),
    (1, 1024, 4096, 32): (32, 64, 64, 32, 64),
    (1, 2048, 4096, 32): (64, 128, 32, 64, 32),
    (1, 256, 4096, 64): (32, 32, 64, 32, 32),
    (1, 512, 4096, 64): (64, 64, 32, 64, 32),
    (1, 1024, 4096, 64): (32, 32, 64, 32, 32),
    (1, 2048, 4096, 64): (64, 128, 32, 32, 64),
    (1, 256, 4096, 128): (32, 32, 64, 32, 32),
    (1, 512, 4096, 128): (128, 32, 64, 32, 32),
    (1, 1024, 4096, 128): (64, 32, 64, 64, 32),
    (1, 2048, 4096, 128): (128, 32, 64, 32, 32),
    (1, 256, 4096, 32): (64, 32, 64, 32, 32),
    (1, 512, 4096, 32): (128, 32, 64, 32, 32),
    (1, 1024, 4096, 32): (64, 32, 64, 64, 32),
    (1, 2048, 4096, 32): (64, 32, 64, 64, 32),
    (1, 256, 4096, 64): (32, 32, 64, 32, 32),
    (1, 512, 4096, 64): (128, 32, 64, 32, 32),
    (1, 1024, 4096, 64): (64, 32, 64, 64, 32),
    (1, 2048, 4096, 64): (32, 32, 64, 32, 32),
    (1, 256, 4096, 128): (128, 32, 64, 64, 32),
    (1, 512, 4096, 128): (64, 32, 64, 32, 32),
    (1, 1024, 4096, 128): (32, 32, 64, 32, 32),
    (1, 2048, 4096, 128): (128, 32, 64, 32, 32),
    (1, 256, 4096, 32): (32, 64, 64, 32, 64),
    (1, 512, 4096, 32): (128, 32, 64, 32, 32),
    (1, 1024, 4096, 32): (64, 64, 32, 64, 32),
    (1, 256, 4096, 64): (32, 32, 64, 32, 32),
    (1, 512, 4096, 64): (128, 32, 32, 32, 32),
    (1, 1024, 4096, 64): (64, 32, 64, 64, 32),
    (1, 256, 4096, 128): (64, 32, 64, 64, 32),
    (1, 512, 4096, 128): (128, 32, 32, 32, 32),
    (1, 1024, 4096, 128): (32, 32, 64, 32, 32),
    (4, 256, 4096, 32): (64, 32, 64, 64, 32),
    (4, 512, 4096, 32): (128, 32, 64, 32, 32),
    (4, 1024, 4096, 32): (64, 32, 64, 32, 32),
    (4, 2048, 4096, 32): (128, 32, 64, 32, 32),
    (4, 256, 4096, 64): (64, 32, 64, 64, 32),
    (4, 512, 4096, 64): (32, 32, 64, 32, 32),
    (4, 1024, 4096, 64): (64, 64, 64, 32, 32),
    (4, 2048, 4096, 64): (128, 32, 64, 32, 32),
    (4, 256, 4096, 128): (64, 32, 64, 64, 32),
    (4, 512, 4096, 128): (32, 64, 64, 32, 64),
    (4, 1024, 4096, 128): (64, 64, 64, 32, 32),
    (4, 2048, 4096, 128): (64, 128, 64, 64, 32),
    (8, 256, 4096, 32): (64, 32, 64, 64, 32),
    (8, 512, 4096, 32): (32, 32, 64, 32, 32),
    (8, 1024, 4096, 32): (128, 32, 64, 32, 32),
    (8, 2048, 4096, 32): (64, 64, 64, 64, 64),
    (8, 256, 4096, 64): (64, 64, 64, 64, 32),
    (8, 512, 4096, 64): (32, 32, 64, 32, 32),
    (8, 1024, 4096, 64): (64, 32, 64, 32, 32),
    (8, 2048, 4096, 64): (128, 32, 32, 64, 32),
    (8, 256, 4096, 128): (64, 32, 64, 32, 32),
    (8, 512, 4096, 128): (128, 32, 64, 32, 32),
    (8, 1024, 4096, 128): (128, 128, 64, 32, 64),
    (8, 2048, 4096, 128): (64, 128, 64, 64, 32),
    (16, 256, 4096, 32): (128, 32, 64, 32, 32),
    (16, 512, 4096, 32): (64, 32, 64, 64, 32),
    (16, 1024, 4096, 32): (128, 32, 64, 32, 32),
    (16, 2048, 4096, 32): (128, 64, 32, 32, 64),
    (16, 256, 4096, 64): (64, 32, 64, 32, 32),
    (16, 512, 4096, 64): (128, 32, 64, 32, 32),
    (16, 1024, 4096, 64): (64, 64, 64, 32, 32),
    (16, 2048, 4096, 64): (64, 128, 64, 64, 32),
    (16, 256, 4096, 128): (128, 128, 64, 128, 32),
    (16, 512, 4096, 128): (128, 64, 32, 32, 64),
    (16, 1024, 4096, 128): (128, 128, 32, 32, 128),
    (16, 2048, 4096, 128): (128, 128, 32, 64, 64),
    (32, 256, 4096, 32): (64, 64, 64, 32, 32),
    (32, 512, 4096, 32): (128, 32, 64, 32, 32),
    (32, 1024, 4096, 32): (64, 32, 32, 32, 32),
    (32, 256, 4096, 64): (64, 64, 64, 32, 32),
    (32, 512, 4096, 64): (128, 32, 32, 64, 32),
    (32, 1024, 4096, 64): (128, 64, 32, 32, 64),
    (32, 256, 4096, 128): (64, 64, 64, 32, 32),
    (32, 512, 4096, 128): (64, 128, 32, 32, 64),
    (32, 1024, 4096, 128): (64, 128, 64, 32, 32),
    (4, 256, 32, 4096): (64, 128, 64, 32, 32),
    (4, 512, 32, 4096): (128, 128, 32, 32, 128),
    (4, 1024, 32, 4096): (128, 64, 32, 32, 32),
    (4, 2048, 32, 4096): (128, 128, 32, 32, 128),
    (4, 256, 64, 4096): (128, 128, 32, 64, 64),
    (4, 512, 64, 4096): (128, 128, 32, 32, 128),
    (4, 1024, 64, 4096): (128, 128, 32, 32, 128),
    (4, 2048, 64, 4096): (64, 128, 32, 64, 32),
    (4, 256, 128, 4096): (128, 128, 32, 32, 128),
    (4, 512, 128, 4096): (128, 128, 32, 32, 128),
    (4, 1024, 128, 4096): (64, 128, 64, 32, 32),
    (4, 2048, 128, 4096): (128, 128, 32, 32, 128),
    (8, 256, 32, 4096): (64, 128, 32, 32, 64),
    (8, 512, 32, 4096): (64, 128, 64, 32, 32),
    (8, 1024, 32, 4096): (128, 128, 32, 32, 128),
    (8, 2048, 32, 4096): (64, 128, 64, 32, 64),
    (8, 256, 64, 4096): (128, 128, 32, 32, 128),
    (8, 512, 64, 4096): (128, 64, 32, 32, 64),
    (8, 1024, 64, 4096): (128, 128, 32, 32, 128),
    (8, 2048, 64, 4096): (64, 128, 64, 32, 64),
    (8, 256, 128, 4096): (128, 64, 64, 32, 32),
    (8, 512, 128, 4096): (64, 128, 32, 32, 64),
    (8, 1024, 128, 4096): (128, 64, 32, 32, 32),
    (8, 2048, 128, 4096): (128, 64, 64, 32, 64),
    (16, 256, 32, 4096): (64, 128, 32, 32, 64),
    (16, 512, 32, 4096): (128, 32, 32, 64, 32),
    (16, 1024, 32, 4096): (32, 128, 32, 32, 128),
    (16, 2048, 32, 4096): (32, 128, 32, 32, 128),
    (16, 256, 64, 4096): (64, 128, 32, 32, 64),
    (16, 512, 64, 4096): (128, 64, 32, 32, 64),
    (16, 1024, 64, 4096): (32, 128, 32, 32, 128),
    (16, 2048, 64, 4096): (64, 32, 32, 32, 32),
    (16, 256, 128, 4096): (128, 128, 32, 32, 128),
    (16, 512, 128, 4096): (128, 32, 32, 64, 32),
    (16, 1024, 128, 4096): (32, 64, 32, 32, 32),
    (16, 2048, 128, 4096): (128, 128, 64, 32, 64),
    (32, 256, 32, 4096): (64, 128, 32, 32, 64),
    (32, 512, 32, 4096): (128, 32, 32, 64, 32),
    (32, 1024, 32, 4096): (128, 128, 64, 32, 128),
    (32, 256, 64, 4096): (64, 128, 32, 64, 32),
    (32, 512, 64, 4096): (128, 32, 32, 64, 32),
    (32, 1024, 64, 4096): (128, 32, 32, 64, 32),
    (32, 256, 128, 4096): (64, 128, 32, 64, 32),
    (32, 512, 128, 4096): (128, 128, 64, 64, 64),
    (32, 1024, 128, 4096): (128, 128, 32, 64, 64),
}



class LoraUnorderedBatchInfer:

    def __init__(self, base_model, adapters, infer_adapter=None, num_problems=(None, None)):
        self.base_model = base_model

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)

        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")

        # dLoRA
        self.scheduler = "valora"
        self.num_problems = num_problems[0]
        self.delora_index = num_problems[1][0]
        self.delora_tk_index = num_problems[1][1]

        # if self.scheduler == "dlora":
        # SGMM
        self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y = 64, 32, 32, 32, 32
        self.tb_x_b, self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b = 64, 64, 32, 64, 64
        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            if isinstance(infer_adapter, NaiveInferAdapter):
                self.key_buffer = infer_adapter.key_buffer
                self.value_buffer = infer_adapter.value_buffer


            else:
                self.key_buffer = infer_adapter.mem_manager.key_buffer
                self.value_buffer = infer_adapter.mem_manager.value_buffer
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                # idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx


        self.kv_embed_dim = base_model.tp_k_head_num_ * base_model.head_dim_

    def get_config_a(self, shape):
        t1 = time.time()
        if self.scheduler in ["strawman", "dlora"]:
            self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y = 64, 32, 32, 32, 32
            return
        if shape in shape_to_config:
            (self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y) = shape_to_config[shape]
        else:
            self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y = 64, 32, 32, 32, 32
        print("Part A uses config:", self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
        logging.info(
            f'get config: {time.time() - t1:.6f} seconds')
    def get_config_b(self, shape):
        if self.scheduler in ["strawman", "dlora"]:
            self.tb_x_b, self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b = 64, 64, 32, 64, 32
            return
        if shape in shape_to_config:
            (self.tb_x_b, self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b) = shape_to_config[shape]
        else:
            self.tb_x_b, self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b = 64, 64, 32, 64, 64
        print("Part B uses config:", self.tb_x_b, self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
    @torch.no_grad()
    def forward(
            self,
            batch_size,  # number of request
            total_token_num,
            max_len_in_batch,
            input_ids,  # 1D input tensor
            b_loc,  # mapping to memory pool
            b_start_loc,  # the start index of each request
            b_seq_len,  # the current length of each request
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            no_lora_copy=False,
            output_counts=None, lora_ids=None, start_ids=None):
        # print(self.scheduler)
        # Notice that batch_lora only support decoding
        self.output_counts = output_counts
        self.lora_ids = lora_ids
        self.start_ids = start_ids
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)
        b_seq_len_copy = b_seq_len
        self.delta = []
        self.delora_delta = []
        self.max_b_seq_len = torch.max(b_seq_len).item()
        decode_start_time = time.time()
        # reset req_bins/batch_req_bins
        if self.scheduler == "ours":
            t1 = time.time()
            if no_lora_compute and self.num_problems > 1:
                # self.req_bins = self.req_bins[self.delora_index:]
                sub_tensor = self.req_bins[self.delora_index:]
                repeated_tensor = sub_tensor.repeat(2)
                half_length = repeated_tensor.size(0) // 2
                repeated_tensor[half_length:] = self.req_bins[0]
                self.req_bins = repeated_tensor
                b_seq_len_copy = b_seq_len[self.delora_index:]
                b_seq_len_copy = b_seq_len_copy.repeat(2)
            
            else:
                if self.delora_index > 0:
                    self.num_problems -= 1
                self.req_bins = self.req_bins[self.delora_index:]
                b_seq_len_copy = b_seq_len[self.delora_index:]
                if len(self.req_bins) == 0 or self.num_problems == 1:
                    no_lora_compute = True
                    self.num_problems = 1
            logging.info(
                f'batch infer 276 if: {time.time() - t1:.6f} seconds')
        base_layer_infer = self.base_model.layers_infer[0]

        if is_prefill:
            # dlora
            if self.scheduler == "ours":
                self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len_copy)
            else:
                self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len)
            if self.scheduler == "dlora":
                # model_mapping = self.batch_req_bins.clone().detach().view(-1, 1)

                adapter_tensor = torch.IntTensor(len(self.batch_req_bins), len(self.req_bins)).to('cuda')
                adapter_tensor.zero_()
                # adapter_tensor.scatter_(1, model_mapping, 0)
                self.adapter_mapping = adapter_tensor.cuda()
                # print("adapter_mapping", self.adapter_mapping)
            self.de_length = len(self.batch_req_bins) // 2
            if not no_lora_compute or (no_lora_compute and self.num_problems > 1 and self.scheduler=="ours"):
                for i in range(3):
                    self.delta.append(
                        torch.zeros((len(self.batch_req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))
                    self.delora_delta.append(
                        torch.zeros((len(self.batch_req_bins), base_layer_infer.embed_dim_), dtype=torch.float16, device="cuda"))
                # if self.scheduler == "ours":
                self.get_config_a((len(self.req_bins), b_seq_len_copy[0].item(), base_layer_infer.embed_dim_, self.max_lora_dim))
                self.get_config_b((len(self.req_bins), b_seq_len_copy[0].item(), self.max_lora_dim, len(self.batch_req_bins)))
            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids,
                                 b_loc, b_start_loc, b_seq_len, no_lora_compute)
        else:
            if self.scheduler in ["ours"]:
                no_lora_compute = False
            if self.scheduler == "dlora":
                # model_mapping = self.req_bins.clone().detach().view(-1, 1)
                adapter_tensor = torch.IntTensor(len(self.req_bins), len(self.req_bins)).to('cuda')
                adapter_tensor.zero_()
                # adapter_tensor.scatter_(1, model_mapping, 0)
                self.adapter_mapping = adapter_tensor.cuda()
                # print("adapter_mapping",self.adapter_mapping)
            if not no_lora_compute or (no_lora_compute and self.num_problems > 1 and self.scheduler=="ours"):
                self.de_length = len(self.req_bins) // 2
                for i in range(3):
                    self.delta.append(torch.zeros((len(self.req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))
                    self.delora_delta.append(torch.zeros((len(self.req_bins), base_layer_infer.embed_dim_), dtype=torch.float16, device="cuda"))
                self.get_config_a((len(self.req_bins), b_seq_len_copy[0].item(), base_layer_infer.embed_dim_, self.max_lora_dim))
                self.get_config_b((len(self.req_bins), b_seq_len_copy[0].item(), self.max_lora_dim, len(self.req_bins)))

            result = self._decode(batch_size, total_token_num, max_len_in_batch,
                                  input_ids,
                                  b_loc, b_start_loc, b_seq_len,
                                  no_lora_compute, no_lora_copy)

            decode_end_time = time.time()
            decode_duration = decode_end_time - decode_start_time


            return result

    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_start_loc, b_seq_len, no_lora_compute=False):
        # print("no_lora_compute", no_lora_compute)
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch

        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        b_seq_len_numpy = b_seq_len.cpu().numpy()

        position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                                        for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
            self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
            self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.mem_manager = self.base_model.mem_manager

        infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)

        start_time = time.time()
        infer_state.prefill_key_buffer = torch.empty(
            (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
            dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
            (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
            dtype=torch.float16, device="cuda")
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)

        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute)

        return predict_logics

    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids,
                b_loc, b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch

        start_time = time.time()
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        start_time = time.time()
        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.base_model.mem_manager

        start_time = time.time()
        alloc_mem = self.base_model.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.base_model.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        start_time = time.time()
        infer_state.init_some_extra_state(self.base_model, batch_size, total_token_num, max_len_in_batch,
                                          input_ids, b_loc, b_start_loc, b_seq_len, False)

        predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy)
        # logging.info(f'Step 5 (Token forward pass) duration: {time.time() - start_time:.6f} seconds')
        return predict_logics

    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False):
        cuda_input_ids = input_ids
        t1 = time.time()
        input_embs = self.base_model.pre_infer.context_forward(
            cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        t2 = time.time()
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
        t3 = time.time()

        predict_logics = self.base_model.post_infer.token_forward(
            input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        t4 = time.time()
        pre_time = t2 - t1
        infer_time = t3 - t2
        post_time = t4 - t3
        # 打印和记录时间信息
        # log_message = f"pre: {pre_time:.6f} infer: {infer_time:.6f} post: {post_time:.6f}"
        # print(log_message)
        # logging.info(log_message)
        return predict_logics

    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
            cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
        predict_logics = self.base_model.post_infer.token_forward(
            input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics

    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):

        input_embs = self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)

        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]

        layer_infer._context_ffn(input_embs, infer_state, layer_weight)
        return input_embs

    @final
    # @calculate_time(show=True, min_cost_ms=0)
    def _lora_token_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        input_embs = self._lora_token_attention(layer_id, input_embs, infer_state, no_lora_compute, no_lora_copy)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # mark_start("token_ffn")
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        # mark_end("token_ffn")
        return input_embs

    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        torch.cuda.empty_cache()
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        input1 = None
        torch.cuda.synchronize()
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        torch.cuda.synchronize()
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        torch.cuda.synchronize()
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        # print("shape error?", self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)

        # try:
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        # except:
        #     input_emb = torch.ones_like(input_embs).cuda()
        #     print("try except")
        #     # del input_embs
        #     # torch.cuda.empty_cache()
        #     # torch.cuda.synchronize()
        #     return input_emb
        # o = None
        torch.cuda.synchronize()
        return input_embs

    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # try:
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        # except:
        #     检查是否有 NaN 或 Inf 值
        #     has_nan_or_inf = torch.isnan(o).any() or torch.isinf(o).any()
        #     # print("try except")
        #     # input_emb = torch.ones_like(input_embs).cuda()
        #     del input_embs
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()
            # return input_emb
        # o = None
        return input_embs

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False,
                            no_lora_copy=False) -> torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
        # @TODO: fix me, filter requests querying only base model
        # assert (len(q) == len(self.req_bins))

        if not no_lora_compute:
            # mark_start("get_q")
            if self.scheduler in ["strawman", "dlora"]:
                if self.scheduler != "ours":
                    if self.scheduler == "strawman":
                        # punica
                        for i in range(1):
                            delta_qA = self.delta[1]
                            dispatch_sgmm(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                        self.key_buffer[layer_id],
                                        self.infer_adapter.a_start, self.infer_adapter.a_len,
                                        self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling,
                                        self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                        self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                            dispatch_sgmm(q,
                                        delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                        self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                        self.req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                        rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        for i in range(1):
                            result = torch.einsum('bk, bi, ikr, ird->bd',
                                                  input_embs.view(-1, base_layer_infer.embed_dim_)[:len(self.req_bins)],
                                                  self.adapter_mapping,
                                                  self.key_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                        max(self.infer_adapter.a_len)//4),
                                                  self.value_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                        base_layer_infer.embed_dim_)
                                                  )
                            torch.cuda.synchronize()
                        # logging.info(f'dlora_duration: {time.time() - sgmm_start_time:.6f} seconds')
                else:
                    delta_qA = self.delta[0]
                    # print("shape error*", self.output_counts, self.start_ids, input_embs.view(-1, base_layer_infer.embed_dim_).shape, self.delora_index, self.num_problems)
                    sgmm_start_time = time.time()
                    dispatch_sgmm(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.req_bins, 0, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids,
                                  tmp_d, self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                                  self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(q[self.delora_index:], delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.req_bins, 0, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
                    torch.cuda.synchronize()
                    # logging.info(f'sgmm_decode_duration: {time.time() - sgmm_start_time:.6f} seconds')
            else:
                delta_qA = self.delta[0]
                dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_),
                              self.key_buffer[layer_id],
                              self.infer_adapter.a_start, self.infer_adapter.a_len,
                              self.infer_adapter.a_loc, self.req_bins, 0, self.infer_adapter.a_scaling)
                dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None
            # mark_end("get_q")
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours" and False:
            # deLoRA
            delta_qA = self.delta[0]
            qA_delora = self.delora_delta[0]
            sub_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:]
            expanded_tensor = torch.cat([sub_tensor, sub_tensor], dim=0)

            dispatch_sgmm(delta_qA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.req_bins, 0,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(qA_delora, delta_qA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 0, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            q[self.delora_index:] += qA_delora[:self.de_length] - qA_delora[self.de_length:]

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_k")
            if self.scheduler in ["strawman", "dlora"]:
                if self.scheduler != "ours":
                    if self.scheduler in ["strawman"]:
                        # punica
                        for i in range(1):
                            delta_kA = self.delta[1]
                            dispatch_sgmm(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                        self.key_buffer[layer_id],
                                        self.infer_adapter.a_start, self.infer_adapter.a_len,
                                        self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling,
                                        self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                        self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                            dispatch_sgmm(cache_k.view(-1, base_layer_infer.embed_dim_),
                                        delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                        self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                        self.req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                        rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        for i in range(1):
                            input = input_embs.view(-1, base_layer_infer.embed_dim_)[:len(self.req_bins)]
                            key = self.key_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4].view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                   max(self.infer_adapter.a_len)//4)
                            value = self.value_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4].view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                        base_layer_infer.embed_dim_)
                            torch.cuda.synchronize()
                            result = torch.einsum('bk, bi, ikr, ird->bd',
                                                  input,
                                                  self.adapter_mapping,
                                                  key, value
                                                  )
                            torch.cuda.synchronize()
                        # logging.info(f'dlora_duration: {time.time() - sgmm_start_time:.6f} seconds {input_embs.view(-1, base_layer_infer.embed_dim_).shape}')
                else:

                    delta_kA = self.delta[1]
                    # print("&&&memory error", self.output_counts, self.start_ids, input_embs.view(-1, base_layer_infer.embed_dim_).shape,
                    #       self.delora_index)

                    dispatch_sgmm(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_index:],
                                  delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)

                    # torch.cuda.synchronize()
            else:
                delta_kA = self.delta[1]
                dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_),
                              self.key_buffer[layer_id],
                              self.infer_adapter.a_start, self.infer_adapter.a_len,
                              self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                              delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None
            # mark_end("get_k")
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours" and False:
            # deLoRA
            delta_kA = self.delta[1]
            kA_delora = self.delora_delta[1]
            sub_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:]
            expanded_tensor = torch.cat([sub_tensor, sub_tensor], dim=0)

            dispatch_sgmm(delta_kA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.req_bins, 1,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(kA_delora,
                          delta_kA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 1, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_index:] +=\
                kA_delora[:self.de_length] - kA_delora[self.de_length:]

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_v")
            if self.scheduler in ["strawman", "dlora", "ours"]:
                if self.scheduler != "ours":
                    if self.scheduler in ["strawman"]:
                        # punica
                        for i in range(1):
                            delta_vA = self.delta[2]

                            dispatch_sgmm(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                        self.key_buffer[layer_id],
                                        self.infer_adapter.a_start, self.infer_adapter.a_len,
                                        self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling,
                                        self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                        self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                            dispatch_sgmm(cache_v.view(-1, base_layer_infer.embed_dim_),
                                        delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                        self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                        self.req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                        rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        for i in range(1):
                            result = torch.einsum('bk, bi, ikr, ird->bd',
                                                  input_embs.view(-1, base_layer_infer.embed_dim_)[:len(self.req_bins)],
                                                  self.adapter_mapping,
                                                  self.key_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len) // 4]
                                                  .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                        max(self.infer_adapter.a_len)//4),
                                                  self.value_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                        base_layer_infer.embed_dim_)
                                                  )
                else:
                    delta_vA = self.delta[2]

                    dispatch_sgmm(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.req_bins, 2, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_index:],
                                  delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.req_bins, 2, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
                    torch.cuda.synchronize()
            else:
                delta_vA = self.delta[2]
                dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:],
                              self.key_buffer[layer_id],
                              self.infer_adapter.a_start, self.infer_adapter.a_len,
                              self.infer_adapter.a_loc, self.req_bins, 2, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                              delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.req_bins, 2, self.infer_adapter.a_scaling)

        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours" and False:
            # deLoRA
            delta_vA = self.delta[2]
            vA_delora = self.delora_delta[2]
            sub_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_index:]
            expanded_tensor = torch.cat([sub_tensor, sub_tensor], dim=0)

            dispatch_sgmm(delta_vA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.req_bins, 2,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(vA_delora,
                          delta_vA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 2, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            torch.cuda.synchronize()
            cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_index:] += \
                vA_delora[:self.de_length] - vA_delora[self.de_length:]
        torch.cuda.synchronize()
        return q

    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False) -> torch.Tensor:
        mm_start_time = time.time()
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        mm_start_time = time.time()


        combined_output = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.qkv_weight_)
        logging.info(f'shape: {input_embs.view(-1, base_layer_infer.embed_dim_).shape} ')


        q, k_output, v_output = combined_output.split([base_model.tp_k_head_num_ * base_model.head_dim_,
                                                       base_model.tp_k_head_num_ * base_model.head_dim_,
                                                       base_model.tp_k_head_num_ * base_model.head_dim_], dim=1)
        cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).copy_(k_output).contiguous()
        cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).copy_(v_output).contiguous()
        q = q.contiguous()
        torch.cuda.synchronize()
        # print(f'mm_time: {time.time() - mm_start_time:.6f} seconds')

        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            delta_qA = self.delta[0]
            delta_kA = self.delta[1]
            delta_vA = self.delta[2]
            if self.scheduler in ["strawman", "dlora", "ours"]:
                if self.scheduler != "ours":
                    if self.scheduler in ["strawman"]:
                        # punica
                        for i in range(3):
                            dispatch_sgmm(self.delta[i], input_embs.view(-1, base_layer_infer.embed_dim_),
                                          self.key_buffer[layer_id],
                                          self.infer_adapter.a_start, self.infer_adapter.a_len,
                                          self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling,
                                          self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                          self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                            dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                          self.delta[i], self.value_buffer[layer_id], self.infer_adapter.a_start,
                                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                          self.batch_req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                          rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems,64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        for i in range(3):
                            # print(input_embs.view(-1, base_layer_infer.embed_dim_).shape, len(self.req_bins), base_layer_infer.embed_dim_,
                            #                             max(self.infer_adapter.a_len), self.key_buffer[layer_id][
                            #                       :len(self.req_bins) * max(self.infer_adapter.a_len)//4].shape)
                            result = torch.einsum('bk, bi, ikr, ird->bd',
                                                  input_embs.view(-1, base_layer_infer.embed_dim_),
                                                  self.adapter_mapping,
                                                  self.key_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                        max(self.infer_adapter.a_len)//4),
                                                  self.value_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                        base_layer_infer.embed_dim_)
                                                  )
                            # torch.cuda.synchronize()
                        # logging.info(f'dlora_duration: {time.time() - sgmm_start_time:.6f} seconds')
                else:
                    # torch.cuda.synchronize()
                    # sgmm_start_time = time.time()
                    # print("shape error@",delta_qA.shape, input_embs.view(-1, base_layer_infer.embed_dim_).shape,self.delora_tk_index
                    #       ,self.delora_index,len(self.req_bins))
                    dispatch_sgmm(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)

                    dispatch_sgmm(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)

                    dispatch_sgmm(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(q[self.delora_tk_index:], delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 0, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)

                    dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:],
                                  delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 1, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)

                    dispatch_sgmm(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:],
                                  delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 2, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
                    torch.cuda.synchronize()

        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours":
            # deLoRA
            delta_qA = self.delta[0]
            qA_delora = self.delora_delta[0]
            delta_kA = self.delta[1]
            kA_delora = self.delora_delta[1]
            delta_vA = self.delta[2]
            vA_delora = self.delora_delta[2]
            sub_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:]
            expanded_tensor = torch.cat([sub_tensor, sub_tensor], dim=0)
            dispatch_sgmm(delta_qA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins, 0,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:],
                          self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(qA_delora, delta_qA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins, 0, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:],
                          self.start_ids[1:], tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            q[self.delora_tk_index:] += qA_delora[:self.de_length] - qA_delora[self.de_length:]

            dispatch_sgmm(delta_kA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins, 1,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:],
                          self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(kA_delora,
                          delta_kA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins, 1, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:],
                          self.start_ids[1:], tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:] +=\
                kA_delora[:self.de_length] - kA_delora[self.de_length:]

            dispatch_sgmm(delta_vA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins, 2,
                          self.infer_adapter.a_scaling, self.output_counts[1:], rank_counts[1:],
                          self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(vA_delora,
                          delta_vA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins, 2, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:],
                          self.start_ids[1:], tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            torch.cuda.synchronize()
            cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:] += \
                vA_delora[:self.de_length] - vA_delora[self.de_length:]

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.cuda.synchronize()

        logging.info(f'all_compute_time: {time.time() - mm_start_time:.6f} seconds')
        return q

    def _lora_get_qkv_copy(self, layer_id, input_embs, cache_k, cache_v, infer_state,
                           no_lora_compute=False) -> torch.Tensor:
        mm_start_time = time.time()
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)

        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        assert (len(q) == len(self.batch_req_bins))
        torch.cuda.synchronize()
        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            delta_qA = self.delta[0]
            if self.scheduler in ["strawman", "dlora", "ours"]:
                if self.scheduler != "ours":
                    # print(input_embs.view(-1, base_layer_infer.embed_dim_).shape, len(self.req_bins), base_layer_infer.embed_dim_,
                    #                             max(self.infer_adapter.a_len))
                    result = torch.einsum('bk, bi, ikr, ird->bd',
                                          input_embs.view(-1, base_layer_infer.embed_dim_),
                                          self.adapter_mapping,
                                          self.key_buffer[layer_id][
                                          :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                          .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                max(self.infer_adapter.a_len)//4),
                                          self.value_buffer[layer_id][
                                          :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                          .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                base_layer_infer.embed_dim_)
                                          )
                    torch.cuda.synchronize()
                else:
                    dispatch_sgmm(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 0, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x, self.tb_y,
                                  self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    # torch.cuda.synchronize()


            else:
                if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
                    # if 1 == 0:

                    lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_),
                                             self.key_buffer[layer_id].view(-1, self.kv_embed_dim),
                                             delta_qA, self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                             self.infer_adapter.a_len, infer_state.b_start_loc,
                                             infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_,
                                             0, self.max_lora_dim, self.max_b_seq_len)
                    torch.cuda.synchronize()
                    lora_get_qkvo_fwd_expand(delta_qA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim),
                                             q, self.infer_adapter.a_scaling,
                                             self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                             self.infer_adapter.a_len, infer_state.b_start_loc,
                                             infer_state.b_seq_len, self.req_bins, self.kv_embed_dim,
                                             0, self.max_lora_dim, self.max_b_seq_len)
                else:
                    dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling)
                    torch.cuda.synchronize()
                    dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours":
            # deLoRA
            delta_qA = self.delta[0]
            dispatch_sgmm(delta_qA[self.delora_tk_index:],
                          input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins[self.delora_tk_index:], 0,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:],
                          self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(q[self.delora_tk_index:], delta_qA[self.delora_tk_index:], self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins[self.delora_tk_index:], 0, self.infer_adapter.a_scaling, self.output_counts,
                          rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems - 1, self.tb_x, self.tb_y,
                          self.tb_z, self.wp_x, self.wp_y, self.tb_z)
            # delora
            torch.cuda.synchronize()
            dispatch_sgmm(delta_qA[self.delora_tk_index:],
                          input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins[:-self.delora_tk_index], 0,
                          self.infer_adapter.a_scaling * (-1), self.output_counts, rank_counts, self.lora_ids, self.start_ids,
                          tmp_d, 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(q[self.delora_tk_index:], delta_qA[self.delora_tk_index:], self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins[:-self.delora_tk_index], 0, self.infer_adapter.a_scaling * (-1),
                          self.output_counts,
                          rank_counts, self.lora_ids, self.start_ids, tmp_d, 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x,
                          self.wp_y, self.tb_z)

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)

        # k (S, H)
        # torch.cuda.synchronize()
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        torch.cuda.synchronize()
        # print("mm time:",time.time() - mm_start_time)

        # print("print:", no_lora_compute, self.num_problems)
        if not no_lora_compute:
            delta_kA = self.delta[1]
            if self.scheduler in ["strawman", "dlora", "ours"]:
                if self.scheduler != "ours":
                    if self.scheduler == "strawman":
                        # punica
                        dispatch_sgmm(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                      self.key_buffer[layer_id],
                                      self.infer_adapter.a_start, self.infer_adapter.a_len,
                                      self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling,
                                      self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                      self.num_problems, 64, 64, 32, 64, 32, 32)
                        torch.cuda.synchronize()
                        dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                      delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                      self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                      self.batch_req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                      rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, 64, 64, 32, 64, 32, 32)
                        torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        result = torch.einsum('bk, bi, ikr, ird->bd',
                                              input_embs.view(-1, base_layer_infer.embed_dim_),
                                              self.adapter_mapping,
                                              self.key_buffer[layer_id][
                                              :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                              .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                    max(self.infer_adapter.a_len)//4),
                                              self.value_buffer[layer_id][
                                              :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                              .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                    base_layer_infer.embed_dim_)
                                              )
                        # torch.cuda.synchronize()
                        # logging.info(f'dlora_duration: {time.time() - sgmm_start_time:.6f} seconds')
                else:

                    sgmm_start_time = time.time()
                    dispatch_sgmm(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                  delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x, self.tb_y,
                                  self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    # logging.info(f'sgmm_duration: {time.time() - sgmm_start_time:.6f} seconds')
            else:
                if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
                    # if 1 == 0:
                    #     torch.cuda.synchronize()
                    lora_get_qkvo_fwd_shrink_start_time = time.time()
                    lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_),
                                             self.key_buffer[layer_id].view(-1, self.kv_embed_dim),
                                             delta_kA, self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                             self.infer_adapter.a_len, infer_state.b_start_loc,
                                             infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_,
                                             1, self.max_lora_dim, self.max_b_seq_len)
                    torch.cuda.synchronize()
                    # logging.info(
                    #     f'lora_get_qkvo_fwd_shrink_duration: {time.time() - lora_get_qkvo_fwd_shrink_start_time:.6f} seconds')

                    lora_get_qkvo_fwd_expand_start_time = time.time()
                    lora_get_qkvo_fwd_expand(delta_kA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim),
                                             cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                             self.infer_adapter.a_scaling,
                                             self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                             self.infer_adapter.a_len, infer_state.b_start_loc,
                                             infer_state.b_seq_len, self.req_bins, self.kv_embed_dim,
                                             1, self.max_lora_dim, self.max_b_seq_len)
                    torch.cuda.synchronize()
                    # logging.info(
                    #     f'lora_get_qkvo_fwd_expand_duration: {time.time() - lora_get_qkvo_fwd_expand_start_time:.6f} seconds')
                    # print("SLORA Cost:", time.time() - lora_get_qkvo_fwd_shrink_start_time, input_embs.shape)

                else:

                    dispatch_bgmv_1_start_time = time.time()
                    dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                    # torch.cuda.synchronize()
                    # logging.info(f'dispatch_bgmv_1_duration: {time.time() - dispatch_bgmv_1_start_time:.6f} seconds')
                    # logging.info(
                    #     f'xA size: {base_model.tp_k_head_num_ * base_model.head_dim_} {self.value_buffer[layer_id].shape, self.infer_adapter.a_start, self.infer_adapter.a_len, self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling} ')
                    dispatch_bgmv_2_start_time = time.time()
                    dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                  delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                    torch.cuda.synchronize()
                    # print("SLORA:",time.time() - dispatch_bgmv_1_start_time)
                    # logging.info(f'dispatch_bgmv_2_duration: {time.time() - dispatch_bgmv_2_start_time:.6f} seconds')
                    # t3 = time.time()
                    # print("SLORA Cost:", t3 - t2)
                # delta_kA = None
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours":
            # deLoRA
            # print(self.delora_index)

            delta_kA = self.delta[0]
            dispatch_sgmm(delta_kA[self.delora_tk_index:],
                          input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins[self.delora_tk_index:], 0,
                          self.infer_adapter.a_scaling, self.output_counts, rank_counts, self.lora_ids, self.start_ids,
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:],
                          delta_kA[self.delora_tk_index:], self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins[self.delora_tk_index:], 0, self.infer_adapter.a_scaling, self.output_counts,
                          rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems - 1, self.tb_x, self.tb_y,
                          self.tb_z, self.wp_x, self.wp_y, self.tb_z)
            # delora
            dispatch_sgmm(delta_kA[self.delora_tk_index:],
                          input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins[:-self.delora_tk_index], 0,
                          self.infer_adapter.a_scaling * (-1), self.output_counts, rank_counts, self.lora_ids, self.start_ids,
                          tmp_d, 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_index:],
                          delta_kA[self.delora_index:], self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins[:-self.delora_index], 0, self.infer_adapter.a_scaling * (-1),
                          self.output_counts,
                          rank_counts, self.lora_ids, self.start_ids, tmp_d, 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x,
                          self.wp_y, self.tb_z)

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)

        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        torch.cuda.synchronize()
        if not no_lora_compute:
            if self.scheduler in ["strawman", "dlora", "ours"]:
                if self.scheduler != "ours":

                    result = torch.einsum('bk, bi, ikr, ird->bd',
                                          input_embs.view(-1, base_layer_infer.embed_dim_),
                                          self.adapter_mapping,
                                          self.key_buffer[layer_id][
                                          :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                          .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                max(self.infer_adapter.a_len)//4),
                                          self.value_buffer[layer_id][
                                          :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                          .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                base_layer_infer.embed_dim_)
                                          )
                    torch.cuda.synchronize()
                else:
                    delta_vA = self.delta[2]
                    dispatch_sgmm(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                  delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 2, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x, self.tb_y,
                                  self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    # torch.cuda.synchronize()

            else:
                delta_vA = self.delta[2]
                if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
                    # if 1 ==0:
                    lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_),
                                             self.key_buffer[layer_id].view(-1, self.kv_embed_dim),
                                             delta_vA, self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                             self.infer_adapter.a_len, infer_state.b_start_loc,
                                             infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_,
                                             2, self.max_lora_dim, self.max_b_seq_len)
                    torch.cuda.synchronize()
                    lora_get_qkvo_fwd_expand(delta_vA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim),
                                             cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_),
                                             self.infer_adapter.a_scaling,
                                             self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                             self.infer_adapter.a_len, infer_state.b_start_loc,
                                             infer_state.b_seq_len, self.req_bins, self.kv_embed_dim,
                                             2, self.max_lora_dim, self.max_b_seq_len)
                else:
                    dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_),
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)
                    # torch.cuda.synchronize()
                    dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                                  delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours":
            # deLoRA
            # print("deLoRA",)
            delta_vA = self.delta[0]
            dispatch_sgmm(delta_vA[self.delora_tk_index:],
                          input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins[self.delora_tk_index:], 0,
                          self.infer_adapter.a_scaling, self.output_counts, rank_counts, self.lora_ids, self.start_ids,
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:],
                          delta_vA[self.delora_tk_index:], self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins[self.delora_tk_index:], 0, self.infer_adapter.a_scaling, self.output_counts,
                          rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems - 1, self.tb_x, self.tb_y,
                          self.tb_z, self.wp_x, self.wp_y, self.tb_z)
            # delora
            dispatch_sgmm(delta_vA[self.delora_tk_index:],
                          input_embs.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins[:-self.delora_tk_index], 0,
                          self.infer_adapter.a_scaling * (-1), self.output_counts, rank_counts, self.lora_ids, self.start_ids,
                          tmp_d, 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[self.delora_tk_index:],
                          delta_vA[self.delora_tk_index:], self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins[:-self.delora_tk_index], 0, self.infer_adapter.a_scaling * (-1),
                          self.output_counts,
                          rank_counts, self.lora_ids, self.start_ids, tmp_d, 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x,
                          self.wp_y, self.tb_z)
        # torch.cuda.synchronize()
        logging.info(f'all_compute_time: {time.time() - mm_start_time:.6f} seconds')
        return q

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False) -> torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # t1 = time.time()
        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.o_weight_)
        # torch.cuda.synchronize()
        # print("MM time:", time.time()-t1)
        # print(self.scheduler)
        if not no_lora_compute:
            # mark_start("get_o")
            if self.scheduler in ["strawman", "dlora"]:
                if self.scheduler != "ours":
                    if self.scheduler == "strawman":
                        # punica
                        for i in range(1):
                            dispatch_sgmm(self.delta[i], input.view(-1, base_layer_infer.embed_dim_),
                                        self.key_buffer[layer_id],
                                        self.infer_adapter.a_start, self.infer_adapter.a_len,
                                        self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling,
                                        self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                        self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                            dispatch_sgmm(o,
                                        self.delta[i], self.value_buffer[layer_id], self.infer_adapter.a_start,
                                        self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                        self.req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                        rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        for i in range(1):
                            # print(input.view(-1, base_layer_infer.embed_dim_).shape,len(self.req_bins))
                            result = torch.einsum('bk, bi, ikr, ird->bd',
                                                  input.view(-1, base_layer_infer.embed_dim_),
                                                  self.adapter_mapping,
                                                  self.key_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                        max(self.infer_adapter.a_len)//4),
                                                  self.value_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                        base_layer_infer.embed_dim_)
                                                  )
                            # torch.cuda.synchronize()
                        # logging.info(f'dlora_duration: {time.time() - sgmm_start_time:.6f} seconds')
                else:
                    delta_oA = self.delta[0]
                    # print("&*^",input.view(-1, base_layer_infer.embed_dim_).shape,self.delora_index,self.output_counts,self.start_ids,self.lora_ids)
                    dispatch_sgmm(delta_oA, input.view(-1, base_layer_infer.embed_dim_)[self.delora_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(o[self.delora_index:], delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.req_bins, 3, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
                    torch.cuda.synchronize()
                return o
            else:
                delta_oA = self.delta[0]
                t1 = time.time()
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_),
                              self.key_buffer[layer_id],
                              self.infer_adapter.a_start, self.infer_adapter.a_len,
                              self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling)
                dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.req_bins, 3, self.infer_adapter.a_scaling)
                torch.cuda.synchronize()
                decode_duration = time.time() - t1
                # logging.info(f'Decode time: {decode_duration:.6f} seconds')
                # delta_oA = None
                # mark_end("get_o")
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours" and False:
            # deLoRA
            delta_oA = self.delta[0]
            oA_delora = self.delora_delta[0]
            sub_tensor = input.view(-1, base_layer_infer.embed_dim_)[self.delora_index:]
            expanded_tensor = torch.cat([sub_tensor, sub_tensor], dim=0)

            dispatch_sgmm(delta_oA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.req_bins, 3,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(oA_delora,
                          delta_oA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 3, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            torch.cuda.synchronize()
            o[self.delora_index:] += oA_delora[:self.de_length] - oA_delora[self.de_length:]
            torch.cuda.synchronize()
        return o

    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False) -> torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.o_weight_)
        # print(self.scheduler)
        torch.cuda.synchronize()
        if not no_lora_compute:
            delta_oA = self.delta[0]

            if self.scheduler in ["strawman", "dlora", "ours"]:
                if self.scheduler != "ours":
                    if self.scheduler in ["strawman"]:
                        # punica
                        for i in range(1):
                            dispatch_sgmm(self.delta[i], input.view(-1, base_layer_infer.embed_dim_),
                                        self.key_buffer[layer_id],
                                        self.infer_adapter.a_start, self.infer_adapter.a_len,
                                        self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling,
                                        self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                        self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                            dispatch_sgmm(o,
                                        self.delta[i], self.value_buffer[layer_id], self.infer_adapter.a_start,
                                        self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                        self.batch_req_bins, 1, self.infer_adapter.a_scaling, self.output_counts,
                                        rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, 64, 64, 32, 64, 32, 32)
                            torch.cuda.synchronize()
                    else:
                        sgmm_start_time = time.time()
                        for i in range(1):
                            result = torch.einsum('bk, bi, ikr, ird->bd',
                                                  input.view(-1, base_layer_infer.embed_dim_),
                                                  self.adapter_mapping,
                                                  self.key_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), base_layer_infer.embed_dim_,
                                                        max(self.infer_adapter.a_len)//4),
                                                  self.value_buffer[layer_id][
                                                  :len(self.req_bins) * max(self.infer_adapter.a_len)//4]
                                                  .view(len(self.req_bins), max(self.infer_adapter.a_len)//4,
                                                        base_layer_infer.embed_dim_)
                                                  )
                            torch.cuda.synchronize()
                        # logging.info(f'dlora_duration: {time.time() - sgmm_start_time:.6f} seconds')
                else:
                    dispatch_sgmm(delta_oA, input.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:],
                                  self.key_buffer[layer_id],
                                  self.infer_adapter.a_start, self.infer_adapter.a_len,
                                  self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling,
                                  self.output_counts, rank_counts, self.lora_ids, self.start_ids, tmp_d,
                                  self.num_problems, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.tb_z)
                    torch.cuda.synchronize()
                    dispatch_sgmm(o[self.delora_tk_index:], delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 3, self.infer_adapter.a_scaling, self.output_counts,
                                  rank_counts, self.lora_ids, self.start_ids, tmp_d, self.num_problems, self.tb_x_b,
                                  self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
                    torch.cuda.synchronize()

                return o
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
                # if 1 == 0:
                t1 = time.time()
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_),
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim),
                                         delta_oA, self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                         self.infer_adapter.a_len, infer_state.b_start_loc,
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_,
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim),
                                         o, self.infer_adapter.a_scaling,
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start,
                                         self.infer_adapter.a_len, infer_state.b_start_loc,
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_,
                                         3, self.max_lora_dim, self.max_b_seq_len)

                torch.cuda.synchronize()

            else:
                t1 = time.time()
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_),
                              self.key_buffer[layer_id],
                              self.infer_adapter.a_start, self.infer_adapter.a_len,
                              self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                torch.cuda.synchronize()
            delta_oA = None
        elif no_lora_compute and self.num_problems > 1 and self.scheduler == "ours":
            # deLoRA
            delta_oA = self.delta[0]
            oA_delora = self.delora_delta[0]
            sub_tensor = input.view(-1, base_layer_infer.embed_dim_)[self.delora_tk_index:]
            expanded_tensor = torch.cat([sub_tensor, sub_tensor], dim=0)

            dispatch_sgmm(delta_oA,
                          expanded_tensor,
                          self.key_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.batch_req_bins, 3,
                          self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y,
                          self.tb_z)
            torch.cuda.synchronize()
            dispatch_sgmm(oA_delora,
                          delta_oA, self.value_buffer[layer_id],
                          self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.batch_req_bins, 3, self.infer_adapter.a_scaling, self.output_counts[1:],
                          rank_counts[1:], self.lora_ids[1:], self.start_ids[1:],
                          tmp_d, self.num_problems - 1, self.tb_x_b,
                          self.tb_y_b, self.tb_z_b, self.wp_x_b, self.wp_y_b, self.tb_z_b)
            torch.cuda.synchronize()
            o[self.delora_tk_index:] += oA_delora[:self.de_length] - oA_delora[self.de_length:]
        torch.cuda.synchronize()
        return o
