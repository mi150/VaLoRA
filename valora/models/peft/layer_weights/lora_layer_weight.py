# import gc
# import torch
# import torch.nn as nn


# class LoraLayerWeight:
#     def __init__(self, layer_num, tp_rank, world_size, lora_config, network_config, data_type=torch.float16,
#                  no_lora_swap=False, prefetch_stream=None):
#         self.layer_num_ = layer_num
#         self.tp_rank_ = tp_rank
#         self.world_size_ = world_size
#         self.data_type_ = data_type
#         self.lora_config = lora_config
#         self.network_config = network_config

#         # lora params
#         self.q_lora_A = None
#         self.q_lora_B = None
#         self.k_lora_A = None
#         self.k_lora_B = None
#         self.v_lora_A = None
#         self.v_lora_B = None

#         self.prefetch_stream = prefetch_stream

#         # debug
#         self.no_lora_swap = no_lora_swap


#     def load_to_torch(self, path):
#         numpy_type = {"fp32": np.float32, "fp16": np.float16}[self.data_type_]
#         torch_type = {"fp32": torch.float32, "fp16": torch.float16}[self.data_type_]
#         return torch.from_numpy(np.fromfile(path, dtype=numpy_type)).to(torch_type)


#     def load_dummy_weights(self, swap):
#         n_embed = self.network_config["hidden_size"]
#         split_n_embed = n_embed // self.world_size_
#         rank = self.lora_config["r"]
#         if not swap or self.no_lora_swap:
#             self.q_lora_A = (torch.rand((rank, split_n_embed), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.q_lora_B = (torch.rand((split_n_embed, rank), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.k_lora_A = (torch.rand((rank, split_n_embed), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.k_lora_B = (torch.rand((split_n_embed, rank), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.v_lora_A = (torch.rand((rank, split_n_embed), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.v_lora_B = (torch.rand((split_n_embed, rank), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.o_lora_A = (torch.rand((rank, split_n_embed), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#             self.o_lora_B = (torch.rand((split_n_embed, rank), 
#                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
#         else:
#             self.q_lora_A_home = ((torch.rand((rank, split_n_embed), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.q_lora_A = None
#             self.q_lora_B_home = ((torch.rand((split_n_embed, rank), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.q_lora_B = None
#             self.k_lora_A_home = ((torch.rand((rank, split_n_embed), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.k_lora_A = None
#             self.k_lora_B_home = ((torch.rand((split_n_embed, rank), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.k_lora_B = None
#             self.v_lora_A_home = ((torch.rand((rank, split_n_embed), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.v_lora_A = None
#             self.v_lora_B_home = ((torch.rand((split_n_embed, rank), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.v_lora_B = None
#             self.o_lora_A_home = ((torch.rand((rank, split_n_embed), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.o_lora_A = None
#             self.o_lora_B_home = ((torch.rand((split_n_embed, rank), 
#                                             dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
#             self.o_lora_B = None

#             num_head = self.network_config["num_attention_heads"]
#             self.w_combined_home = torch.concat(
#                 [self.q_lora_A_home.T.reshape(rank, num_head, -1),
#                  self.k_lora_A_home.T.reshape(rank, num_head, -1),
#                  self.v_lora_A_home.T.reshape(rank, num_head, -1),
#                  self.o_lora_A_home.T.reshape(rank, num_head, -1),
#                  self.q_lora_B_home.T.reshape(rank, num_head, -1),
#                  self.k_lora_B_home.T.reshape(rank, num_head, -1),
#                  self.v_lora_B_home.T.reshape(rank, num_head, -1),
#                  self.o_lora_B_home.T.reshape(rank, num_head, -1)]).pin_memory()
#             self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, -1)
#             self.w_combined = None
#         return
 

#     def load_hf_weights(self, weights, swap=False, dummy=False):
#         if dummy:
#             self.load_dummy_weights(swap)
#             return

#         if swap and not self.no_lora_swap:
#             self.load_hf_weights_cpu(weights)
#             return

#         n_embed = self.network_config["hidden_size"]
#         split_n_embed = n_embed // self.world_size_

#         prefix = list(weights.keys())[0]
#         prefix = prefix[:prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"
#         tp_idx = (split_n_embed * self.tp_rank_, split_n_embed * (self.tp_rank_ + 1))

#         # q_proj A, B
#         if f"{prefix}.q_proj.lora_A.weight" in weights:
#             self.q_lora_A = weights[f"{prefix}.q_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.q_lora_A = self.q_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
#             self.q_lora_A = self.q_lora_A.cuda()

#         if f"{prefix}.q_proj.lora_B.weight" in weights:
#             self.q_lora_B = weights[f"{prefix}.q_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.q_lora_B = self.q_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
#             self.q_lora_B = self.q_lora_B.cuda()

#         # k_proj A, B
#         if f"{prefix}.k_proj.lora_A.weight" in weights:
#             self.k_lora_A = weights[f"{prefix}.k_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.k_lora_A = self.k_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
#             self.k_lora_A = self.k_lora_A.cuda()

#         if f"{prefix}.k_proj.lora_B.weight" in weights:
#             self.k_lora_B = weights[f"{prefix}.k_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.k_lora_B = self.k_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
#             self.k_lora_B = self.k_lora_B.cuda()

#         # v_proj A, B
#         if f"{prefix}.v_proj.lora_A.weight" in weights:
#             self.v_lora_A = weights[f"{prefix}.v_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.v_lora_A = self.v_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
#             self.v_lora_A = self.v_lora_A.cuda()

#         if f"{prefix}.v_proj.lora_B.weight" in weights:
#             self.v_lora_B = weights[f"{prefix}.v_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.v_lora_B = self.v_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
#             self.v_lora_B = self.v_lora_B.cuda()

#         # o_proj A, B
#         if f"{prefix}.o_proj.lora_A.weight" in weights:
#             self.o_lora_A = weights[f"{prefix}.o_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.o_lora_A = self.o_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
#             self.o_lora_A = self.o_lora_A.cuda()

#         if f"{prefix}.o_proj.lora_B.weight" in weights:
#             self.o_lora_B = weights[f"{prefix}.o_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.o_lora_B = self.o_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
#             self.o_lora_B = self.o_lora_B.cuda()

#         return


#     def load_hf_weights_cpu(self, weights):
#         n_embed = self.network_config["hidden_size"]
#         split_n_embed = n_embed // self.world_size_

#         prefix = list(weights.keys())[0]
#         prefix = prefix[:prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"
#         tp_idx = (split_n_embed * self.tp_rank_, split_n_embed * (self.tp_rank_ + 1))

#         # q_proj A, B
#         if f"{prefix}.q_proj.lora_A.weight" in weights:
#             self.q_lora_A_home = weights[f"{prefix}.q_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.q_lora_A_home = self.q_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.q_lora_A = None

#         if f"{prefix}.q_proj.lora_B.weight" in weights:
#             self.q_lora_B_home = weights[f"{prefix}.q_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.q_lora_B_home = self.q_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.q_lora_B = None

#         # k_proj A, B
#         if f"{prefix}.k_proj.lora_A.weight" in weights:
#             self.k_lora_A_home = weights[f"{prefix}.k_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.k_lora_A_home = self.k_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.k_lora_A = None

#         if f"{prefix}.k_proj.lora_B.weight" in weights:
#             self.k_lora_B_home = weights[f"{prefix}.k_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.k_lora_B_home = self.k_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.k_lora_B = None

#         # v_proj A, B
#         if f"{prefix}.v_proj.lora_A.weight" in weights:
#             self.v_lora_A_home = weights[f"{prefix}.v_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.v_lora_A_home = self.v_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.v_lora_A = None

#         if f"{prefix}.v_proj.lora_B.weight" in weights:
#             self.v_lora_B_home = weights[f"{prefix}.v_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.v_lora_B_home = self.v_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.v_lora_B = None

#         # o_proj A, B
#         if f"{prefix}.o_proj.lora_A.weight" in weights:
#             self.o_lora_A_home = weights[f"{prefix}.o_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
#             self.o_lora_A_home = self.o_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.o_lora_A = None

#         if f"{prefix}.o_proj.lora_B.weight" in weights:
#             self.o_lora_B_home = weights[f"{prefix}.o_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
#             self.o_lora_B_home = self.o_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
#             self.o_lora_B = None
        
#         rank = self.lora_config["r"]
#         num_head = self.network_config["num_attention_heads"]
#         self.w_combined_home = torch.concat(
#             [self.q_lora_A_home.T.reshape(rank, num_head, -1),
#                 self.k_lora_A_home.T.reshape(rank, num_head, -1),
#                 self.v_lora_A_home.T.reshape(rank, num_head, -1),
#                 self.o_lora_A_home.T.reshape(rank, num_head, -1),
#                 self.q_lora_B_home.T.reshape(rank, num_head, -1),
#                 self.k_lora_B_home.T.reshape(rank, num_head, -1),
#                 self.v_lora_B_home.T.reshape(rank, num_head, -1),
#                 self.o_lora_B_home.T.reshape(rank, num_head, -1)]).pin_memory()
#         self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, -1)
#         self.w_combined = None

#         return


#     def load_to_gpu(self, prefetch=False, bmm=False):
#         if not bmm:
#             if self.w_combined is None:
#                 if prefetch:
#                     self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
#                 else:
#                     self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
#         else:
#             if self.q_lora_A is None:
#                 self.q_lora_A = self.q_lora_A_home.to("cuda", non_blocking=True)
#                 self.q_lora_B = self.q_lora_B_home.to("cuda", non_blocking=True)
#                 self.k_lora_A = self.k_lora_A_home.to("cuda", non_blocking=True)
#                 self.k_lora_B = self.k_lora_B_home.to("cuda", non_blocking=True)
#                 self.v_lora_A = self.v_lora_A_home.to("cuda", non_blocking=True)
#                 self.v_lora_B = self.v_lora_B_home.to("cuda", non_blocking=True)
#                 self.o_lora_A = self.o_lora_A_home.to("cuda", non_blocking=True)
#                 self.o_lora_B = self.o_lora_B_home.to("cuda", non_blocking=True)
 

#     def offload_from_gpu(self):
#         if self.no_lora_swap:
#             return
#         #assert self.q_lora_A is not None
#         self.w_combined = None
#         self.q_lora_A = None
#         self.q_lora_B = None
#         self.k_lora_A = None
#         self.k_lora_B = None
#         self.v_lora_A = None
#         self.v_lora_B = None
#         self.o_lora_A = None
#         self.o_lora_B = None

"""
    下面的内容是针对Qwen的Adapter进行修改的
"""
        
import gc
import torch
import torch.nn as nn
import numpy as np

class LoraLayerWeight:
    def __init__(self, layer_num, tp_rank, world_size, lora_config, network_config, data_type=torch.float16,
                 no_lora_swap=False, prefetch_stream=None):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.lora_config = lora_config
        self.network_config = network_config

        # lora params
        self.c_attn_A = None
        self.c_attn_B = None
        self.w1_A = None
        self.w1_B = None
        self.w2_A = None
        self.w2_B = None
        self.c_proj_A = None
        self.c_proj_B = None

        self.c_attn_A_home = None
        self.c_attn_B_home = None
        self.w1_A_home = None
        self.w1_B_home = None
        self.w2_A_home = None
        self.w2_B_home = None
        self.c_proj_A_home = None
        self.c_proj_B_home = None

        self.prefetch_stream = prefetch_stream

        # debug
        self.no_lora_swap = no_lora_swap

    def load_to_torch(self, path):
        numpy_type = {"fp32": np.float32, "fp16": np.float16}[self.data_type_]
        torch_type = {"fp32": torch.float32, "fp16": torch.float16}[self.data_type_]
        return torch.from_numpy(np.fromfile(path, dtype=numpy_type)).to(torch_type)

    def load_dummy_weights(self, swap):
        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        rank = self.lora_config["r"]
        if not swap or self.no_lora_swap:
            self.c_attn_A = (torch.rand((rank, split_n_embed), 
                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.c_attn_B = (torch.rand((split_n_embed, rank), 
                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.w1_A = (torch.rand((rank, split_n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.w1_B = (torch.rand((split_n_embed, rank), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.w2_A = (torch.rand((rank, split_n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.w2_B = (torch.rand((split_n_embed, rank), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.c_proj_A = (torch.rand((rank, split_n_embed), 
                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.c_proj_B = (torch.rand((split_n_embed, rank), 
                                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        else:
            self.c_attn_A_home = ((torch.rand((rank, split_n_embed), 
                                              dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.c_attn_A = None
            self.c_attn_B_home = ((torch.rand((split_n_embed, rank), 
                                              dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.c_attn_B = None
            self.w1_A_home = ((torch.rand((rank, split_n_embed), 
                                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.w1_A = None
            self.w1_B_home = ((torch.rand((split_n_embed, rank), 
                                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.w1_B = None
            self.w2_A_home = ((torch.rand((rank, split_n_embed), 
                                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.w2_A = None
            self.w2_B_home = ((torch.rand((split_n_embed, rank), 
                                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.w2_B = None
            self.c_proj_A_home = ((torch.rand((rank, split_n_embed), 
                                              dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.c_proj_A = None
            self.c_proj_B_home = ((torch.rand((split_n_embed, rank), 
                                              dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.c_proj_B = None

            num_head = self.network_config["num_attention_heads"]
            self.w_combined_home = torch.concat(
                [self.c_attn_A_home.T.reshape(rank, num_head, -1) if self.c_attn_A_home is not None else torch.zeros(0, num_head, -1),
                 self.w1_A_home.T.reshape(rank, num_head, -1) if self.w1_A_home is not None else torch.zeros(0, num_head, -1),
                 self.w2_A_home.T.reshape(rank, num_head, -1) if self.w2_A_home is not None else torch.zeros(0, num_head, -1),
                 self.c_proj_A_home.T.reshape(rank, num_head, -1) if self.c_proj_A_home is not None else torch.zeros(0, num_head, -1),
                 self.c_attn_B_home.T.reshape(rank, num_head, -1) if self.c_attn_B_home is not None else torch.zeros(0, num_head, -1),
                 self.w1_B_home.T.reshape(rank, num_head, -1) if self.w1_B_home is not None else torch.zeros(0, num_head, -1),
                 self.w2_B_home.T.reshape(rank, num_head, -1) if self.w2_B_home is not None else torch.zeros(0, num_head, -1),
                 self.c_proj_B_home.T.reshape(rank, num_head, -1) if self.c_proj_B_home is not None else torch.zeros(0, num_head, -1)]).pin_memory()
            self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, -1)
            self.w_combined = None
        return

    def load_hf_weights(self, weights, swap=False, dummy=False):
        if dummy:
            self.load_dummy_weights(swap)
            return

        if swap and not self.no_lora_swap:
            self.load_hf_weights_cpu(weights)
            return

        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_

        prefix = list(weights.keys())[0]
        prefix = prefix[:prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"
        tp_idx = (split_n_embed * self.tp_rank_, split_n_embed * (self.tp_rank_ + 1))

        # c_attn A, B
        if f"{prefix}.c_attn.lora_A.weight" in weights:
            self.c_attn_A = weights[f"{prefix}.c_attn.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.c_attn_A = self.c_attn_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.c_attn_A = self.c_attn_A.cuda()

        if f"{prefix}.c_attn.lora_B.weight" in weights:
            self.c_attn_B = weights[f"{prefix}.c_attn.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.c_attn_B = self.c_attn_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.c_attn_B = self.c_attn_B.cuda()

        # w1 A, B
        if f"{prefix}.w1.lora_A.weight" in weights:
            self.w1_A = weights[f"{prefix}.w1.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.w1_A = self.w1_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.w1_A = self.w1_A.cuda()

        if f"{prefix}.w1.lora_B.weight" in weights:
            self.w1_B = weights[f"{prefix}.w1.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.w1_B = self.w1_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.w1_B = self.w1_B.cuda()

        # w2 A, B
        if f"{prefix}.w2.lora_A.weight" in weights:
            self.w2_A = weights[f"{prefix}.w2.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.w2_A = self.w2_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.w2_A = self.w2_A.cuda()

        if f"{prefix}.w2.lora_B.weight" in weights:
            self.w2_B = weights[f"{prefix}.w2.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.w2_B = self.w2_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.w2_B = self.w2_B.cuda()

        # c_proj A, B
        if f"{prefix}.c_proj.lora_A.weight" in weights:
            self.c_proj_A = weights[f"{prefix}.c_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.c_proj_A = self.c_proj_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.c_proj_A = self.c_proj_A.cuda()

        if f"{prefix}.c_proj.lora_B.weight" in weights:
            self.c_proj_B = weights[f"{prefix}.c_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.c_proj_B = self.c_proj_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.c_proj_B = self.c_proj_B.cuda()

        return

    def load_hf_weights_cpu(self, weights):
        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_

        prefix = list(weights.keys())[0]
        prefix = prefix[:prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"
        tp_idx = (split_n_embed * self.tp_rank_, split_n_embed * (self.tp_rank_ + 1))

        rank = self.lora_config["r"]
        num_head = self.network_config["num_attention_heads"]

        target_dtype = torch.half if self.data_type_ == torch.float16 else torch.float

        # Helper function to reshape and handle None values
        def safe_reshape(tensor, rank, num_head, embed_dim):
            if tensor is not None:
                return tensor.T.reshape(rank, num_head, embed_dim // num_head)
            else:
                return torch.zeros(rank, num_head, embed_dim // num_head, dtype=target_dtype, device="cpu").pin_memory()

        # c_attn A, B
        if f"{prefix}.c_attn.lora_A.weight" in weights:
            self.c_attn_A_home = weights[f"{prefix}.c_attn.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.c_attn_A_home = self.c_attn_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.c_attn_A = None
            print(f"Loaded c_attn_A_home with shape: {self.c_attn_A_home.shape}")

        if f"{prefix}.c_attn.lora_B.weight" in weights:
            self.c_attn_B_home = weights[f"{prefix}.c_attn.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.c_attn_B_home = self.c_attn_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.c_attn_B = None
            print(f"Loaded c_attn_B_home with shape: {self.c_attn_B_home.shape}")

        # w1 A, B
        if f"{prefix}.w1.lora_A.weight" in weights:
            self.w1_A_home = weights[f"{prefix}.w1.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.w1_A_home = self.w1_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.w1_A = None
            print(f"Loaded w1_A_home with shape: {self.w1_A_home.shape}")

        if f"{prefix}.w1.lora_B.weight" in weights:
            self.w1_B_home = weights[f"{prefix}.w1.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.w1_B_home = self.w1_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.w1_B = None
            print(f"Loaded w1_B_home with shape: {self.w1_B_home.shape}")

        # w2 A, B
        if f"{prefix}.w2.lora_A.weight" in weights:
            self.w2_A_home = weights[f"{prefix}.w2.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.w2_A_home = self.w2_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.w2_A = None
            print(f"Loaded w2_A_home with shape: {self.w2_A_home.shape}")

        if f"{prefix}.w2.lora_B.weight" in weights:
            self.w2_B_home = weights[f"{prefix}.w2.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.w2_B_home = self.w2_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.w2_B = None
            print(f"Loaded w2_B_home with shape: {self.w2_B_home.shape}")

        # c_proj A, B
        if f"{prefix}.c_proj.lora_A.weight" in weights:
            self.c_proj_A_home = weights[f"{prefix}.c_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.c_proj_A_home = self.c_proj_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.c_proj_A = None
            print(f"Loaded c_proj_A_home with shape: {self.c_proj_A_home.shape}")

        if f"{prefix}.c_proj.lora_B.weight" in weights:
            self.c_proj_B_home = weights[f"{prefix}.c_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.c_proj_B_home = self.c_proj_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.c_proj_B = None
            print(f"Loaded c_proj_B_home with shape: {self.c_proj_B_home.shape}")

        self.w_combined_home = torch.concat([
            safe_reshape(self.c_attn_A_home, rank, num_head, n_embed),
            safe_reshape(self.w1_A_home, rank, num_head, n_embed),
            safe_reshape(self.w2_A_home, rank, num_head, n_embed),
            safe_reshape(self.c_proj_A_home, rank, num_head, n_embed),
            safe_reshape(self.c_attn_B_home, rank, num_head, n_embed),
            safe_reshape(self.w1_B_home, rank, num_head, n_embed),
            safe_reshape(self.w2_B_home, rank, num_head, n_embed),
            safe_reshape(self.c_proj_B_home, rank, num_head, n_embed)
        ]).pin_memory()
        self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, n_embed // num_head).to(target_dtype)
        self.w_combined = None



        return


    def load_to_gpu(self, prefetch=False, bmm=False):
        if not bmm:
            if self.w_combined is None:
                if prefetch:
                    self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
                else:
                    self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
        else:
            if self.c_attn_A is None:
                self.c_attn_A = self.c_attn_A_home.to("cuda", non_blocking=True)
                self.c_attn_B = self.c_attn_B_home.to("cuda", non_blocking=True)
                self.w1_A = self.w1_A_home.to("cuda", non_blocking=True)
                self.w1_B = self.w1_B_home.to("cuda", non_blocking=True)
                self.w2_A = self.w2_A_home.to("cuda", non_blocking=True)
                self.w2_B = self.w2_B_home.to("cuda", non_blocking=True)
                self.c_proj_A = self.c_proj_A_home.to("cuda", non_blocking=True)
                self.c_proj_B = self.c_proj_B_home.to("cuda", non_blocking=True)

    def offload_from_gpu(self):
        if self.no_lora_swap:
            return
        self.w_combined = None
        self.c_attn_A = None
        self.c_attn_B = None
        self.w1_A = None
        self.w1_B = None
        self.w2_A = None
        self.w2_B = None
        self.c_proj_A = None
        self.c_proj_B = None
