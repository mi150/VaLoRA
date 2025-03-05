import torch
import math
import numpy as np
from valora.common.basemodel import TransformerLayerWeight

class QwenTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.att_norm_weight_ = None
        self.q_weight_ = None
        self.k_weight_ = None
        self.v_weight_ = None
        self.o_weight_ = None
        self.ffn_norm_weight_ = None
        self.up_proj = None
        self.gate_proj = None
        self.down_proj = None

    def load_hf_weights(self, weights, dummy=False):
        if dummy:
            self._load_qkvo_dummy_weights()
            self._load_ffn_dummy_weights()
        else:
            self._load_qkvo_weights(weights)
            self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.up_proj,
                   self.gate_proj,
                   self.down_proj]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors

    def _load_qkvo_dummy_weights(self):
        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # input layernorm params
        self.att_norm_weight_ = (torch.rand((n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
        # attention params
        self.q_weight_ = (torch.rand((split_n_embed, n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.k_weight_ = (torch.rand((split_n_embed, n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.v_weight_ = (torch.rand((split_n_embed, n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.qkv_weight_ = torch.cat([self.q_weight_, self.k_weight_, self.v_weight_], dim=1)
        # attention output dense params
        self.o_weight_ = (torch.rand((n_embed, split_n_embed),
                                     dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3

    def _load_ffn_dummy_weights(self):
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        self.ffn_norm_weight_ = (torch.rand((n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

        self.up_proj = (torch.rand((split_inter_size, n_embed),
                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.gate_proj = (torch.rand((split_inter_size, n_embed),
                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.down_proj = (torch.rand((n_embed, split_inter_size),
                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"transformer.h.{self.layer_num_}.ln_1.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_1.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for Qwen
        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weight = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
            self.q_weight_ = self._cuda(qkv_weight[:n_embed, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1))
            self.k_weight_ = self._cuda(qkv_weight[n_embed:2 * n_embed, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1))
            self.v_weight_ = self._cuda(qkv_weight[2 * n_embed:, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1))
            self.qkv_weight_ = torch.cat([self.q_weight_, self.k_weight_, self.v_weight_], dim=1)
        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            self.o_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))

    def _load_ffn_weights(self, weights):
        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_2.weight"])
    
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"transformer.h.{self.layer_num_}.mlp.w1.weight" in weights:
            self.up_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w1.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self._cuda(self.up_proj.transpose(0, 1))

        if f"transformer.h.{self.layer_num_}.mlp.w2.weight" in weights:
            self.gate_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w2.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self._cuda(self.gate_proj.transpose(0, 1))

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.down_proj = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
