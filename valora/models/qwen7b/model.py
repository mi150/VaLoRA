import os
import json
import torch
import math
import numpy as np

from valora.models.llama.model import LlamaTpPartModel
from valora.models.qwen7b.layer_weights.transformer_layer_weight import QwenTransformerLayerWeight
from valora.models.qwen7b.layer_weights.pre_and_post_layer_weight import QwenPreAndPostLayerWeight
from valora.models.qwen7b.layer_infer.transformer_layer_infer import QwenTransformerLayerInfer
from valora.models.qwen7b.infer_struct import QwenInferStateInfo
from valora.common.build_utils import repair_config
from valora.common.mem_allocator import MemoryAllocator
from valora.common.int8kv_mem_manager import INT8KVMemoryManager

class Qwen7bTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = QwenPreAndPostLayerWeight
    transformer_weight_class = QwenTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = QwenTransformerLayerInfer

    # infer state class
    infer_state_class = QwenInferStateInfo
    
    def __init__(
        self,
        tp_rank,
        world_size,
        weight_dir,
        max_total_token_num,
        mem_adapter_size,
        load_way="HF",
        mode=[],
        dummy=False,
    ):
        super().__init__(
            tp_rank,
            world_size,
            weight_dir,
            max_total_token_num,
            mem_adapter_size,
            load_way,
            mode,
            dummy=dummy,
        )

    def _init_config(self):
        super()._init_config()
        self._reset_num_key_value_heads()
        # 修复配置中的键名
        repair_config(self.config, same_names=["ffn_hidden_size", "intermediate_size"])
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_epsilon"])
        return

    def _verify_params(self):
        assert self.load_way == "HF", "Qwen only supports HF format to load now!"
        # assert self.config["num_key_value_heads"] % self.world_size == 0
        # assert self.config["num_attention_heads"] % self.world_size == 0
        return

    # def _init_mem_manager(self):
    #     self.mem_manager = self.memory_manager_class(
    #         tot_size=self.max_total_token_num + self.mem_adapter_size,
    #         cache_size=self.max_total_token_num,
    #         dtype=torch.float16,
    #         head_num=self.config["num_key_value_heads"] // self.world_size_,
    #         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
    #         layer_num=self.config["num_hidden_layers"]
    #     )
    #     return
    def _init_mem_manager(self):
        self.mem_manager = self.memory_manager_class(
            tot_size=self.max_total_token_num + self.mem_adapter_size,
            cache_size=self.max_total_token_num,
            dtype=torch.float16,
            head_num=self.config["num_key_value_heads"] // self.world_size_,
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"],
        )
        return

    def _reset_num_key_value_heads(self):
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return

    def _init_some_value(self):
        self.head_dim_ = self.config["hidden_size"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["num_hidden_layers"]
        self.vocab_size = self.config["vocab_size"]
        return

    def _init_custom(self):
        """
        Initialize Qwen-specific features like dynamic NTK and logarithmic attention.
        """
        if self.config.get("use_dynamic_ntk", False) and self.config.get("use_logn_attn", False):
            # self._init_to_get_dynamic_ntk_rotary()
            self._init_to_get_rotary()
            self._init_qwen_logn_attn()
        else:
            super()._init_custom()
            self.logn_tensor = None
        return

    def _init_nkt_alpha(self, total_seq_len_supported):
        ntk_alphas = []
        for seq_len in range(1, total_seq_len_supported + 1):
            ntk_alpha = max(2 ** math.ceil(math.log(seq_len / self.config.get("seq_length", 2048), 2) + 1), 1)
            ntk_alphas.append(ntk_alpha)
        ntk_alphas = np.array(ntk_alphas, dtype=np.int32)
        self.max_ntk_alpha = math.ceil(math.log(ntk_alphas.max(), 2))
        return np.unique(ntk_alphas)

    def _init_qwen_dynamic_ntk(self): # size not match
        total_seq_len_supported = self.config.get("max_position_embeddings", 8 * 1024)
        seq_len = self.config.get("seq_length", 2048)

        ntk_alphas = self._init_nkt_alpha(total_seq_len_supported)
        self._cos_cached = []
        self._sin_cached = []

        for ntk_alpha in ntk_alphas:
            base = self.config.get("rotary_emb_base", 10000)
            base = base * ntk_alpha ** (self.head_dim_ / (self.head_dim_ - 2))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32)
                    / self.head_dim_
                )
            )

            t = torch.arange(total_seq_len_supported + 128 * 1024, device="cpu", dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached.append(torch.cos(freqs).to(torch.float16).cuda()) # 有一个data_type
            self._sin_cached.append(torch.sin(freqs).to(torch.float16).cuda()) # 有一个data_type

        self._cos_cached = torch.stack(self._cos_cached, dim=0).contiguous()
        self._sin_cached = torch.stack(self._sin_cached, dim=0).contiguous()
        return
    
    
    def _init_to_get_dynamic_ntk_rotary(self): # dynamic rope but slow
        # total_seq_len_supported = self.config.get("max_position_embeddings", 8 * 1024)
        # ntk_alphas = self._init_nkt_alpha(total_seq_len_supported)

        
        
        
        max_position_embeddings = self.config.get("max_position_embeddings", 8192)
        base = self.config.get("rotary_emb_base", 10000.0)
        scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)
        max_seq_len = 32 * max_position_embeddings
        self._cos_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=torch.float16, device="cuda")
        self._sin_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=torch.float16, device="cuda")
        
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached[0:max_position_embeddings, :] = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached[0:max_position_embeddings, :] = torch.sin(freqs).to(torch.float16).cuda()

        for seq_loc_index in range(max_position_embeddings, max_seq_len, 1):
            new_base = base * ((scaling_factor * (seq_loc_index + 1) / max_position_embeddings) - (scaling_factor - 1)) ** (self.head_dim_ / (self.head_dim_ - 2))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
            t = torch.tensor([seq_loc_index,], device="cpu", dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached[seq_loc_index:seq_loc_index + 1, :] = torch.cos(freqs).to(torch.float16).cuda()
            self._sin_cached[seq_loc_index:seq_loc_index + 1, :] = torch.sin(freqs).to(torch.float16).cuda()
        return
    
    def _init_to_get_rotary(self, default_base=10000.0): # fast rope
        # total_seq_len_supported = self.config.get("max_position_embeddings", 8 * 1024)
        # ntk_alphas = self._init_nkt_alpha(total_seq_len_supported)


        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rotary_emb_base", float(default_base))

        if "max_position_embeddings" in self.config:
            max_seq_len = self.config["max_position_embeddings"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 8192
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        try:
            ntk_alpha = float(os.environ.get("SLORA_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                print(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_ - 2)))
        except:
            pass

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
    
    
    def _init_qwen_logn_attn(self):
        total_seq_len_supported = self.config.get("max_position_embeddings", 8 * 1024)
        seq_len = self.config.get("seq_length", 2048)
        logn_list = [
            math.log(i, seq_len) if i > seq_len else 1
            for i in range(1, total_seq_len_supported + 128 * 1024 + 1)
        ]
        self.logn_tensor = torch.tensor(logn_list).cuda()
        return