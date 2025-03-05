import os
import json
import torch
from valora.models.qwen.layer_infer.pre_layer_infer import QwenPreLayerInfer
from valora.models.qwen.layer_infer.post_layer_infer import QwenPostLayerInfer
from valora.models.qwen.layer_infer.transformer_layer_infer import QwenTransformerLayerInfer
from valora.models.qwen.layer_weights.pre_and_post_layer_weight import QwenPreAndPostLayerWeight
from valora.models.qwen.layer_weights.transformer_layer_weight import QwenTransformerLayerWeight


from valora.models.qwen.infer_struct import QwenInferStateInfo
from valora.common.mem_allocator import MemoryAllocator
from valora.common.int8kv_mem_manager import INT8KVMemoryManager
from valora.common.basemodel import TpPartBaseModel

class QwenTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = QwenPreAndPostLayerWeight
    transformer_weight_class = QwenTransformerLayerWeight

    # infer class
    pre_layer_infer_class = QwenPreLayerInfer
    post_layer_infer_class = QwenPostLayerInfer
    transformer_layer_infer_class = QwenTransformerLayerInfer

    # infer state class
    infer_state_class = QwenInferStateInfo
    
    # Mem manager class
    memory_manager_class = MemoryAllocator

    def __init__(self, tp_rank, world_size, weight_dir, 
                 max_total_token_num, mem_adapter_size, load_way="HF", mode=[], dummy=False):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size, load_way, mode, dummy=dummy)
        return
    
    def _init_config(self):
        super()._init_config()
        return 
    
    def _verify_params(self):
        assert self.load_way == "HF", "qwen only support HF format to load Now!"

    def _init_mem_manager(self):
        mem_dict = {
            "int8kv" : INT8KVMemoryManager
        }
        for _mode in self.mode:
            if _mode in mem_dict:
                print("Model using mode", _mode)
                self.memory_manager_class = mem_dict[_mode]
                
        
        
        
        tot_size = self.max_total_token_num + self.mem_adapter_size
        cache_size = self.max_total_token_num
        dtype = torch.float16
        head_num = self.config["num_attention_heads"] // self.world_size_
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        layer_num = self.config["num_hidden_layers"]

        # 打印参数
        print(f"tot_size: {tot_size}")
        print(f"cache_size: {cache_size}")
        print(f"dtype: {dtype}")
        print(f"head_num: {head_num}")
        print(f"head_dim: {head_dim}")
        print(f"layer_num: {layer_num}")
        
        
        
        
        self.mem_manager = self.memory_manager_class(tot_size=self.max_total_token_num + self.mem_adapter_size, 
                                                     cache_size=self.max_total_token_num,
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_attention_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"])

    def _init_custom(self):
        if self.config.get("use_dynamic_ntk", False):
            # self._init_to_get_dynamic_ntk_rotary()
            self._init_to_get_rotary()
            
        else:
            self._init_to_get_rotary()
        return

    def _init_to_get_rotary(self, default_base=10000.0):
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

    def _init_to_get_dynamic_ntk_rotary(self):
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
            print(seq_loc_index)
            new_base = base * ((scaling_factor * (seq_loc_index + 1) / max_position_embeddings) - (scaling_factor - 1)) ** (self.head_dim_ / (self.head_dim_ - 2))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
            t = torch.tensor([seq_loc_index,], device="cpu", dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached[seq_loc_index:seq_loc_index + 1, :] = torch.cos(freqs).to(torch.float16).cuda()
            self._sin_cached[seq_loc_index:seq_loc_index + 1, :] = torch.sin(freqs).to(torch.float16).cuda()
        return
