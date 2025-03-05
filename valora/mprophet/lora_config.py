import re
from valora.utils.model_load import hf_load_config


class LoRAConfig:
    name: str
    
    def __init__(self, name: str, config=None, weight_dir=None):
        
        self.name = name
        print(f"lora name:{self.name}")
        
        if weight_dir is not None:
            weight_dir = re.sub(r'-(\d+)$', '', weight_dir)
            config, _ = hf_load_config(weight_dir)

        if config is not None:
            self.config = config
            self._init_from_dict(config)
            return

        if "alpaca-lora-7b" in name:
            self.base_model = None
            self.rank = 16
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "bactrian-x-llama-7b-lora" in name:
            self.base_model = None
            self.rank = 64
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "dummy-lora-7b-rank" in name:
            self.base_model = None
            self.rank = int(re.search(r'rank-(\d+)', name).group(1))
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "dummy-lora-13b-rank" in name:
            self.base_model = None
            self.rank = int(re.search(r'rank-(\d+)', name).group(1))
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "Qwen-VL" in name:
            self.base_model = "/data01/tuwenming/models/Qwen-VL-Chat"
            self.rank = 64
            self.target_modules = ["w2", "c_attn", "attn.c_proj", "w1"]
            
        elif "llava-v1.5-7b" in name:
            self.base_model = "/data01/tuwenming/S-LoRA/models/LLava/llava-v1.5-7b"
            self.rank = 128
            self.target_modules =  [
                    "k_proj",
                    "gate_proj",
                    "v_proj",
                    "q_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                ]
        elif "llava-v1.5-13b" in name:
            self.base_model = "/data01/tuwenming/S-LoRA/models/LLava/llava-v1.5-13b"
            self.rank = 128
            self.target_modules =  [
                    "k_proj",
                    "gate_proj",
                    "v_proj",
                    "q_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                ]
        elif "InternVL2-8B" in name:
            self.base_model = "/data01/tuwenming/S-LoRA/models/InternVL/InternVL2-8B"
            self.rank = 16
            self.target_modules =  ["attention.wqkv", "attention.wo"]
        elif "InternVL2-26B" in name:
            self.base_model = "/data01/tuwenming/S-LoRA/models/InternVL/InternVL2-26B"
            self.rank = 16
            self.target_modules =  ["attention.wqkv", "attention.wo"]
        else:
            raise NotImplementedError

    
    def _init_from_dict(self, config):
        self.base_model = config["base_model_name_or_path"]
        self.rank = config["r"]
        self.target_modules = config["target_modules"]


def get_lora_config_json(name):
    print(f"lora name:{name}")
    
    if "alpaca-lora-7b" in name:
        config = {"base_model_name_or_path": "decapoda-research/llama-7b-hf",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": 16,
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "bactrian-x-llama-7b-lora" in name:
        config = {
                  "base_model_name_or_path": "decapoda-research/llama-7b-hf",
                  "bias": "none",
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "init_lora_weights": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": 64,
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "dummy-lora-7b-rank-" in name:
        config = {"base_model_name_or_path": "/data01/tuwenming/S-LoRA/models/huggyllama/llama-7b",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": int(re.search(r'rank-(\d+)', name).group(1)),
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "dummy-lora-13b-rank-" in name:
        config = {"base_model_name_or_path": "meta-llama/Llama-2-13b-hf",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.1,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": int(re.search(r'rank-(\d+)', name).group(1)),
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "Qwen-VL" in name:
        config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": "/data01/tuwenming/models/Qwen-VL-Chat",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 64,
            "rank_pattern": {},
            "revision": None,
            "target_modules": ["w2", "c_attn", "attn.c_proj", "w1"],
            "task_type": "CAUSAL_LM",
        }
    elif "llava-v1.5-7b" in name:
        config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": "/data01/tuwenming/S-LoRA/models/LLava/llava-v1.5-7b",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 128,
            "rank_pattern": {},
            "revision": None,
            "target_modules": [
                "k_proj",
                "gate_proj",
                "v_proj",
                "q_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ],
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
    elif "llava-v1.5-13b" in name:
        config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": "/data01/tuwenming/S-LoRA/models/LLava/llava-v1.5-13b",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 128,
            "rank_pattern": {},
            "revision": None,
            "target_modules": [
                "v_proj",
                "q_proj",
                "down_proj",
                "gate_proj",
                "up_proj",
                "k_proj",
                "o_proj",
            ],
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
    elif "InternVL2-8B" in name:
        config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": "/data01/tuwenming/S-LoRA/models/InternVL/InternVL2-8B",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "rank_pattern": {},
            "revision": None,
            "target_modules": ["attention.wqkv", "attention.wo"],
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
    elif "InternVL2-26B" in name:
        config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": "/data01/tuwenming/S-LoRA/models/InternVL/InternVL2-26B",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "rank_pattern": {},
            "revision": None,
            "target_modules": ["attention.wo", "attention.wqkv"],
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
    else:
        raise Exception(f"unrecognized: {name}")
    return config
