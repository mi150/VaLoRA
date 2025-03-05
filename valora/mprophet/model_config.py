from valora.utils.model_load import hf_load_config

# TODO: add qwen7b & qwenvl configuration
class ModelConfig:
    name: str
    
    def __init__(self, name: str, config=None, model_dir=None):
        self.name = name

        if model_dir is not None:
            config, _ = hf_load_config(model_dir, mode="model")

        if config is not None:
            self._init_from_dict(config)
            return

        if "opt" in name.lower():
            if "opt-125m" in name.lower():
                self.max_seq_len = 2048
                self.num_hidden_layers = 12
                self.n_head=12
                self.hidden_size=768
                self.ffn_embed_dim=768 * 4
            elif "opt-6.7b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=4096 * 4
            elif "opt-13b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=40
                self.n_head=40
                self.hidden_size=5120
                self.ffn_embed_dim=5120 * 4
            elif "opt-30b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=48
                self.n_head=56
                self.hidden_size=7168
                self.ffn_embed_dim=7168 * 4
            elif "opt-175b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=96
                self.n_head=96
                self.hidden_size=12288
                self.ffn_embed_dim=12288 * 4

        elif "llama" in name.lower():
            if "llama-7b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            elif "llama-13b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=40
                self.n_head=40
                self.hidden_size=5120
                self.ffn_embed_dim=13824
            elif "llama-30b-m" in name.lower():
                # Parameters are modified to fit the requirements of custom kernels.
                # Not the official parameters.
                self.max_seq_len=2048
                self.num_hidden_layers=48
                self.n_head=56
                self.hidden_size=7168
                self.ffn_embed_dim=19456
            elif "llama-70b-m" in name.lower():
                # Parameters are modified to fit the requirements of custom kernels.
                # Not the official parameters.
                self.max_seq_len=2048
                self.num_hidden_layers=80
                self.n_head=64
                self.hidden_size=8192
                self.ffn_embed_dim=24576
            elif "llama-14-layer" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=14
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            elif "llama-16-layer" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=16
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            elif "llama-2-7b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            else:
                raise NotImplementedError
        
        elif "qwen" in name.lower():
            if "qwen7b" in name.lower():
                self.max_seq_len=8192
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=22016
            elif "qwenvl" in name.lower():
                self.max_seq_len=8192
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=22016
            else:    
                self.max_seq_len=8192
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=22016
        elif "llava-v1.5-7b" in name.lower():
            self.max_seq_len=4096
            self.num_hidden_layers=32
            self.n_head=32
            self.hidden_size=4096
            self.ffn_embed_dim=11008
        elif "llava-v1.5-13b" in name.lower():
            self.max_seq_len=4096
            self.num_hidden_layers=40
            self.n_head=40
            self.hidden_size=5120
            self.ffn_embed_dim=13824
        elif "InternVL2-8B" in name.lower():
            self.max_seq_len = 32768
            self.num_hidden_layers = 32
            self.n_head = 32
            self.hidden_size = 4096
            self.ffn_embed_dim = 14336
        elif "InternVL2-26B" in name.lower():
            self.max_seq_len = 32768
            self.num_hidden_layers = 48
            self.n_head = 48
            self.hidden_size = 6144
            self.ffn_embed_dim = 16384

        else:
            raise NotImplementedError


    def _init_from_dict(self, config):
        if "llama" in self.name.lower():
            if "max_sequence_length" in config:
                self.max_seq_len = config["max_sequence_length"]
            else:
                self.max_seq_len = config["max_position_embeddings"]
            self.num_hidden_layers = config["num_hidden_layers"]
            self.n_head = config["num_attention_heads"]
            self.hidden_size = config["hidden_size"]
            self.ffn_embed_dim = config["intermediate_size"]
            self.vocab_size = config["vocab_size"]
        
        elif "qwen" in self.name.lower():
            if "max_sequence_length" in config:
                self.max_seq_len = config["max_sequence_length"]
            else:
                self.max_seq_len = config["max_position_embeddings"]
            self.num_hidden_layers = config["num_hidden_layers"]
            self.n_head = config["num_attention_heads"]
            self.hidden_size = config["hidden_size"]
            self.ffn_embed_dim = config["intermediate_size"]
            self.vocab_size = config["vocab_size"]
        elif "llava" in self.name.lower():
            if "max_length" in config:
                self.max_seq_len = config["max_length"]
            else:
                self.max_seq_len = config["max_position_embeddings"]
            self.num_hidden_layers = config["num_hidden_layers"]
            self.n_head = config["num_attention_heads"]
            self.hidden_size = config["hidden_size"]
            self.ffn_embed_dim = config["intermediate_size"]
            self.vocab_size = config["vocab_size"]
        elif "internvl" in self.name.lower():
            self.max_seq_len = config["llm_config"]["max_position_embeddings"]
            self.num_hidden_layers = config["llm_config"]["num_hidden_layers"]
            self.n_head = config["llm_config"]["num_attention_heads"]
            self.hidden_size = config["llm_config"]["hidden_size"]
            self.ffn_embed_dim = config["llm_config"]["intermediate_size"]
            self.vocab_size = config["llm_config"]["vocab_size"]

        else:
            raise NotImplementedError


def get_config_json(name):
    print(f"Model Name:{name}")
    if "llama-7b" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-13b" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 13824, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 40, "num_hidden_layers": 40, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-30b-m" in name.lower():
        # Parameters are modified to fit the requirements of custom kernels.
        # Not the official parameters.

        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 7168, "intermediate_size": 19456, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 56, "num_hidden_layers": 48, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-70b-m" in name.lower():
        # Parameters are modified to fit the requirements of custom kernels.
        # Not the official parameters.
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 8192, "intermediate_size": 24576, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 64, "num_hidden_layers": 80, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-2-7b" in name.lower():
        config = {
            "_name_or_path": "huggyllama/llama-7b",
            "architectures": [
              "LlamaForCausalLM"
            ],
            "attention_bias": False,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "max_sequence_length": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.34.0",
            "use_cache": True,
            "vocab_size": 32000
        }

    elif "llama-14-layer" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 14, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}

    elif "llama-16-layer" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 16, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}

    elif "qwen" in name.lower():
        if "qwen7b" in name.lower():
            config = {
            "_name_or_path": "/data01/tuwenming/S-LoRA/models/Qwen/Qwen-7B",
            "architectures": [
                "QWenLMHeadModel"
            ],
            "attn_dropout_prob": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_qwen.QWenConfig",
                "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"
            },
            "eos_token_id": 151643,
            "bf16": False,
            "emb_dropout_prob": 0.0,
            "fp16": False,
            "fp32": False,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 22016,
            "kv_channels": 128,
            "layer_norm_epsilon": 1e-06,
            "max_position_embeddings": 8192,
            "model_type": "qwen7b", ######## qwen -> qwen7b
            "no_bias": True,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "onnx_safe": None,
            "rotary_emb_base": 10000,
            "rotary_pct": 1.0,
            "scale_attn_weights": True,
            "seq_length": 2048,
            "tie_word_embeddings": False,
            "tokenizer_type": "QWenTokenizer",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.31.0",
            "use_cache": True,
            "use_dynamic_ntk": True,
            "use_flash_attn": False,
            "use_logn_attn": True,
            "vocab_size": 151936
            }
        
        elif "qwenvl" in name.lower():
            config = {
            "_name_or_path": "/data01/tuwenming/models/Qwen-VL-Chat",
            "architectures": [
                "QWenLMHeadModel"
            ],
            "attn_dropout_prob": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_qwen.QWenConfig",
                "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"
            },
            "eos_token_id": 151643,
            "bf16": False,
            "emb_dropout_prob": 0.0,
            "fp16": False,
            "fp32": False,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 22016,
            "kv_channels": 128,
            "layer_norm_epsilon": 1e-06,
            "max_position_embeddings": 8192,
            "model_type": "qwenvl", ######## qwen -> qwenvl
            "no_bias": True,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "onnx_safe": None,
            "rotary_emb_base": 10000,
            "rotary_pct": 1.0,
            "scale_attn_weights": True,
            "seq_length": 2048,
            "tie_word_embeddings": False,
            "tokenizer_type": "QWenTokenizer",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.31.0",
            "use_cache": True,
            "use_dynamic_ntk": True,
            "use_flash_attn": False,
            "use_logn_attn": True,
            "visual": {
                "heads": 16,
                "image_size": 448,
                "image_start_id": 151857,
                "layers": 48,
                "mlp_ratio": 4.9231,
                "output_dim": 4096,
                "patch_size": 14,
                "width": 1664
            },
            "vocab_size": 151936
            }
        
        else:
            config = {
            "_name_or_path": "/data01/tuwenming/models/Qwen-VL-Chat",
            "architectures": [
                "QWenLMHeadModel"
            ],
            "attn_dropout_prob": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_qwen.QWenConfig",
                "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"
            },
            "eos_token_id": 151643,
            "bf16": False,
            "emb_dropout_prob": 0.0,
            "fp16": False,
            "fp32": False,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 22016,
            "kv_channels": 128,
            "layer_norm_epsilon": 1e-06,
            "max_position_embeddings": 8192,
            "model_type": "qwen",
            "no_bias": True,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "onnx_safe": None,
            "rotary_emb_base": 10000,
            "rotary_pct": 1.0,
            "scale_attn_weights": True,
            "seq_length": 2048,
            "tie_word_embeddings": False,
            "tokenizer_type": "QWenTokenizer",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.31.0",
            "use_cache": True,
            "use_dynamic_ntk": True,
            "use_flash_attn": False,
            "use_logn_attn": True,
            "visual": {
                "heads": 16,
                "image_size": 448,
                "image_start_id": 151857,
                "layers": 48,
                "mlp_ratio": 4.9231,
                "output_dim": 4096,
                "patch_size": 14,
                "width": 1664
            },
            "vocab_size": 151936
            }
    elif "llava-v1.5-7b" in name.lower():
        config={
            "_name_or_path": "llava-v1.5-7b",
            "architectures": [
                "LlavaLlamaForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "freeze_mm_mlp_adapter": False,
            "freeze_mm_vision_resampler": False,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "image_aspect_ratio": "pad",
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_length": 4096,
            "max_position_embeddings": 4096,
            "mm_hidden_size": 1024,
            "mm_projector_type": "mlp2x_gelu",
            "mm_resampler_type": None,
            "mm_use_im_patch_token": False,
            "mm_use_im_start_end": False,
            "mm_vision_select_feature": "patch",
            "mm_vision_select_layer": -2,
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
            "model_type": "llava",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.31.0",
            "tune_mm_mlp_adapter": False,
            "tune_mm_vision_resampler": False,
            "unfreeze_mm_vision_tower": False,
            "use_cache": True,
            "use_mm_proj": True,
            "vocab_size": 32000
            }
    elif "llava-v1.5-13b" in name.lower():
        config={
            "_name_or_path": "llava-v1.5-13b",
            "architectures": [
                "LlavaLlamaForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "freeze_mm_mlp_adapter": False,
            "freeze_mm_vision_resampler": False,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "image_aspect_ratio": "pad",
            "initializer_range": 0.02,
            "intermediate_size": 13824,
            "max_length": 4096,
            "max_position_embeddings": 4096,
            "mm_hidden_size": 1024,
            "mm_projector_type": "mlp2x_gelu",
            "mm_resampler_type": None,
            "mm_use_im_patch_token": False,
            "mm_use_im_start_end": False,
            "mm_vision_select_feature": "patch",
            "mm_vision_select_layer": -2,
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
            "model_type": "llava",
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 40,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.31.0",
            "tune_mm_mlp_adapter": False,
            "tune_mm_vision_resampler": False,
            "unfreeze_mm_vision_tower": False,
            "use_cache": True,
            "use_mm_proj": True,
            "vocab_size": 32000
            }
    elif "InternVL2-8B" in name.lower():
        config = {
            "_commit_hash": None,
            
            "architectures": [
                "InternVLChatModel"
            ],
            "auto_map": {
                "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
                "AutoModel": "modeling_internvl_chat.InternVLChatModel",
                "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
            },
            "downsample_ratio": 0.5,
            "dynamic_image_size": True,
            "force_image_size": 448,
            "num_hidden_layers": 32,
            "llm_config": {
                "_name_or_path": "internlm/internlm2_5-7b-chat",
                "add_cross_attention": False,
                "architectures": [
                "InternLM2ForCausalLM"
                ],
                "attn_implementation": "flash_attention_2",
                "auto_map": {
                "AutoConfig": "configuration_internlm2.InternLM2Config",
                "AutoModel": "modeling_internlm2.InternLM2ForCausalLM",
                "AutoModelForCausalLM": "modeling_internlm2.InternLM2ForCausalLM"
                },
                "bad_words_ids": None,
                "begin_suppress_tokens": None,
                "bias": False,
                "bos_token_id": 1,
                "chunk_size_feed_forward": 0,
                "cross_attention_hidden_size": None,
                "decoder_start_token_id": None,
                "diversity_penalty": 0.0,
                "do_sample": False,
                "early_stopping": False,
                "encoder_no_repeat_ngram_size": 0,
                "eos_token_id": 2,
                "exponential_decay_length_penalty": None,
                "finetuning_task": None,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
                },
                "initializer_range": 0.02,
                "intermediate_size": 14336,
                "is_decoder": False,
                "is_encoder_decoder": False,
                "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
                },
                "length_penalty": 1.0,
                "max_length": 20,
                "max_position_embeddings": 32768,
                "min_length": 0,
                "model_type": "internlm2",
                "no_repeat_ngram_size": 0,
                "num_attention_heads": 32,
                "num_beam_groups": 1,
                "num_beams": 1,
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_return_sequences": 1,
                "output_attentions": False,
                "output_hidden_states": False,
                "output_scores": False,
                "pad_token_id": 2,
                "prefix": None,
                "pretraining_tp": 1,
                "problem_type": None,
                "pruned_heads": {},
                "remove_invalid_values": False,
                "repetition_penalty": 1.0,
                "return_dict": True,
                "return_dict_in_generate": False,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                "factor": 2.0,
                "type": "dynamic"
                },
                "rope_theta": 1000000,
                "sep_token_id": None,
                "suppress_tokens": None,
                "task_specific_params": None,
                "temperature": 1.0,
                "tf_legacy_loss": False,
                "tie_encoder_decoder": False,
                "tie_word_embeddings": False,
                "tokenizer_class": None,
                "top_k": 50,
                "top_p": 1.0,
                "torch_dtype": "bfloat16",
                "torchscript": False,
                "transformers_version": "4.37.2",
                "typical_p": 1.0,
                "use_bfloat16": True,
                "use_cache": True,
                "vocab_size": 92553
            },
            "max_dynamic_patch": 12,
            "min_dynamic_patch": 1,
            "model_type": "internvl_chat",
            "ps_version": "v2",
            "select_layer": -1,
            "template": "internlm2-chat",
            "torch_dtype": "bfloat16",
            "use_backbone_lora": 0,
            "use_llm_lora": 0,
            "use_thumbnail": True,
            "vision_config": {
                "architectures": [
                "InternVisionModel"
                ],
                "attention_dropout": 0.0,
                "drop_path_rate": 0.0,
                "dropout": 0.0,
                "hidden_act": "gelu",
                "hidden_size": 1024,
                "image_size": 448,
                "initializer_factor": 1.0,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "layer_norm_eps": 1e-06,
                "model_type": "intern_vit_6b",
                "norm_type": "layer_norm",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 24,
                "output_attentions": False,
                "output_hidden_states": False,
                "patch_size": 14,
                "qk_normalization": False,
                "qkv_bias": True,
                "return_dict": True,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.37.2",
                "use_bfloat16": True,
                "use_flash_attn": True
            }
        }

    elif "InternVL2-26B" in name.lower():
        config = {
    "_commit_hash": None,
    "architectures": [
        "InternVLChatModel"
    ],
    "auto_map": {
        "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
        "AutoModel": "modeling_internvl_chat.InternVLChatModel",
        "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
    },
    "downsample_ratio": 0.5,
    "dynamic_image_size": True,
    "force_image_size": 448,
    "num_hidden_layers": 48,
    "llm_config": {
        "_name_or_path": "internlm/internlm2-chat-20b",
        "add_cross_attention": False,
        "architectures": [
        "InternLM2ForCausalLM"
        ],
        "attn_implementation": "flash_attention_2",
        "auto_map": {
        "AutoConfig": "configuration_internlm2.InternLM2Config",
        "AutoModel": "modeling_internlm2.InternLM2ForCausalLM",
        "AutoModelForCausalLM": "modeling_internlm2.InternLM2ForCausalLM"
        },
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bias": False,
        "bos_token_id": 1,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 16384,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 32768,
        "min_length": 0,
        "model_type": "internlm2",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 48,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 2,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
        "factor": 3.0,
        "type": "dynamic"
        },
        "rope_theta": 1000000,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": False,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": None,
        "transformers_version": "4.37.2",
        "typical_p": 1.0,
        "use_bfloat16": True,
        "use_cache": True,
        "vocab_size": 92553
    },
    "max_dynamic_patch": 12,
    "min_dynamic_patch": 1,
    "model_type": "internvl_chat",
    "ps_version": "v2",
    "select_layer": -1,
    "template": "internlm2-chat",
    "torch_dtype": "bfloat16",
    "use_backbone_lora": 0,
    "use_llm_lora": 0,
    "use_thumbnail": True,
    "vision_config": {
        "architectures": [
        "InternVisionModel"
        ],
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 3200,
        "image_size": 448,
        "initializer_factor": 0.1,
        "initializer_range": 1e-10,
        "intermediate_size": 12800,
        "layer_norm_eps": 1e-06,
        "model_type": "intern_vit_6b",
        "norm_type": "rms_norm",
        "num_attention_heads": 25,
        "num_channels": 3,
        "num_hidden_layers": 45,
        "output_attentions": False,
        "output_hidden_states": False,
        "patch_size": 14,
        "qk_normalization": True,
        "qkv_bias": False,
        "return_dict": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.37.2",
        "use_bfloat16": True,
        "use_flash_attn": True
    }
    }

    else:
        raise NotImplementedError

    return  config
