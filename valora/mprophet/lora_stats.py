from valora.mprophet.constants import GB, T, get_num_bytes
from valora.mprophet.hardware_parameters import TFLOPS
from valora.mprophet.model_config import ModelConfig
from valora.mprophet.lora_config import LoRAConfig
from valora.mprophet.measure import ModelProphet


class LoRAProphet:

    def __init__(self, name: str, base_name: str,
                 lora_config=None, adapter_dir=None,
                 base_config=None, base_model_dir=None):
        self.name = name
        self.lora_config = LoRAConfig(name, config=lora_config, weight_dir=adapter_dir)

        self.base_name = base_name
        self.base_model = ModelProphet(base_name, config=base_config, model_dir=base_model_dir)
        self.base_config = self.base_model.model_config


    def get_layer_size(self, dtype="fp16"):
        dbytes = get_num_bytes(dtype)
        m = self.base_config
        size = dbytes * (m.hidden_size * self.lora_config.rank * 2 * 4)
        return size


    def get_adapter_size(self, dtype="fp16"):
        return self.get_layer_size(dtype) * self.base_config.num_hidden_layers


    def get_base_size(self, dtype="fp16"):
        return self.base_model.get_model_size(dtype=dtype)


if __name__ == "__main__":
    # alpaca = LoRAProphet("alpaca-lora-7b", "llama-7b",
    #                       adapter_dir="/home/ubuntu/models/lora-adapters/llama-7b/alpaca-lora-7b",
    #                       base_model_dir="/home/ubuntu/models/llama-7b-hf")

    # bactrian  = LoRAProphet("bactrian-x-llama-7b-lora", "llama-7b",
    #                       adapter_dir="/home/ubuntu/models/lora-adapters/llama-7b/bactrian-x-llama-7b-lora",
    #                       base_model_dir="/home/ubuntu/models/llama-7b-hf")

    # for adapter in [alpaca, bactrian]:
    #     print("=" * 10, adapter.name, "=" * 10)
    #     print(f"adapter size (GB): {adapter.get_adapter_size() / GB:.2f}")
    #     print(f"base size (GB): {adapter.get_base_size() / GB:.2f}")
    Qwen = LoRAProphet("airbus-lora", "QwenVL",
                       adapter_dir="/data01/tuwenming/Qwen-VL/LoRAs/airbus1500",
                       base_model_dir="/data01/tuwenming/models/Qwen-VL-Chat")
    for adapter in [Qwen]:
        print("=" * 10, adapter.name, "=" * 10)
        print(f"adapter size (GB): {adapter.get_adapter_size() / GB:.2f}")
        print(f"base size (GB): {adapter.get_base_size() / GB:.2f}")

 
 
