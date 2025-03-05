from huggingface_hub import snapshot_download

# snapshot_download(repo_id="huggyllama/llama-7b",
#                   local_dir="/data01/tuwenming/S-LoRA/models/huggyllama/llama-7b")

# snapshot_download(repo_id="huggyllama/llama-13b",
#                   local_dir="/data01/tuwenming/S-LoRA/models/huggyllama/llama-13b")

snapshot_download(repo_id="Qwen/Qwen-VL",
                  local_dir="/data01/tuwenming/S-LoRA/models/Qwen/Qwen-VL")