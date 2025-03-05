---
license: mit
datasets:
- yahma/alpaca-cleaned
---

This repo contains a low-rank adapter for LLaMA-7b
fit on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

This version of the weights was trained with the following hyperparameters:

- Epochs: 10 (load from best epoch)
- Batch size: 128
- Cutoff length: 512
- Learning rate: 3e-4
- Lora _r_: 16
- Lora target modules: q_proj, k_proj, v_proj, o_proj

That is:

```
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca-512-qkvo' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```

Instructions for running it can be found at https://github.com/tloen/alpaca-lora.