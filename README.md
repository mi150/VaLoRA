# VaLoRA

## Abstract
Large Multimodal Models (LMMs) have shown significant progress in various complex vision tasks with the solid linguistic and reasoning capacity inherited from large language models (LMMs).
Low-rank adaptation (LoRA) offers a promising method to integrate external knowledge into LMMs, compensating for their limitations on domain-specific tasks.
However, the existing LoRA model serving is excessively computationally expensive and causes extremely high latency.
In this paper, we present an end-to-end solution that empowers diverse vision tasks and enriches vision applications with LoRA LMMs.
Our system, VaLoRA, enables accurate and efficient vision tasks by 1) an accuracy-aware LoRA adapter generation approach that generates LoRA adapters rich in domain-specific knowledge to meet application-specific accuracy requirements, 2) an adaptive-tiling LoRA adapters batching operator that efficiently computes concurrent heterogeneous LoRA adapters, 
and 3) a flexible LoRA adapter orchestration mechanism that manages application requests and LoRA adapters to achieve the lowest average response latency.
We prototype VaLoRA on five popular vision tasks on three LMMs.
Experiment results reveal that VaLoRA improves 24-62% of the accuracy compared to the original LMMs and reduces 20-89% of the latency compared to the state-of-the-art LoRA model serving systems.

## Requirements
* CUDA 11.8 compatible GPU
  * Recommended: GPUs from the Ampere family, like the A100, which support bfloat16 operations.
  * Note: Older GPUs from the Turing family like the T4, which do not support bfloat16, are not supported.
* 1.13 <= PyTorch <= 2.0.1

## Installation
```bash
conda create -n valora python=3.9
conda activate valora 
# Optional: Install CUDA via conda for a smoother installation experience,
# but you may need to manually set the Anaconda path variables.
# conda install cuda -c nvidia/label/cuda-11.8.0
# set environment variables: export TORCH_CUDA_ARCH_LIST="8.0 8.6"
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .
# install ATMM
cd atmm
pip install -e .
```
Make sure triton==2.1.0

## Prepare the dataset and model
Modify the path of base models and LoRA adapters in exp_suite.py.
For example, modify BASE_MODEL to use a custom model path like this: 
```bash
BASE_MODEL = {"qwenvl": "your_model_path"}
LORA_DIR = {"your_adapter_path",}
```


## Example run

You can prepare the LoRA adapters based on these open-source repositories ([Qwen-VL](https://github.com/QwenLM/Qwen-VL), [Intern-VL](https://github.com/OpenGVLab/InternVL))

If you want to train an adapter for video analytics tasks with vision task head, you can follow Qwen-VL/modeling_qwen.py to modify QWenLMHeadModel_AR class and follow Qwen-VL/finetune_ar.py to modify your finetuning code.

Dummy weights: Change $TASK with the task type to be tested (e.g., vqa, vat), and $TRACE and $DATASET refer to the request trace (like [AzureLLMInferenceTrace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)) and the dataset (like [sharegpt4v](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json)) to be used, respectively.
```bash
cd benchmarks
python launch_server.py --num-adapter 16 --num-token 30000 --model-setting qwenvl --scheduler ours
python run_exp.py --debug --model-setting qwenvl --num_adapters 16 --skewness 0.6 --req_rate 2 --task $TASK --trace_file $TRACE --dataset $DATASET
```

Test kernels performance:
```bash
cd test/kernel
python test_kernel_correctness.py
```

You can draw the fig.14 and fig.17 in the paper with the following code：


```bash
# For fig.14, change the scheduler(slora, punica, dlora) to the baseline that you want to compare.
cd benchmarks
python launch_server.py --num-adapter 16 --num-token 30000 --model-setting qwenvl --scheduler ours
python run_exp.py --debug --model-setting qwenvl --num_adapters 16 --skewness 0.8 --req_rate 6 --task vqa --trace_file
# Replacement data and run
cd ../draw
python fig14.py
```
[Fig.14](./draw/compare_vqa_var.pdf)

```bash
# For fig.17
cd test/kernel
python test_kernel_correctness.py.py
cd ../../draw
# Replacement test data and run
python fig17.py
```

[Fig.17](./draw/opcost.pdf)

## Citation

If you use VaLoRA for your research, please cite our [paper](https://arxiv.org/pdf/2411.00915):

```bibtex
@inproceedings{mi2025VaLoRA,
    title={Empower Vision Applications with LoRA LMM}, 
    author={Liang, Mi and Weijun, Wang and Wenming, Tu and Qingfeng, He and Rui, Kong and Xinyu, Fang and Yazhu, Dong and Yikang, Zhang and Yunchun, Li and Meng, Li and Haipeng, Dai and Guihai, Chen and Yunxin, Liu},
    year={2025},
    booktitle = {ACM European Conference on Computer Systems (EuroSys)},
}
```
