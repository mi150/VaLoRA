"""
To run with real mode:
python run_exp.py --backend valora --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend valora --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""
import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple
import logging
import aiohttp

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


"""
    add debug configs 

"""


def debug():
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        logging.info("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass


from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests
import trace_llava
sys.path.append("../bench_lora")
from valora.utils.metric import reward, attainment_func

GB = 1024 ** 3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
vllm_packed_adapter_dir_to_url_map = {}


# def get_peak_mem(server):
#     url = server + "/get_peak_mem"
#     response = requests.post(url)
#     return response.json()["peak_mem"]


async def send_request(
        backend: str,
        server: str,
        req_id: str,
        model_dir: str,
        adapter_dir: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        multimodal_params,
        debug: bool,
) -> None:

    request_start_time = time.time()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        url = server + "/generate"
    elif backend == "vllm-packed":
        url = vllm_packed_adapter_dir_to_url_map[adapter_dir] + "/generate"
    else:
        url = server + "/generate_stream"
    # 构建输入的数据
    if backend in ["ours", "valora", "strawman"]:
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'req_id': req_id,
            'multimodal_params': multimodal_params,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                # 'temperature': 0.1,
            }
        }
    elif backend in ["lightllm"]:
        data = {
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                # 'temperature': 0.1,
            },
        }
    elif backend in ["vllm", "vllm-packed"]:
        data = {
            'prompt': prompt,
            'max_tokens': output_len,
            'ignore_eos': True,
        }

    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    tokens_info = []

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    # token_data = json.loads(chunk)
                    # tokens_info.append(token_data)
                    # logging.info(chunk)

                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")


            if '"finished": -1' not in output:
                break
            else:
                first_token_latency = None
                break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    logging.info(f"req_id {req_id} prompt_len {prompt_len} output_len {output_len} "
                 f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s")
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency, first_token_latency))
    return (prompt_len, output_len, request_latency, first_token_latency)


async def benchmark(
        backend: str,
        server: str,
        input_requests: List[Tuple[str, str, str, int, int]],
        debug=False,
) -> None:

    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            logging.info(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                         f"{req.adapter_dir}")
        task = asyncio.create_task(send_request(backend, server,
                                                req.req_id, req.model_dir, req.adapter_dir, req.prompt,
                                                req.prompt_len, req.output_len, req.multimodal_params, debug))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency



def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = num_adapters // len(adapter_dirs) + 1

    if backend == "vllm-packed":
        num_iter = num_adapters // len(adapter_dirs)

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    return ret


def get_res_stats(per_req_latency, benchmark_time, backend, warmup_time=0, warmup_num=0, avg_len=0):
    # get throughput
    num_abort = len([i for i in per_req_latency if i[3] is None])
    per_req_latency = [i for i in per_req_latency if i[3] is not None]
    throughput = len(per_req_latency) / benchmark_time
    # logging.info(per_req_latency)
    # if backend == "valora":
    #     peak_mem = get_peak_mem(server)
    #     logging.info(f"GPU peak memory (GB):", [[f"{x / GB:.2f}" for x in tpg] for tpg in peak_mem])
    logging.info(f"Total time: {benchmark_time:.6f} s")
    logging.info(f"Aborted Request: {num_abort}")
    logging.info(f"Throughput: {throughput:.6f} requests/s")

    strip_throughput = (len(per_req_latency) - warmup_num * 2) / (benchmark_time - warmup_time * 2)
    logging.info(f"Throughput strip: {strip_throughput:.6f} requests/s")

    # compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _ in per_req_latency])
    logging.info(f"Average latency: {avg_latency:.6f} s")
    avg_per_token_latency = avg_latency / avg_len
    # avg_per_token_latency = np.mean([
    #     latency / (prompt_len + output_len)
    #     for prompt_len, output_len, latency, _ in per_req_latency
    # ])
    logging.info(f"Average latency per token: {avg_per_token_latency:.6f} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency, _ in per_req_latency
    ])
    logging.info("Average latency per output token: "
                 f"{avg_per_output_token_latency:.6f} s")

    # compute the first token latency
    first_token_latency = [latency for _, _, _, latency in per_req_latency]
    avg_first_token_latency = np.mean(first_token_latency)
    logging.info(f"Average first token latency: {avg_first_token_latency:.6f} s")
    logging.info(f"90 percentile first token latency: < {np.percentile(first_token_latency, 90):.6f} s")
    logging.info(f"50 percentile first token latency: < {np.percentile(first_token_latency, 50):.6f} s")
    abort_satisfaction = [0] * num_abort
    satisfaction = [reward(latency) for _, _, _, latency in per_req_latency] + abort_satisfaction
    avg_satisfaction = np.mean(satisfaction)
    logging.info(f"Average satisfaction: {avg_satisfaction:.2f}")
    logging.info(f"90 percentile satisfaction: > {np.percentile(satisfaction, 10):.2f}")
    logging.info(f"50 percentile satisfaction: > {np.percentile(satisfaction, 50):.2f}")

    attainment = [attainment_func(latency) for _, _, _, latency in per_req_latency] + abort_satisfaction
    avg_attainment = np.mean(attainment)
    logging.info(f"Average attainment: {avg_attainment:.2f}")

    # dump results
    if backend == "valora":
        # TODO
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {"total_time": benchmark_time, "gpu_peak_mem": single_gpu_peak_mem, "num_abort": num_abort,
              "throughput": throughput, "strip_throughput": strip_throughput,
              "avg_latency": avg_latency, "avg_per_token_latency": avg_per_token_latency,
              "avg_per_output_token_latency": avg_per_output_token_latency,
              "avg_first_token_latency": avg_first_token_latency,
              "avg_satisfaction": avg_satisfaction,
              "avg_attainment": avg_attainment}
    res = {"config": to_dict(config), "result": result}

    return res


def run_exp(model_setting, backend, server, dataset_path, trace_file, output, mode, num_adapters, req_rate, skewness, task, seed=42,
            debug=False, is_ours=False):
    # if mode == "real":
    #     logging.info("*** num_adapters, cv and alpha are not used in real mode ***")
    # logging.info([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    # num_adapters, alpha, req_rate, cv, duration, input_range, output_range = config
    # assert duration >= 30
    if mode == "synthetic":  # Prompt为 "Hello" * Prompt_len
        base_model = BASE_MODEL[model_setting]
        adapter_dirs = LORA_DIR[model_setting]
        # logging.info(adapter_dirs, model_setting)
        adapter_dirs = get_adapter_dirs(num_adapters, adapter_dirs)
        adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]
        if num_adapters == 0:
            adapter_dirs = [(base_model, None)]
            num_adapters = 1
        if "llava" in model_setting:
            requests = trace_llava.generate_requests(dataset_path, trace_file, num_adapters, req_rate,
                                         adapter_dirs, seed, skewness=skewness, task=task, is_ours=is_ours)
        else:
            requests = generate_requests(dataset_path, trace_file, num_adapters, req_rate,
                                         adapter_dirs, seed, skewness=skewness, task=task, is_ours=is_ours)
        avg_prompt_len = np.mean([req.prompt_len for req in requests])
        avg_output_len = np.mean([req.output_len for req in requests])
        avg_len = np.mean([req.prompt_len + req.output_len + (586 if "llava" in model_setting else 256)
                           *len(req.multimodal_params["images"]) for req in requests])
        # print(avg_len+3.5*256)

        print("avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)
        # while 1: pass
    else:
        # first generate your data using real_trace/clean_chat_data.py
        base_model = BASE_MODEL[model_setting]
        adapter_dirs = LORA_DIR[model_setting]
        adapter_dirs, requests = get_real_requests(trace_file="real_trace/AzureLLMInferenceTrace_code_1week.csv",
                                                   req_rate=req_rate, duration=duration,
                                                   base_model=base_model, adapter_dirs=adapter_dirs,
                                                   input_range=input_range, output_range=output_range,
                                                   seed=seed)
        # logging.info(requests)
        avg_prompt_len = np.mean([req.prompt_len for req in requests])
        avg_output_len = np.mean([req.output_len for req in requests])
        avg_len = np.mean([req.prompt_len + req.output_len + 256*len(req.multimodal_params["images"]) for req in requests])
        # logging.info("num_adapters", len(adapter_dirs), "num_requests", len(requests), "avg_len:", avg_len,
        #              "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)

    if debug:
        logging.info(f"num requests: {len(requests)}")  # 120个request
        # for req in requests[:4]:
        #     logging.info(req)
        # for req in requests:
        #     logging.info(req)

    if backend == "vllm-packed":
        for i in range(len(adapter_dirs)):
            vllm_packed_adapter_dir_to_url_map[adapter_dirs[i][1]] = f"http://127.0.0.1:{8080 + i}"

    # benchmark
    benchmark_start_time = time.time()
    per_req_latency = asyncio.run(benchmark(backend, server, requests, debug))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    warmup_time = 10
    warmup_num = int(req_rate * warmup_time)
    res = get_res_stats(per_req_latency, benchmark_time, backend,
                        warmup_time=warmup_time, warmup_num=warmup_num, avg_len=avg_len)

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ours",
                        choices=["ours", "valora", "vllm", "lightllm", "vllm-packed"])
    parser.add_argument("--suite", type=str, default="default")

    parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-copy", action="store_true")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])

    parser.add_argument("--server", type=str, default="http://localhost:8085")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_adapters", type=int, default=16)
    parser.add_argument("--skewness", type=float, default=0.5)
    parser.add_argument("--req_rate", type=int, default=16)
    parser.add_argument("--task", default="vqa", choices=["vqa", "vat"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trace_file", type=str, default="real_trace/AzureLLMInferenceTrace_code_1week.csv")
    parser.add_argument("--dataset", type=str, default="datasets/sharegpt4v_instruct_gpt4-vision_cap100k.json")
    args = parser.parse_args()

    # debug()

    assert not args.no_lora_copy or args.no_lora_compute
    assert not (args.debug and args.breakdown)

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.mode}_" + args.backend + ".jsonl"
    if args.no_lora_swap and args.no_lora_compute and args.no_lora_copy:
        args.output = "no_lora_compute_swap_copy_results.jsonl"
    elif args.no_lora_swap and args.no_lora_compute:
        args.output = "no_lora_compute_swap_results.jsonl"
    elif args.no_lora_swap:
        args.output = "no_lora_swap_results.jsonl"
    elif args.no_lora_compute:
        args.output = "no_lora_compute_results.jsonl"
    if args.debug or args.breakdown:
        args.output = "debug_" + args.output

    suites = get_all_suites(mode=args.mode, debug=args.debug, suite=args.suite, breakdown=args.breakdown)

    # if not args.append:
    #     os.system(f"rm {args.output}")
    #     results = []
    # else:
    #     with open(args.output, "r") as f:
    #         lines = f.readlines()
    #     results = [json.loads(line)["config"] for line in lines]
    results = []

    log_dir = 'log/experiment'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f'{log_dir}/result_{args.server[-4:]}_{args.model_setting}_{args.num_adapters}_{args.req_rate}_{args.skewness}_{args.task}.log'
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                       format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.shutdown()
    # if int(args.server[-4:]) in [8085]:
    #     is_ours = True
    # else:
    #     is_ours = False
    for config in tqdm(suites, desc="suites"):
        if to_dict(config) not in results:
            run_exp(args.model_setting, args.backend, args.server, args.dataset, args.trace_file,
                    args.output, args.mode, args.num_adapters, args.req_rate, args.skewness, args.task, args.seed,
                    args.debug, True)
