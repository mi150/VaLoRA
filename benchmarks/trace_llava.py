from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import re
import base64
torch.manual_seed(1234)


# class Request:
#     def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time):
#         self.req_id = req_id
#         self.model_dir = model_dir
#         self.adapter_dir = adapter_dir
#         self.prompt = prompt
#         self.prompt_len = prompt_len
#         self.output_len = output_len
#         self.req_time = req_time
#
#     def __repr__(self):
#         return f"req_id={self.req_id}, " \
#                f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
#                f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
#                f"req_time={self.req_time}, max_new_tokens={40}"


class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time, parameters,
                 multimodal_params):
        self.req_id = req_id
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time
        self.parameters = parameters
        self.max_new_tokens = parameters["max_new_tokens"]
        self.multimodal_params = multimodal_params

    def __repr__(self):
        return (
            f"req_id={self.req_id}, "
            f"model_dir={self.model_dir}, lora_dir={self.adapter_dir}, "
            f"prompt_len={self.prompt_len}, output_len={self.output_len}, "
            f"req_time={self.req_time}, parameters={self.parameters}, "
            f"inputs={self.prompt},multimodal_params={self.multimodal_params}, max_new_tokens={self.max_new_tokens}"
        )


def dummy_prompt(prompt_len, num):
    # return "Hello " * (prompt_len - 2 + 258)
    # return "<img></img>" * num + "Hello " * (prompt_len)
    return "<image>"*num+"Hello " * (prompt_len)
    #qwen,internvl<img>
    #llava <image><image>
    # return "How Are You?"



def get_arrival_times(req_rate,
                      file_path='benchmarks/real_trace/AzureLLMInferenceTrace_code_1week.csv'):
    data = pd.concat(tqdm(pd.read_csv(file_path, iterator=True, chunksize=1000), desc="Reading CSV", unit="chunk"))
    data = data.head(500)

    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])

    np.random.seed(42)

    arrival_times = []

    data['SECOND'] = data['TIMESTAMP'].dt.floor('S')
    unique_seconds = data['SECOND'].unique()

    start_time = data['TIMESTAMP'].min().floor('D').timestamp()

    for second in unique_seconds:
        requests_in_second = data[data['SECOND'] == second]

        if len(requests_in_second) >= req_rate:
            sampled_requests = requests_in_second.sample(n=req_rate, random_state=42)
            arrival_times.extend(sampled_requests['TIMESTAMP'].apply(lambda x: x.timestamp() - start_time).tolist())

    return arrival_times


def generate_requests(json_file, trace_file, num_adapters, rate,
                      adapter_dirs,  # (base_dir, adapter_dir)
                      seed=42, skewness=0.5, task='vqa', is_ours=False):
    np.random.seed(seed)

    print(len(adapter_dirs),num_adapters)
    arr_time = get_arrival_times(rate, trace_file)
    tot_req = len(arr_time)
    # tot_req = 1
    # generate adapter id
    # valora trace
    # probs = np.random.power(alpha, tot_req)
    #
    # ind = (probs * num_adapters).astype(int)

    ind = [1 if i < int(tot_req * skewness) else random.randint(0, num_adapters-1) for i in range(tot_req)]
    # ind = [0 for i in range(tot_req)]
    # ind[-1] = 1
    # ind = [i%num_adapters for i in range(tot_req)]
    # generate input output len
    # input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    # output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    if task == 'vqa':
        token_counts = process_json(json_file, is_ours=is_ours)
    else:
        token_counts = process_detect_json(json_file, is_ours=is_ours)
    fixed_input_len = 1022
    fixed_output_len = 1
    # use fixed input and output lengths
    input_lens = np.full(tot_req, fixed_input_len)
    output_lens = np.full(tot_req, fixed_output_len)

    uri = "datasets/test_dummy.jpg"

    with open(uri, 'rb') as fin:
        b64 = base64.b64encode(fin.read()).decode("utf-8")
    if task == 'vqa':
        images = [{'type': "base64", "data": b64}]
    else:
        images = [{'type': "base64", "data": b64} for _ in range(6)]
    fixed_input_len = 128
    fixed_output_len = 256
    # use fixed input and output lengths
    input_lens = np.full(tot_req, fixed_input_len)
    output_lens = np.full(tot_req, fixed_output_len)

    # generate timestamp
    requests = []
    tic = 0
    # shape = 1 / (cv * cv)
    # scale = cv * cv / req_rate
    # # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    # # 访问时间？
    # intervals = np.random.gamma(shape, scale / 10, tot_req)
    intervals = [1 if i == 4 else 0 for i in range(tot_req)]
    # print(intervals)

    # intervals = np.zeros(tot_req)
    # print(intervals,scale,shape)
    # print(token_counts)
    for i in range(tot_req):
        # tic += intervals[i]
        # print(ind[i])
        requests.append(Request(
            req_id=int(i),
            model_dir=adapter_dirs[ind[i]][0],
            adapter_dir=adapter_dirs[ind[i]][1],
            prompt=dummy_prompt(int(token_counts[i][0]), int(token_counts[i][2])),
            prompt_len=int(token_counts[i][0]),
            # output_len=1,
            output_len=min(120, int(token_counts[i][1])),
            # output_len=min(120, int(token_counts[i][1])),
            req_time=float(arr_time[i]),
            parameters={
                "max_new_tokens": min(120, int(token_counts[i][1])),
                "stop_sequences": ["\n"]
            },
            multimodal_params={
                "images": [{'type': "base64", "data": b64} for _ in range(int(token_counts[i][2]))]
            }
        ))
    return requests


def get_real_requests(trace_file, req_rate, duration, base_model, adapter_dirs, input_range, output_range, seed=42):
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    conversations = downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range)
    model_mapping = generate_model_mapping(conversations, adapter_dirs)
    conversations = sort_and_rescale_by_req_time(conversations, duration)
    reqs = parse_into_req(base_model, conversations, model_mapping, tokenizer)
    return model_mapping.values(), reqs


# functions below are used to generate real requests
def downsample(json_file, req_rate, duration, tokenizer, input_range, output_range):
    with open(json_file, "r") as file:
        all_conversations = json.load(file)

    more_ratio = 2
    need_num = int(req_rate * duration)
    # sample a bit more than needed
    selected_indicies = np.random.choice(len(all_conversations), more_ratio * need_num, replace=False)
    downsampled_conversations = [all_conversations[idx] for idx in selected_indicies]
    for idx, conv in enumerate(downsampled_conversations):
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        if prompt_len >= input_range[1] or output_len >= output_range[1]:
            # to avoid OOM in some configurations
            downsampled_conversations.pop(idx)
    downsampled_conversations = downsampled_conversations[:need_num]
    print(f"Downsampled {len(downsampled_conversations)}")
    return downsampled_conversations


def generate_model_mapping(conversations, adapter_dirs):
    model_mapping = {}
    num_ranks = [0] * len(adapter_dirs)
    for conv in conversations:
        model = conv["model"]
        if model not in model_mapping.keys():
            adapter_dir = random.choice(adapter_dirs)
            name = f"{adapter_dir}-{num_ranks[adapter_dirs.index(adapter_dir)]}"
            num_ranks[adapter_dirs.index(adapter_dir)] += 1
            model_mapping[model] = name
    print(model_mapping)
    return model_mapping


def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp'])
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) * duration
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations


def parse_into_req(base_model, conversations, model_mapping, tokenizer):
    reqs = []
    for idx, conv in enumerate(tqdm(conversations, desc="parse into reqs")):
        model = conv["model"]
        name = model_mapping[model]
        # print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)

        req = Request(req_id=idx, model_dir=base_model, adapter_dir=name,
                      prompt=conv["conversation"][0]["content"], prompt_len=prompt_len,
                      output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    # print(reqs)
    return reqs


def process_json(path="InternVL/internvl_chat/playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.json", is_ours=False):
    import json

    with open(path, 'r') as file:
        data = json.load(file)


    word_counts = []

    for item in data:
        conversations = item.get("conversations", [])

        human_word_count = 0
        gpt_word_count = 0

        for conversation in conversations:
            if conversation["from"] == "human":
                human_word_count = len(conversation["value"].split())
            elif conversation["from"] == "gpt":
                gpt_word_count = len(conversation["value"].split())

        if is_ours:
            word_counts.append((8, 128, 1))
        else:
            word_counts.append((human_word_count, gpt_word_count, 1))

    return word_counts


def process_detect_json(path="Qwen-VL/data/airbus+cityscape/train/"
                                                          "airbus+cityscape1classes.json", is_ours=False):
    tokenizer = AutoTokenizer.from_pretrained(
        "/data01/tuwenming/models/Qwen-VL-Chat", trust_remote_code=True
    )
    with open(path, 'r') as file:
        data = json.load(file)

    token_counts = []

    ref_pattern = re.compile(r'<ref>(.*?)</ref>')

    for item in data[:700]:
        conversations = item.get("conversations", [])

        human_token_count = 0
        gpt_token_count = 0

        for conversation in conversations:
            if conversation["from"] == "user":

                tokens = tokenizer.tokenize(conversation["value"])
                human_token_count += len(tokens)
            elif conversation["from"] == "assistant":
                tokens = tokenizer.tokenize(conversation["value"])
                gpt_token_count += len(tokens)
        if is_ours:
            token_counts.append((256, 40, 1))
        else:
            token_counts.append((human_token_count, gpt_token_count, 1))

    for _ in range(700):
        if is_ours:
            token_counts.append((0, 1, 3))
        else:
            token_counts.append((150, 7, 3))
    random.shuffle(token_counts)
    return token_counts
