import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from valora.utils.infer_utils import calculate_time
from collections import Counter
import logging
import torch
import time

logging.basicConfig(filename=f'log/others.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logging.shutdown()
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

no_lora_req = "Qwen-VL/LoRAs/refcoco/LoRA_A-0"

N = 32 * 4096
output_counts = torch.ones(N, dtype=torch.long, device="cuda")
rank_counts = torch.full((N,), 64, dtype=torch.long, device="cuda")
lora_ids = torch.zeros(N, dtype=torch.long, device="cuda")
start_ids = torch.ones(N, dtype=torch.long, device="cuda")
tmp_d = torch.zeros(N * 60, dtype=torch.int8, device="cuda")
weight_output_counts = torch.ones(N, dtype=torch.long, device="cuda") * 4096
weight_rank_counts = torch.ones(N, dtype=torch.long, device="cuda") * 128
weight_lora_ids = torch.arange(4096, dtype=torch.long, device="cuda")
weight_start_ids = torch.arange(4096, dtype=torch.long, device="cuda") * 4096
weight_tmp_d = torch.zeros(4096 * 60, dtype=torch.int8, device="cuda")

def reset_sgmm():
    global output_counts, rank_counts, lora_ids, start_ids, tmp_d
    N = 32 * 2048 * 2
    output_counts = torch.ones(N, dtype=torch.long, device="cuda")
    rank_counts = torch.full((N,), 64, dtype=torch.long, device="cuda")
    lora_ids = torch.zeros(N, dtype=torch.long, device="cuda")
    start_ids = torch.ones(N, dtype=torch.long, device="cuda")
    tmp_d = torch.zeros(N * 60, dtype=torch.int8, device="cuda")

class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        # for dlora
        self.alpha = 1
        self.beta = 0.5
        self.select_adapter = None
        self.delora_index = 0
        self.delora_tk_index = 0
        self.starve_req = []
        self.threshold = 1000

    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def _init_cache_list(self, current_batch: Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                            req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):

        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
                len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            print()
            return False

    def update_counter(self, req):
        pass

    def get_req_skewness(self):
        adapter_count = Counter(req.adapter_dir for req in self.waiting_req_list)
        most_common_adapter, most_common_count = adapter_count.most_common(1)[0]
        total_count = len(self.waiting_req_list)
        skewness = most_common_count / total_count if total_count > 0 else 0
        # print("Now skewness is:", skewness)
        return skewness

    def dlora_schedule(self, infer_mode):
        if infer_mode == "unmerge":
            adapter_count = Counter(req.adapter_dir for req in self.waiting_req_list)
            most_common_adapter, most_common_count = adapter_count.most_common(1)[0]
            if most_common_count / self.running_max_req_size > self.alpha:
                infer_mode = "merge"
                print("R / B = ", most_common_count / self.running_max_req_size, "> alpha, execute ",
                      most_common_adapter)

                return infer_mode, most_common_adapter
            else:
                print("R / B = ", most_common_count / self.running_max_req_size, "< alpha")
                return infer_mode, None
        else:
            most_common_count = len([req for req in self.waiting_req_list if req.adapter_dir == self.select_adapter])
            if most_common_count / self.running_max_req_size < self.beta:
                infer_mode = 'unmerge'
                print("R / B = ", most_common_count / self.running_max_req_size, "< beta")
                return infer_mode, None
            else:
                print("R / B = ", most_common_count / self.running_max_req_size, "> beta")
                return infer_mode, self.select_adapter

    def ours_schedule(self):
        # return "unmerge", None, []
        # how to get delora index, num of other adapters
        t1 = time.time()
        for req in self.waiting_req_list:
            req.credit = time.time() - req.init_time
        starve_req = [index for index, req in enumerate(self.waiting_req_list) if req.credit > self.threshold]
        # starve_req=[]
        max_bs = self.running_max_req_size
        if len(self.waiting_req_list) > 0:
            max_bs = min(max_bs, len(self.waiting_req_list))
        adapter_count = Counter(req.adapter_dir for req in self.waiting_req_list)
        most_common_adapter, most_common_count = adapter_count.most_common(1)[0]
        # return "merge", most_common_adapter, []
        # return "merge", most_common_adapter, []
        logging.info(
            f'ours_schedule: {time.time() - t1:.6f} seconds')
        if len(starve_req) / max_bs <= 0.5 and most_common_count / max_bs > 0.5:
            infer_mode = "merge"
            print(f"{YELLOW}change to merge{RESET}",  len(self.waiting_req_list), most_common_adapter, max_bs)
            return infer_mode, most_common_adapter, starve_req
        else:
            infer_mode = "unmerge"
            print(f"{YELLOW}change to unmerge{RESET}", len(starve_req), most_common_adapter, max_bs)
            return infer_mode, None, []


    def mode_scheduler(self, scheduler: str, last_infer_mode=None):
        if scheduler == "strawman":
            infer_mode = 'unmerge'
            # infer_mode = 'unmerge' if last_infer_mode == 'merge' else 'merge'
        elif scheduler == "dlora":
            # infer_mode, self.select_adapter, self.delora_index = self.ours_schedule(last_infer_mode)
            infer_mode, self.select_adapter = self.dlora_schedule(last_infer_mode)
            # infer_mode = "unmerge"
            # infer_mode = 'unmerge' if last_infer_mode == 'merge' else 'merge'
        elif scheduler == "ours":
            # get mode by our algorithm.
            # infer_mode, self.select_adapter = self.dlora_schedule(last_infer_mode)
            infer_mode, self.select_adapter, self.starve_req = self.ours_schedule()

        else:
            infer_mode = 'unmerge'
            # infer_mode = 'merge'
            # adapter_count = Counter(req.adapter_dir for req in self.waiting_req_list)
            # most_common_adapter, _ = adapter_count.most_common(1)[0]
            # self.select_adapter = most_common_adapter
        if infer_mode != last_infer_mode:
            print(f"{GREEN}Change infer mode from {last_infer_mode} to {infer_mode}{RESET}")
        return infer_mode

    def generate_new_batch(self, current_batch: Batch, lora_ranks: dict[str, int], scheduler=None, infer_mode=None):
        t1 =time.time()
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        if len(self.waiting_req_list) > 0:
            self.starve_req = []
            infer_mode = self.mode_scheduler(scheduler, infer_mode)

        self._init_cache_list(current_batch, lora_ranks)

        if infer_mode == 'merge':
            max_bs = self.running_max_req_size
            # if len(self.waiting_req_list) > 0:
            #     max_bs = min(max_bs, len(self.waiting_req_list))
            can_run_list = []
            new_batch_total_tokens = 0
            merge_cnt = len(self.starve_req)
            self.delora_tk_index = 0
            self.delora_index = 0
            for idx, req in enumerate(self.waiting_req_list.copy()):
                if (self._can_add_new_req(req, lora_ranks) and
                        new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    # print(req.adapter_dir, self.select_adapter, merge_cnt)
                    if idx in self.starve_req and scheduler == "ours":
                        print("add different lora to execute delora")
                        can_run_list.append(req)
                        new_batch_total_tokens += req.input_len
                        self.waiting_req_list.remove(req)
                    elif req.adapter_dir == self.select_adapter and merge_cnt < max_bs:
                        print("add merge lora to execute")
                        merge_cnt += 1
                        can_run_list.insert(0, req)
                        self.delora_tk_index += req.input_len
                        self.delora_index += 1
                        new_batch_total_tokens += req.input_len
                        self.waiting_req_list.remove(req)
                    else:
                        print("nothing add to execute", len(self.waiting_req_list), req.adapter_dir, self.select_adapter,
                              req.adapter_dir == self.select_adapter, merge_cnt, max_bs, merge_cnt < max_bs)
                else:
                    print("other error:",(self._can_add_new_req(req, lora_ranks)), (new_batch_total_tokens + req.input_len <= self.batch_max_tokens))
                    continue

        elif infer_mode == 'unmerge':
            can_run_list = []
            new_batch_total_tokens = 0
            aborted_count = 0
            self.delora_tk_index = 0
            self.delora_index = 0
            for req in self.waiting_req_list.copy():
                if req.aborted:
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                        new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    if scheduler != "ours":
                        can_run_list.append(req)
                        new_batch_total_tokens += req.input_len
                        self.waiting_req_list.remove(req)
                        continue
                    # elif req.adapter_dir == no_lora_req:
                    #     can_run_list.insert(0, req)
                    #     new_batch_total_tokens += req.input_len
                    #     self.delora_tk_index += req.input_len
                    #     self.delora_index += 1
                    else:
                        can_run_list.append(req)
                        new_batch_total_tokens += req.input_len
                    self.waiting_req_list.remove(req)
                else:
                    break

        else:
            # self.waiting_req_list = sorted(self.waiting_req_list, key=lambda req: req.adapter_dir)
            can_run_list = []
            new_batch_total_tokens = 0
            for req in self.waiting_req_list:
                if (self._can_add_new_req(req, lora_ranks) and
                        new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break

        if len(can_run_list) != 0:
            # print("waiting list", self.waiting_req_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            new_batch.calcu_sum_adapters(self.delora_index, self.delora_tk_index)
            print("Generate new batch size:", len(can_run_list))
            # self.waiting_req_list = self.waiting_req_list[len(can_run_list):]
            logging.info(
                f'generate new batch: {time.time() - t1:.6f} seconds')
            return new_batch, infer_mode, self.select_adapter
        else:
            if len(self.waiting_req_list) == 0:
                time.sleep(15)
                print("start")
            else:
                print(self.waiting_req_list)
            return None, None, None

    def next_batch(self):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None
