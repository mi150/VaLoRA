import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue, no_lora_req
from rpyc.utils.classic import obtain
from valora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from valora.server.input_params import InputParams
from valora.models.peft.lora_adapter import get_lora_config
from valora.server.router.profiler import AlphaModel, BetaModel
from valora.server.router.abort_req_queue import AbortReqQueue
from valora.server.router.cluster_req_queue import ClusterReqQueue
from valora.server.router.vtc_req_queue import VTCReqQueue
from valora.server.router.pets_req_queue import PETSReqQueue
from valora.server.router.peft_req_queue import PEFTReqQueue
import logging

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"


def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "valora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    elif input_params.scheduler in ["strawman", "dlora", "ours"]:
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[None] = 0
        self.req_queue = get_scheduler(input_params, adapter_dirs)

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 100
        self.merged_adapter = None
        self.infer_mode = "unmerge"

        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)

    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))

        await asyncio.gather(*init_model_ret)
        return

    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/valora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir + "/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return

    def add_req(
            self,
            adapter_dir: str,
            prompt_ids: List[int],
            sampling_params: SamplingParams,
            request_id: str
    ):
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        # for req in self.req_queue.waiting_req_list:
        #     if req.request_id == request_id:
        #         req.has_generate_finished = True
        #         req.aborted = True
        return

    async def loop_for_fwd(self, ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 1 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:",
                          self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                self.stats_tool.print_stats()

            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):

        self.max_wait_tokens = 10000
        if self.input_params.scheduler in ["strawman", "dlora", "ours"]:
            if self.running_batch is None:
                new_batch, infer_mode, select_adapter = self.req_queue.generate_new_batch(self.running_batch,
                                                                                          self.lora_ranks,
                                                                                          self.input_params.scheduler,
                                                                                          self.infer_mode)
                self.infer_mode = infer_mode if infer_mode is not None else self.infer_mode
                print(f"Processing new batching{self.req_queue.waiting_req_list}", end='\r', flush=True)
                if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                    print("enable_abort", self.input_params.enable_abort)
                    self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                    self.req_queue.reset_abort_list()
                if new_batch is not None:
                    print("batch is not none, mode = ", self.infer_mode, self.req_queue.waiting_req_list)
                    if self.infer_mode == "merge":
                        self.stats_tool.count_prompt_tokens(new_batch)
                        self.running_batch = new_batch
                        if not self.input_params.no_lora:
                            # load adapters
                            load_start_time = time.time()

                            ret = []
                            for tp_rank in range(self.world_size):
                                ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                            await asyncio.gather(*ret)
                            print("load cost:", time.time() - load_start_time)
                        # unmerge lora
                        torch.cuda.synchronize()
                        t1 = time.time()
                        ret = []
                        # need to unmerge?
                        for tp_rank in range(self.world_size):
                            if self.model_rpcs[tp_rank].is_merged and \
                                    select_adapter != self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir:
                                ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                                # self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = None
                        await asyncio.gather(*ret)
                        if time.time() - t1 > 1e-4:
                            print(f"{BLUE}first unmerge lora to base model:{RESET}", time.time() - t1)

                        # merge lora to base model
                        torch.cuda.synchronize()
                        ret = []
                        t1 = time.time()
                        for tp_rank in range(self.world_size):
                            if (self.model_rpcs[tp_rank].is_merged and
                                    select_adapter == self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir) \
                                    or select_adapter == no_lora_req:
                                self.model_rpcs[tp_rank].model.input_params.no_lora_compute = True
                                self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = select_adapter
                                continue
                            self.model_rpcs[tp_rank].model.input_params.no_lora_compute = True
                            self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = select_adapter
                            ret.append(self.model_rpcs[tp_rank].merge_adapter())
                        await asyncio.gather(*ret)
                        if time.time() - t1 > 1e-4:
                            print(f"{BLUE}first merge lora to base model:{RESET}", time.time() - t1)


                        pre_start_time = time.time()
                        await self._prefill_batch(self.running_batch)
                        torch.cuda.synchronize()
                        await self._filter_runing_batch()
                        self.has_wait_tokens = 0
                        print("prefill time:", time.time() - pre_start_time)
                    else:
                        self.stats_tool.count_prompt_tokens(new_batch)
                        self.running_batch = new_batch
                        if not self.input_params.no_lora:
                            # load adapters
                            load_start_time = time.time()

                            ret = []
                            for tp_rank in range(self.world_size):
                                ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                            await asyncio.gather(*ret)
                            print("load cost:", time.time() - load_start_time)

                        # unmerge lora
                        torch.cuda.synchronize()
                        t1 = time.time()
                        ret = []
                        # need to unmerge?

                        for tp_rank in range(self.world_size):
                            self.model_rpcs[tp_rank].model.input_params.no_lora_compute = False
                            if self.model_rpcs[tp_rank].is_merged:
                                # self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = None
                                ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                        await asyncio.gather(*ret)
                        if time.time() - t1 > 1e-4:
                            print(f"{BLUE}first unmerge lora to base model:{RESET}", time.time() - t1)
                        torch.cuda.synchronize()
                        pre_start_time = time.time()
                        await self._prefill_batch(self.running_batch)
                        torch.cuda.synchronize()
                        print("prefill time:", time.time() - pre_start_time)
                        await self._filter_runing_batch()
                        self.has_wait_tokens = 0

                return

            if self.has_wait_tokens < self.max_wait_tokens:
                self.stats_tool.count_output_tokens(self.running_batch)
                # prefetch
                if (not self.input_params.no_lora and
                        self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                                                        self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                    next_batch = self.req_queue.next_batch()
                    if next_batch is not None:
                        ret = []
                        for tp_rank in range(self.world_size):
                            ret.append(self.model_rpcs[tp_rank].load_adapters(
                                next_batch.adapter_dirs, prefetch=True))
                        await asyncio.gather(*ret)
                torch.cuda.synchronize()
                # print("self.has_wait_tokens < self.max_wait_tokens")
                dec_start_time = time.time()
                await self._decode_batch(self.running_batch)
                torch.cuda.synchronize()
                print("decode time:", time.time() - dec_start_time)
                await self._filter_runing_batch()

                self.has_wait_tokens += 1
                return
            else:
                new_mini_batch, infer_mode, select_adapter = self.req_queue.generate_new_batch(self.running_batch,
                                                                                          self.lora_ranks,
                                                                                          self.input_params.scheduler,
                                                                                          self.infer_mode)
                self.infer_mode = infer_mode if infer_mode is not None else self.infer_mode
                print("finish generate new batch")
                if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                    print("enable_abort", self.input_params.enable_abort)
                    self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                    self.req_queue.reset_abort_list()
                if new_mini_batch is not None:
                    print("batch is not none, mode = ",self.infer_mode)
                    if self.infer_mode == "merge":
                        self.stats_tool.count_prompt_tokens(new_mini_batch)
                        if not self.input_params.no_lora:
                            # load adapters
                            load_start_time = time.time()

                            ret = []
                            for tp_rank in range(self.world_size):
                                ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                            await asyncio.gather(*ret)
                            print("load cost:", time.time() - load_start_time)
                        # unmerge lora
                        torch.cuda.synchronize()
                        t1 = time.time()
                        ret = []
                        # need to unmerge?
                        for tp_rank in range(self.world_size):
                            if self.model_rpcs[tp_rank].is_merged and \
                                    select_adapter != self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir:
                                ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                                # self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = None
                        await asyncio.gather(*ret)
                        if time.time() - t1 > 1e-4:
                            print(f"{BLUE}first unmerge lora to base model:{RESET}", time.time() - t1)
                        # merge lora to base model
                        torch.cuda.synchronize()
                        ret = []
                        t1 = time.time()
                        for tp_rank in range(self.world_size):
                            if (self.model_rpcs[tp_rank].is_merged and
                                    select_adapter == self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir) \
                                    or select_adapter == no_lora_req:
                                self.model_rpcs[tp_rank].model.input_params.no_lora_compute = True
                                self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = select_adapter
                                continue
                            self.model_rpcs[tp_rank].model.input_params.no_lora_compute = True
                            self.model_rpcs[tp_rank].model.infer_adapter.merged_adapter_dir = select_adapter
                            ret.append(self.model_rpcs[tp_rank].merge_adapter())
                        await asyncio.gather(*ret)
                        if time.time() - t1 > 1e-4:
                            print(f"{BLUE}first merge lora to base model:{RESET}", time.time() - t1)

                        pre_start_time = time.time()
                        await self._prefill_batch(new_mini_batch, minibatch=True)
                        torch.cuda.synchronize()
                        self.has_wait_tokens = 0
                        print("prefill time:", time.time() - pre_start_time)
                    else:
                        self.stats_tool.count_prompt_tokens(new_mini_batch)
                        if not self.input_params.no_lora:
                            # load adapters
                            load_start_time = time.time()

                            ret = []
                            for tp_rank in range(self.world_size):
                                ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                            await asyncio.gather(*ret)
                            print("load cost:", time.time() - load_start_time)

                        # unmerge lora
                        torch.cuda.synchronize()
                        t1 = time.time()
                        ret = []
                        # need to unmerge?

                        for tp_rank in range(self.world_size):
                            self.model_rpcs[tp_rank].model.input_params.no_lora_compute = False
                            if self.model_rpcs[tp_rank].is_merged:
                                ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                        await asyncio.gather(*ret)
                        if time.time() - t1 > 1e-4:
                            print(f"{BLUE}first unmerge lora to base model:{RESET}", time.time() - t1)

                        pre_start_time = time.time()
                        await self._prefill_batch(new_mini_batch, minibatch=True)
                        torch.cuda.synchronize()
                        print("prefill time:", time.time() - pre_start_time)
                    if not new_mini_batch.is_clear():
                        print(self.running_batch.batch_id, new_mini_batch.batch_id)
                        await self._merge_batch(self.running_batch, new_mini_batch)
                        self.running_batch.merge(new_mini_batch)
                    self.has_wait_tokens = 0
                # if new_mini_batch is not None:
                #     self.stats_tool.count_prompt_tokens(new_mini_batch)
                #
                #     if not self.input_params.no_lora:
                #         ret = []
                #         # load_start_time = time.time()
                #         for tp_rank in range(self.world_size):
                #             ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                #         await asyncio.gather(*ret)
                #         # print("mini batch load cost:", time.time() - load_start_time)
                #         # if lora changed then unmerge adapter from base model
                #         ret = []
                #         for tp_rank in range(self.world_size):
                #             if self.model_rpcs[tp_rank].is_merged:
                #                 ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                #         await asyncio.gather(*ret)
                #         print(f"{BLUE}unmerge lora to base model{RESET}")
                #         # else merge lora to base model
                #         ret = []
                #         for tp_rank in range(self.world_size):
                #             if self.model_rpcs[tp_rank].is_merged:
                #                 ret.append(self.model_rpcs[tp_rank].merge_adapter())
                #         await asyncio.gather(*ret)
                #         print(f"{BLUE}merge lora to base model{RESET}")
                #
                #     pre_start_time = time.time()
                #     await self._prefill_batch(new_mini_batch, minibatch=True)
                #     if not new_mini_batch.is_clear():
                #         await self._merge_batch(self.running_batch, new_mini_batch)
                #         self.running_batch.merge(new_mini_batch)
                #     self.has_wait_tokens = 0
                #     print("prefill time:", time.time() - pre_start_time)
                else:
                    dec_start_time = time.time()
                    self.stats_tool.count_output_tokens(self.running_batch)
                    await self._decode_batch(self.running_batch)
                    await self._filter_runing_batch()
                    print("decode time:", time.time() - dec_start_time)

        else:
            step_start_time = time.time()
            if self.running_batch is None:
                # batch_start_time = time.time()
                new_batch, infer_mode, select_adapter = self.req_queue.generate_new_batch(self.running_batch,
                                                                                          self.lora_ranks,
                                                                                          self.input_params.scheduler,
                                                                                          self.infer_mode)
                # if new_batch is not None:
                # print("generate_new_batch cost:", time.time() - batch_start_time)
                if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                    self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                    self.req_queue.reset_abort_list()
                if new_batch is not None:
                    self.stats_tool.count_prompt_tokens(new_batch)
                    self.running_batch = new_batch

                    if not self.input_params.no_lora:
                        # load adapters
                        ret = []
                        load_start_time = time.time()
                        for tp_rank in range(self.world_size):
                            ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                        await asyncio.gather(*ret)
                        print("load cost:", time.time()-load_start_time)

                    # merge adapter to base model
                    if self.input_params.scheduler == "peft":
                        torch.cuda.synchronize()
                        ret = []
                        for tp_rank in range(self.world_size):
                            ret.append(self.model_rpcs[tp_rank].merge_adapter())
                        await asyncio.gather(*ret)
                    pre_start_time = time.time()
                    await self._prefill_batch(self.running_batch)
                    torch.cuda.synchronize()
                    # await self._filter_runing_batch()
                    self.has_wait_tokens = 0
                    print("prefill time:", time.time() - pre_start_time)
                return

            if self.has_wait_tokens < self.max_wait_tokens:
                self.stats_tool.count_output_tokens(self.running_batch)
                # prefetch
                if (not self.input_params.no_lora and
                        self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                                                        self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                    next_batch = self.req_queue.next_batch()
                    if next_batch is not None:
                        ret = []
                        load_start_time = time.time()
                        for tp_rank in range(self.world_size):
                            ret.append(self.model_rpcs[tp_rank].load_adapters(
                                next_batch.adapter_dirs, prefetch=True))
                        await asyncio.gather(*ret)
                        print("load cost:", time.time() - load_start_time)
                dec_start_time = time.time()
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
                print("decode time:", time.time() - dec_start_time)
                self.has_wait_tokens += 1
                return
            else:
                # batch_start_time = time.time()
                new_mini_batch, infer_mode, select_adapter = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
                # print("generate_new_mini_batch cost:", time.time() - batch_start_time)
                if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                    self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                    self.req_queue.reset_abort_list()
                if new_mini_batch is not None:
                    print("new_mini_batch", new_mini_batch)
                    self.stats_tool.count_prompt_tokens(new_mini_batch)

                    if not self.input_params.no_lora:
                        ret = []
                        # load_start_time = time.time()
                        for tp_rank in range(self.world_size):
                            ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                        await asyncio.gather(*ret)
                        # print("mini batch load cost:", time.time() - load_start_time)
                    pre_start_time = time.time()
                    await self._prefill_batch(new_mini_batch, minibatch=True)
                    if not new_mini_batch.is_clear():
                        await self._merge_batch(self.running_batch, new_mini_batch)
                        self.running_batch.merge(new_mini_batch)
                    self.has_wait_tokens = 0
                    print("prefill time:", time.time() - pre_start_time)
                else:
                    dec_start_time = time.time()
                    self.stats_tool.count_output_tokens(self.running_batch)
                    await self._decode_batch(self.running_batch)
                    await self._filter_runing_batch()
                    print("decode time:", time.time() - dec_start_time)
            print("one step time:", time.time() - step_start_time)

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs, batch.num_problems) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch, minibatch=True):
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        return

    async def _decode_batch(self, batch: Batch):
        self.req_queue.update_counter(batch)
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in
                range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in
                range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        if has_new_finished_req:
            batch.filter_finished()

            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                offload_adapters_time = time.time()
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)
                print("offload_adapters_time:", time.time() - offload_adapters_time)

            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
        return

    async def _filter_runing_batch(self):
        print(len(self.running_batch.reqs))
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                ret = []
                offload_adapters_time = time.time()
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)
                print("offload_adapters_time:", time.time() - offload_adapters_time)
            self.running_batch = None
            return

    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return

    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))

        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, tuple) and len(recv_req) == 5:
                adapter_dir, prompt_ids, sampling_params, multimodal_params, request_id = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)

            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(obj=abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               )

    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats=not args.disable_log_stats,
            log_stats_interval=args.log_stats_interval,
        )

        asyncio.run(router.wait_to_model_ready())
        if input_params.profile:
            asyncio.run(router.profile_prefill())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            cache_dir = os.path.expanduser("~/.cache/valora")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir + "/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir + "/profile_results.pkl")

    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
