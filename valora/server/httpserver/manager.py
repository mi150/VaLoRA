import zmq
import zmq.asyncio
import rpyc
import hashlib
import asyncio
import uvloop
from typing import Union
from transformers import AutoTokenizer
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq, BatchAbortReq
from ..embed_cache.utils import get_shm_name_data, get_shm_name_embed, create_shm
# TODO: Add VisualServer to process visaul message
class HttpServerManager:
    def __init__(
        self,
        model_weightdir,
        tokenizor_mode,
        router_port,
        httpserver_port,
        cache_port,
        visual_port,
        total_token_num,
        max_req_input_len,
        max_req_total_len,
        trust_remote_code,
        enable_multimodal,
        dummy=False,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")
        
        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            self.cache_client = rpyc.connect("localhost", cache_port)
            self.send_to_visual = context.socket(zmq.PUSH)
            self.send_to_visual.connect(f"tcp://127.0.0.1:{visual_port}")
        
        
        try: 
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code) 
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode) 

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)
        
        # print(self.tokenizer)
        
        self.total_token_num = total_token_num
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len
    # connect cache server, calculate md5, alloc resource, return uuid
    
    
    async def _alloc_resource(self, data, num):
        md5sum = hashlib.md5(data).hexdigest()
        wait_time = 1
        while True:
            record = self.cache_client.root.alloc(md5sum, num)
            # hit or new
            if record:
                uid = record["id"]
                if not self.cache_client.root.get_item_data(uid):
                    create_shm(get_shm_name_data(uid), data)
                    self.cache_client.root.set_item_data(uid)
                return record
            # cache full
            else:
                await asyncio.sleep(wait_time)
                wait_time = min(wait_time + 2, 9)

    async def _alloc_multimodal_resources(self, multimodal_params):
        for img in multimodal_params.images:
            record = await self._alloc_resource(img.read(), self.tokenizer.get_image_token_length(img))
            img.uuid = record["id"]
            img.token_id = record["token_id"]
            img.token_num = record["token_num"]
    
    async def _release_multimodal_resources(self, multimodal_params):
        if multimodal_params is not None:
            for img in multimodal_params.images:
                if img.uuid is not None:
                    self.cache_client.root.release(img.uuid)
                    img.uuid = None
                    img.token_id = None
                    img.token_num = None

    def tokens(self, prompt):
        prompt_ids = self.tokenizer.encode(prompt)
        return len(prompt_ids)

    # Open multi Modality Model End


    async def generate(self, adapter_dir, prompt, sampling_params, request_id, multimodal_params):
        # print(self.tokenizer)
        # prompt_ids = self.tokenizer.encode(prompt)
        
        if self.enable_multimodal:
            # print(multimodal_params.to_dict())
            # assert len(multimodal_params.images) <= self.args.cache_capacity, "too many images!"
            assert len(multimodal_params.images) <= 1000, "too many images!"

            await self._alloc_multimodal_resources(multimodal_params)
            print(multimodal_params)
            prompt_ids = self.tokenizer.encode(prompt, multimodal_params = multimodal_params)
        else:
            prompt_ids = self.tokenizer.encode(prompt)
            
            
            
        prompt_tokens = len(prompt_ids)
        if prompt_tokens > self.max_req_input_len:
            raise ValueError(
                f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}"
            )
        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        # TODO:
        
        if self.enable_multimodal:
            self.send_to_visual.send_pyobj(
                (adapter_dir, prompt_ids, sampling_params, multimodal_params, request_id)
            )
        else:
            self.send_to_router.send_pyobj((adapter_dir, prompt_ids, sampling_params, request_id))
    
    
        event = asyncio.Event()
        self.req_id_to_out_inf[request_id] = ("", {}, False, event)
        # print("self.req_id_to_out_inf:",self.req_id_to_out_inf)
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=50)
            except asyncio.TimeoutError:
                pass
            event.clear()
            # request_id is aborted by the backend system for traffic control
            if request_id not in self.req_id_to_out_inf:
                yield "", {}, -1
                break
            out_str, metadata, finished, _ = self.req_id_to_out_inf[request_id]
            if len(metadata) != 0:
                self.req_id_to_out_inf[request_id] = ("", {}, finished, event)
                metadata["prompt_tokens"] = prompt_tokens
                yield out_str, metadata, finished
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                    if self.enable_multimodal:
                        await self._release_multimodal_resources(multimodal_params)
                except:
                    pass
                break
        # print("one req finish:", request_id)
        # print("now self.req_id_to_out_inf:",self.req_id_to_out_inf)
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        if self.enable_multimodal:
            self.send_to_visual.send_pyobj(abort_req)
        try:
            req = self.req_id_to_out_inf[request_id]
            if self.enable_multimodal:
                await self._release_multimodal_resources(req.multimodal_params)
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans: Union[BatchStrOut, BatchAbortReq] = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, (BatchStrOut, BatchAbortReq)), f"error recv type {type(recv_ans)}"
            if isinstance(recv_ans, BatchStrOut):
                for req_id, text, metadata, finished, abort in recv_ans.reqs_infs:
                    try:
                        if not abort:
                            _, _, _, event = self.req_id_to_out_inf[req_id]
                            self.req_id_to_out_inf[req_id] = (
                                text,
                                metadata,
                                finished,
                                event,
                            )
                            event.set()
                        else:
                            del self.req_id_to_out_inf[req_id]
                    except:
                        pass
            elif isinstance(recv_ans, BatchAbortReq):
                print("abort reqs:", recv_ans.reqs)
                for req_id in recv_ans.reqs:
                    try:
                        del self.req_id_to_out_inf[req_id]
                    except:
                        pass

        return
