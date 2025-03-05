import json
import numpy as np
import unicodedata
from valora.models.qwen7b.model import Qwen7bTpPartModel
from .layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer

from valora.server.multimodal_params import MultimodalParams, ImageItem



# Warp of the origal tokenizer
class QWenVLTokenizer:

    def __init__(self, tokenizer, model_cfg):
        self.tokenizer = tokenizer
        # <img>: 151857
        self.image_start_tag = tokenizer.image_start_tag
        self.image_start_id = tokenizer.img_start_id
        # </img>: 151858
        self.image_end_tag = tokenizer.image_end_tag
        self.image_end_id = tokenizer.img_end_id
        # <imgpad>: 151859
        self.image_length = model_cfg['visual'].get("n_queries", 256)

    def _list_find(self, input_list, target, start_idx):
        cur_list = input_list[start_idx:]
        if target in cur_list:
            return cur_list.index(target) + start_idx
        return -1

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

    # <img>xxx</img> -> Picture {image_idx}:<img>xxx</img>\n
    def _format_prompt(self, prompt):
        parts = prompt.split(self.image_start_tag)
        prompt = parts[0]
        for idx, part in enumerate(parts[1:]):
            prompt += f'Picture {idx + 1}:' + self.image_start_tag + part
        parts = prompt.split(self.image_end_tag)
        prompt = parts[0]
        for part in parts[1:]:
            prompt += self.image_end_tag + '\n' + part
        return prompt

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None):
        prompt = unicodedata.normalize("NFC", prompt)
        prompt = self._format_prompt(prompt)
        origin_ids = self.tokenizer.tokenizer.encode(prompt, allowed_special='all', disallowed_special=())

        input_ids = []
        image_id = 0
        end = 0
        while True:
            # <img>xxx</img> -> <img><imgpad>*256</img>
            start = self._list_find(origin_ids, self.image_start_id, end)
            if start == -1:
                break
            input_ids.extend(origin_ids[end: start])
            end = self._list_find(origin_ids, self.image_end_id, start)
            if end == -1:
                raise ValueError("Unclosed image token")
            print(f"image_id:{image_id}")
            print(f"images:{multimodal_params.images}")
            if len(multimodal_params.images) != 0:
                token_id = multimodal_params.images[image_id].token_id
                token_num = multimodal_params.images[image_id].token_num
                assert token_num == self.image_length, "invalid token num: {} vs {}!".format(token_num, self.image_length)

                input_ids.append(self.image_start_id)
                input_ids.extend(range(token_id, token_id + token_num))
                input_ids.append(self.image_end_id)
                end += 1
                image_id += 1

        input_ids.extend(origin_ids[end: ])
        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            assert image_cnt == image_id, "invalid image tag num: {} vs {}!".format(image_cnt, image_id)
        return input_ids

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode


class QWenVLTpPartModel(Qwen7bTpPartModel):

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num,mem_adapter_size, load_way="HF", mode=[], dummy=False):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size,load_way, mode, dummy=dummy)
        return