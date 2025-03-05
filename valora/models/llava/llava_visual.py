import torch
import torch.nn.functional as F
import json
import os
from PIL import Image
from typing import List, Union
from safetensors import safe_open


class LlavaVisionModel:
    def __init__(self):
        pass

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in config:
            self.load_hf_model(config, weight_dir)
        else:
            self.load_bin_model(config, weight_dir)

        self.vision_tower.requires_grad_(False)
        self.device = torch.device("cpu")

        assert "model.mm_projector.0.weight" in self.projector_weights
        assert "model.mm_projector.0.bias" in self.projector_weights
        assert "model.mm_projector.2.weight" in self.projector_weights
        assert "model.mm_projector.2.bias" in self.projector_weights

    def load_hf_model(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration
        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy
        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor
        model = LlavaForConditionalGeneration.from_pretrained(
            weight_dir,
            torch_dtype=torch.float16,
        )
        self.vision_tower = model.vision_tower
        model.multi_modal_projector = None
        model.language_model = None

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), 'pt', 'cpu')
                for k in d.keys():
                    if "multi_modal_projector.linear_1" in k:
                        self.projector_weights[k.replace("multi_modal_projector.linear_1", "model.mm_projector.0")] = d.get_tensor(k).half()
                    if "multi_modal_projector.linear_2" in k:
                        self.projector_weights[k.replace("multi_modal_projector.linear_2", "model.mm_projector.2")] = d.get_tensor(k).half()

    def load_bin_model(self, config, weight_dir):
        self.select_layer = config.get("mm_vision_select_layer", -2)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half()

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".bin"):
                d = torch.load(os.path.join(weight_dir, f), "cpu")
                for k, v in d.items():
                    if "model.mm_projector" in k:
                        self.projector_weights[k] = v.half()

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        for k, v in self.projector_weights.items():
            self.projector_weights[k] = v.cuda()
        self.device = torch.device("cuda")
        return self

    # batch images infer
    def forward(self, x):
        x = x.half().to(device=self.device)

        x = self.vision_tower(x, output_hidden_states=True)
        x = x.hidden_states[self.select_layer]
        if self.select_feature == "patch" or self.select_feature == "default":
            x = x[:, 1:].contiguous()
        B, L, N = x.shape
        x = x.view(-1, N).half()

        # mm_project
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.0.weight"],
            bias=self.projector_weights["model.mm_projector.0.bias"],
        )
        x = F.gelu(x)
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.2.weight"],
            bias=self.projector_weights["model.mm_projector.2.bias"],
        )
        x = x.view(B, L, -1)
        return x

    def encode(self, image_items: List[Union[str, Image.Image]]):
        images = []
        for item in image_items:
            if isinstance(item, Image.Image):
                image = item
            elif item.startswith("http://") or item.startswith("https://"):
                import requests
                image = Image.open(requests.get(item, stream=True).raw)
            else:
                image = Image.open(item)
            images.append(image.convert("RGB"))

        images = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        return self.forward(images)