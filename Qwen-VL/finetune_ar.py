# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from accelerate.utils import DistributedType
import re
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import time

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def get_head_state_maybe_zero_3(named_params, layers_to_train):
    to_return = {k: maybe_zero_3(t) for k, t in named_params if k in layers_to_train}
    #to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, layers_to_train, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
            state_dict_head = get_head_state_maybe_zero_3(trainer.model.named_parameters(),
                                                          layers_to_train)

        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)
        if trainer.args.use_lora:
            print(state_dict_head)
            if not os.path.exists(output_dir+"_head"):
                os.makedirs(output_dir+"_head")
            torch.save(state_dict_head, output_dir+"_head/modify_head.pth")
            #trainer._save(output_dir+"_head/", state_dict=state_dict_head)

def str2class(label):
    category_dict = {0: 'ApplyEyeMakeup', 1: 'ApplyLipstick', 2: 'Archery', 3: 'BabyCrawling', 4: 'BalanceBeam', 5: 'BandMarching', 6: 'BaseballPitch',
                     7: 'Basketball', 8: 'BasketballDunk', 9: 'BenchPress', 10: 'Biking', 11: 'Billiards', 12: 'BlowDryHair', 13: 'BlowingCandles',
                     14: 'BodyWeightSquats', 15: 'Bowling', 16: 'BoxingPunchingBag', 17: 'BoxingSpeedBag', 18: 'BreastStroke', 19: 'BrushingTeeth',
                     20: 'CleanAndJerk', 21: 'CliffDiving', 22: 'CricketBowling', 23: 'CricketShot', 24: 'CuttingInKitchen', 25: 'Diving', 26: 'Drumming',
                     27: 'Fencing', 28: 'FieldHockeyPenalty', 29: 'FloorGymnastics', 30: 'FrisbeeCatch', 31: 'FrontCrawl', 32: 'GolfSwing', 33: 'Haircut',
                     34: 'Hammering', 35: 'HammerThrow', 36: 'HandstandPushups', 37: 'HandstandWalking', 38: 'HeadMassage', 39: 'HighJump', 40: 'HorseRace',
                     41: 'HorseRiding', 42: 'HulaHoop', 43: 'IceDancing', 44: 'JavelinThrow', 45: 'JugglingBalls', 46: 'JumpingJack', 47: 'JumpRope', 48: 'Kayaking',
                     49: 'Knitting', 50: 'LongJump', 51: 'Lunges', 52: 'MilitaryParade', 53: 'Mixing', 54: 'MoppingFloor', 55: 'Nunchucks', 56: 'ParallelBars',
                     57: 'PizzaTossing', 58: 'PlayingCello', 59: 'PlayingDaf', 60: 'PlayingDhol', 61: 'PlayingFlute', 62: 'PlayingGuitar', 63: 'PlayingPiano',
                     64: 'PlayingSitar', 65: 'PlayingTabla', 66: 'PlayingViolin', 67: 'PoleVault', 68: 'PommelHorse', 69: 'PullUps', 70: 'Punch', 71: 'PushUps',
                     72: 'Rafting', 73: 'RockClimbingIndoor', 74: 'RopeClimbing', 75: 'Rowing', 76: 'SalsaSpin', 77: 'ShavingBeard', 78: 'Shotput', 79: 'SkateBoarding',
                     80: 'Skiing', 81: 'Skijet', 82: 'SkyDiving', 83: 'SoccerJuggling', 84: 'SoccerPenalty', 85: 'StillRings', 86: 'SumoWrestling', 87: 'Surfing', 88: 'Swing',
                     89: 'TableTennisShot', 90: 'TaiChi', 91: 'TennisSwing', 92: 'ThrowDiscus', 93: 'TrampolineJumping', 94: 'Typing', 95: 'UnevenBars', 96: 'VolleyballSpiking',
                     97: 'WalkingWithDog', 98: 'WallPushups', 99: 'WritingOnBoard', 100: 'YoYo'}
    return next(key for key, value in category_dict.items() if value == label)

def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = ""
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets, boxes, labels = [], [], [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target, box, label = [], [], [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        #target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        #assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role == '<|im_start|>user':
                pattern = re.compile(r'<img>.*?</img>\n', re.DOTALL)
                matches = pattern.findall(sentence["value"])
                result = ''.join(matches)
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(result).input_ids + [im_end] + nl_tokens
                # _input_id = tokenizer(role).input_ids
                input_id += _input_id
                # print(input_id)

            if role == '<|im_start|>assistant':
                target.append(str2class(sentence["value"]))


            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
        #assert len(input_id) == len(target)
        # for _ in range(20):
        #     input_id += det_tokens
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))

        input_ids.append(input_id[:max_len])
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_ds(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = ""
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant", "distillation": "<|im_start|>distillation"}

    video_path = None
    # image2tensor
    images = None

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets, boxes, labels = [], [], [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target, box, label = [], [], [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        #target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        #assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role == '<|im_start|>user':

                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(result).input_ids + [im_end] + nl_tokens
                # _input_id = tokenizer(role).input_ids
                input_id += _input_id
                # print(input_id)
            if role == '<|im_start|>assistant':
                target.append(str2class(sentence["value"]))


            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                video_path = sentence["value"]
                # print('get',video_path)
        #assert len(input_id) == len(target)
        # for _ in range(20):
        #     input_id += det_tokens
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))

        input_ids.append(input_id[:max_len])
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        images=images,
    )
def preprocess_od(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )

        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def test():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     quantization_config=GPTQConfig(
    #         bits=4, use_exllama=False
    #     )
    #     if training_args.use_lora and lora_args.q_lora
    #     else None,
    # )

    state_dict = torch.load('/data01/tuwenming/Qwen-VL/LoRAs/ucf-ds-6f-cf_head/modify_head.pth')
    model = AutoPeftModelForCausalLM.from_pretrained(
        "/data01/tuwenming/Qwen-VL/LoRAs/ucf-ds-6f-cf",  # path to the output directory
        device_map="cuda",
        trust_remote_code=True
    )

    model.load_state_dict(state_dict, strict=False)
    # print(state_dict)
    layers_to_train = ['base_model.model.norm.bias', 'base_model.model.head.0.weight', 'base_model.model.head.1.weight',
                       'base_model.model.head.0.bias', 'base_model.model.head.1.bias', 'base_model.model.norm.weight']
    # layers_to_train = ['base_model.model.norm.bias', 'base_model.model.head.weight',
    #                    'base_model.model.head.bias', 'base_model.model.norm.weight']

    for name, param in model.named_parameters():
        if name in layers_to_train:
            print(name,param)
    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual, 'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    model.eval()
    from torch.utils.data import DataLoader
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    # print(data_module["train_dataset"])
    batch_size = 1
    # Start tester
    # 创建数据集和 DataLoader
    data_loader = DataLoader(data_module["train_dataset"], batch_size=batch_size, shuffle=False)
    correct = 0
    error = 0
    data_len = 0
    with torch.no_grad():
        for batch in data_loader:

            input_ids = torch.tensor(batch["input_ids"]).to(model.device)
            labels = torch.tensor(batch["labels"]).to(model.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)
            outputs = model.forward(input_ids=input_ids, labels=labels, attention_mask=attention_mask,)
            labels = torch.tensor(labels, device='cuda', dtype=torch.int32).view(-1)
            comparison = outputs == labels
            num_same_elements = comparison.sum().item()
            if num_same_elements==0:
                error += 1
                print(outputs, labels)
                print(error)
            correct += num_same_elements
            data_len += batch_size
        print(correct, data_len)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, use_exllama=False
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual, 'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        print("lora_args.lora_r:", lora_args.lora_r)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    layers_to_train = ['base_model.model.norm.bias', 'base_model.model.head.0.weight', 'base_model.model.head.1.weight', 'base_model.model.head.0.bias', 'base_model.model.head.1.bias', 'base_model.model.norm.weight']
    for name, param in model.named_parameters():
        if name in layers_to_train:
            param.requires_grad = True


    print(lora_args.lora_target_modules)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, layers_to_train=layers_to_train, bias=lora_args.lora_bias)

if __name__ == "__main__":
    # train()
    test()