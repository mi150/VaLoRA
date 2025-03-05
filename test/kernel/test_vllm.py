"""
This script is mainly used to test whether trtion kernels can run normally
under different conditions, including various batches, numbers of LoRA , and
maximum ranks.
"""
import random
from unittest.mock import patch

import pytest
import torch

from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink
from vllm.triton_utils.libentry import LibEntry

from vllm_utils import (generate_data, generate_data_for_expand_nslices,
                        ref_torch_groupgemm)

HIDDEN_SIZES = [4096]

BATCHES = [128,256,512,1024,4096]
NUM_LORA = [1]
DTYPES = [torch.float16, torch.bfloat16]
MAX_RANKS = [16]
SCALES = [0.5]
SEED = [0]
CUDA_DEVICES = [f"cuda:{7}"]


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_punica_sgmv(
        batches: int,
        num_loras: int,
        rank: int,
        hidden_size: int,
        scaling: float,
        dtype: torch.dtype,
        op_type: str,
        seed: int,
        device: str,
):
    random.seed(seed)
    torch.set_default_device(device)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    seq_length = 1
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        op_type,
        device,
    )
    # print(
    #     inputs_tensor.shape,
    #     lora_weights.shape,
    #     our_out_tensor.shape,
    #     ref_out_tensor.shape,
    #     b_seq_start_loc.shape,
    #     lora_indices_tensor.shape,
    #     seq_len_tensor.shape,
    #     indices.shape,
    # )
    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    ref_torch_groupgemm(
        ref_out_tensor,
        inputs_tensor,
        lora_weights,
        lora_indices_tensor,
        seq_len_tensor,
        batches,
        scaling if op_type == "shrink" else 1.0,
        op_type,
    )
    t1 = time.time()

    if op_type == "shrink":
        sgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            scaling,
        )
    else:
        sgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            add_inputs=True,
        )


    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    # assert_close(our_out_tensor, ref_out_tensor)
    # torch.cuda.synchronize()
    t2 = time.time()
    print(t2-t1)
    return t2 - t1


import time
t_ls = []
for k in MAX_RANKS:
    for i in BATCHES:
        print("BATCHES", i, "MAX_RANKS", k)
        t1 = test_punica_sgmv(i, 1, k, 4096, 1, torch.bfloat16, "shrink", 0, "cuda")

        t1 = 0
        for _ in range(10):
            t1 += test_punica_sgmv(i, 1, k, 4096, 1, torch.bfloat16, "shrink", 0, "cuda")
        torch.cuda.synchronize()
        t_ls.append(t1*100)
print(t_ls)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_punica_bgmv(
        batches: int,
        num_loras: int,
        rank: int,
        hidden_size: int,
        scaling: float,
        dtype: torch.dtype,
        op_type: str,
        seed: int,
        device: str,
):
    from vllm.lora.ops.bgmv_expand import _bgmv_expand_kernel
    from vllm.lora.ops.bgmv_shrink import _bgmv_shrink_kernel

    random.seed(seed)
    torch.set_default_device(device)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    seq_length = 1
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        op_type,
        device,
    )
    if op_type == "shrink":
        # The current _bgmv_shrink_kernel does not require the libentry
        # decoration. The purpose of adding this patch is to test the
        # correctness of libentry.
        with patch(
                "vllm.lora.ops.bgmv_shrink._bgmv_shrink_kernel",
                LibEntry(_bgmv_shrink_kernel),
        ):
            bgmv_shrink(
                inputs_tensor,
                lora_weights,
                our_out_tensor,
                indices,
                scaling,
            )
    else:
        # ditto
        with patch(
                "vllm.lora.ops.bgmv_expand._bgmv_expand_kernel",
                LibEntry(_bgmv_expand_kernel),
        ):
            bgmv_expand(
                inputs_tensor,
                lora_weights,
                our_out_tensor,
                indices,
                add_inputs=True,
            )
    ref_torch_groupgemm(
        ref_out_tensor,
        inputs_tensor,
        lora_weights,
        lora_indices_tensor,
        seq_len_tensor,
        batches,
        scaling if op_type == "shrink" else 1.0,
        op_type,
    )
    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["sgmv"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_punica_expand_nslices(
        batches: int,
        num_loras: int,
        rank: int,
        hidden_size: int,
        nslices: int,
        dtype: torch.dtype,
        op_type: str,
        seed: int,
        device: str,
):
    from vllm.lora.ops.bgmv_expand_slice import _bgmv_expand_slice_kernel

    random.seed(seed)
    torch.set_default_device(device)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    seq_length = 128 if op_type == "sgmv" else 1
    (
        inputs_tensor,
        lora_weights_lst,
        our_outputs,
        ref_outputs,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )
    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    slice_offset = 0
    for index in range(nslices):
        lora_weights = lora_weights_lst[index]
        if op_type == "sgmv":
            sgmv_expand_slice(
                inputs_tensor,
                lora_weights,
                our_outputs,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                slice_offset,
                hidden_size,
                add_inputs=True,
            )
        else:
            # The current _bgmv_expand_slice_kernel does not require the
            # libentry decoration. The purpose of adding this patch is to test
            # the correctness of libentry.
            with patch(
                    "vllm.lora.ops.bgmv_expand_slice._bgmv_expand_slice_kernel",
                    LibEntry(_bgmv_expand_slice_kernel),
            ):
                bgmv_expand_slice(
                    inputs_tensor,
                    lora_weights,
                    our_outputs,
                    indices,
                    slice_offset,
                    slice_size=hidden_size,
                    add_inputs=True,
                )
        ref_torch_groupgemm(
            ref_outputs[:, slice_offset:slice_offset + hidden_size],
            inputs_tensor,
            lora_weights,
            lora_indices_tensor,
            seq_len_tensor,
            batches,
            1.0,
            op_type="expand",
        )

        slice_offset += hidden_size
    assert_close(our_outputs, ref_outputs)
