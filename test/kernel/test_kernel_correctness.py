import os
import time
import torch
import triton
import triton.language as tl
# from sgmm import sgmm_cutlass
from typing import Dict, Optional, List
from valora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from valora._kernels import dispatch_bgmv
from atmm_ops import dispatch_bgmv as dispatch_sgmm

@triton.jit
def triton_batch_lora_B(
    output,
    x,
    w,
    a_start,
    a_len,
    a_loc,
    batch_req_bins,
    a_scaling,
    qkvo_offset: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_LORA_RANK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    return


def batch_lora_forward_B(
    output,
    x,
    w,
    a_start,
    a_len,
    a_loc,
    batch_req_bins,
    qkvo_offset,
    a_scaling,
):
    #print("B", output.shape, x.shape, w.shape, a_start.shape, a_len.shape, a_loc.shape,
    #      batch_req_bins.shape, qkvo_offset, a_scaling.shape)
    NUM_TOKENS, MAX_LORA_RANK = x.shape
    NUM_TOKENS, HIDDEN = output.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(NUM_TOKENS, BLOCK_SIZE_M), triton.cdiv(HIDDEN, BLOCK_SIZE_N))
    triton_batch_lora_B[grid](output, x,
                              w,
                              a_start, a_len, 
                              a_loc, batch_req_bins, a_scaling, qkvo_offset,
                              NUM_TOKENS, HIDDEN, MAX_LORA_RANK,
                              BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)


def test_bgmv(is_sgmm,H,R,N,adapter=1):
    # H = 4096 # hidden size
    # R = 64 # rank size
    # N = 4096 # batch size
    num_adapters = 1 # num adapter
    num_head = 32 # attention head
    if is_sgmm:
        part = "C"  # a symbol
    else:
        part = "A" # a symbol
    # for sgmm
    num_problem = 1
    num_layers = [1]

    if part == "A":
        x = torch.randn((N, H), dtype=torch.float16, device="cuda")
        delta_qA = torch.zeros((len(x), R), dtype=torch.float16, device="cuda")

        forward_func = dispatch_bgmv
    else:
        x = torch.randn((N, R), dtype=torch.float16, device="cuda")
        delta_qA = torch.zeros((len(x), H), dtype=torch.float16, device="cuda")
        forward_func = dispatch_bgmv
    num_layers = 1
    # for sgmm
    dtype = torch.float16
    device = torch.device("cuda")
    x_list = [x for _ in range(num_layers)]
    x_ptr_l = [t.data_ptr() for t in x_list]
    x_ptr = torch.tensor(x_ptr_l, dtype=torch.int64, device=device)
    y_list = [delta_qA for _ in range(num_layers)]
    y_ptr_l = [t.data_ptr() for t in y_list]
    y_ptr = torch.tensor(y_ptr_l, dtype=torch.int64, device=device)
    s_list = [R for _ in range(num_layers)]
    s = torch.tensor(s_list, dtype=torch.int32, device=device)

    key_buffer = torch.randn((R * 4 * num_adapters, num_head, H // num_head), dtype=torch.float16, device="cuda")
    # resized_key_buffer = key_buffer.clone().reshape(R * 4 * num_adapters, num_head, H // num_head)
    if is_sgmm:
        key_buffer = torch.randn((R, H), dtype=torch.float16, device="cuda")

    # for sgmm
    # key_buffer_copy = key_buffer.clone()
    # print(key_buffer_copy.shape)
    if part == "B":
        b_len = torch.tensor([N]* num_adapters, dtype=torch.long, device="cuda")
        b_start = torch.zeros_like(b_len)
        b_start[1:] = torch.cumsum(b_len[:-1], dim=0)
        b_loc = torch.arange(N * num_adapters, dtype=torch.long, device="cuda")

        a_len = torch.tensor([R * 4] * num_adapters, dtype=torch.long, device="cuda")
        a_start = torch.zeros_like(a_len)
        a_start[1:] = torch.cumsum(a_len[:-1], dim=0)
        a_loc = torch.arange(R * 4 * num_adapters, dtype=torch.long, device="cuda")
        a_scaling = torch.tensor([1] * num_adapters, dtype=torch.float16, device="cuda")
        batch_req_bins = torch.concat([
            torch.tensor([i if i<adapter else 0] * ((N + num_adapters - 1) // num_adapters), dtype=torch.long, device="cuda")
            for i in range(num_adapters)])
        batch_req_bins = batch_req_bins[:len(x)]
    else:
        b_len = torch.tensor([N] * num_adapters, dtype=torch.long, device="cuda")
        b_start = torch.zeros_like(b_len)
        b_start[1:] = torch.cumsum(b_len[:-1], dim=0)
        b_loc = torch.arange(N * num_adapters, dtype=torch.long, device="cuda")

        a_len = torch.tensor([H * 4] * num_adapters, dtype=torch.long, device="cuda")
        a_start = torch.zeros_like(a_len)
        a_start[1:] = torch.cumsum(a_len[:-1], dim=0)
        a_loc = torch.arange(H * 4 * num_adapters, dtype=torch.long, device="cuda")
        a_scaling = torch.tensor([1] * num_adapters, dtype=torch.float16, device="cuda")
        batch_req_bins = torch.concat([
            torch.tensor([i] * ((N + num_adapters - 1) // num_adapters), dtype=torch.long, device="cuda")
            for i in range(num_adapters)])
        batch_req_bins = batch_req_bins[:len(x)]
    # print(batch_req_bins.shape)
    N0 = 32 * 2048
    output_counts = torch.zeros(N0, dtype=torch.long, device="cuda")
    rank_counts = torch.zeros(N0, dtype=torch.long, device="cuda")
    lora_ids = torch.zeros(N0, dtype=torch.long, device="cuda")
    start_ids = torch.zeros(N0, dtype=torch.long, device="cuda")
    tmp_d = torch.zeros(N0 * 60, dtype=torch.int8, device="cuda")
    output_counts[0] = N
    rank_counts[0] = R
    lora_ids[0] =  0
    start_ids[0] = 0
    qkvo = 0
    results = []
    for i in range(N):
        a_id = batch_req_bins[i]
        a_w = key_buffer[a_start[a_id] + qkvo * R: a_start[a_id] + (qkvo + 1) * R]
        if is_sgmm:
            # print("is_sgmm")
            a_w = key_buffer
        # if part == "A":
        #     a_w = a_w.reshape(R, H).T
        # elif part == "B":
        #     # print(a_w.shape)
        #     a_w = a_w.reshape(H, R).T
        # else:
        #     a_w = a_w.reshape(R, H)
        # results.append(x[i:i+1, :] @ a_w)
    # ref = delta_qA + torch.concat(results)

    # resized_key_buffer = torch.empty_like(key_buffer[a_start[a_id] + qkvo * R: a_start[a_id] + (qkvo + 1) * R])
    # resized_key_buffer.copy_(key_buffer[a_start[a_id] + qkvo * R: a_start[a_id] + (qkvo + 1) * R])
    # for sgmm
    w_list = [key_buffer for _ in range(num_layers)]
    w_ptr_l = [t.data_ptr() for t in w_list]
    w_ptr = torch.tensor(w_ptr_l, dtype=torch.int64, device=device)
    type_slice = torch.randn((2,2), dtype=dtype, device=device)


    # lora_get_qkvo_fwd_expand(x, key_buffer.view(-1, H),
    #                          delta_qA, a_scaling,
    #                          a_loc, a_start,
    #                          a_len, b_loc,
    #                          b_len, batch_req_bins, H,
    #                          0, R, N)
    # forward_func(delta_qA, x,
    #              key_buffer,
    #              a_start, a_len,
    #              a_loc, batch_req_bins, qkvo, a_scaling)
    # batch_lora_forward_B(delta_qA, x,
    #                      key_buffer,
    #                      a_start, a_len,
    #                      a_loc, batch_req_bins, 0, a_scaling)
    # print(y_list[0])
    # (out, in, lora_w, rank, num_layers, d_in_size, d_out_size, lora_alpha, type_slice)
    # print(y_ptr, x_ptr, w_ptr, s, num_problem, R, H, a_scaling, type_slice)
    # sgmm_cutlass(y_ptr, x_ptr, w_ptr, s, num_layers[0], N, H, 1, type_slice)

    # print(y_list[0])
    # print("max delta:", torch.max(torch.abs(delta_qA - ref)))

    def to_test_SGMM():
        torch.cuda.synchronize()
        # dispatch_sgmm(delta_qA, x,
        #               key_buffer,
        #               a_start, a_len,
        #               a_loc, batch_req_bins, 1, a_scaling,output_counts, rank_counts, lora_ids, start_ids,
        #                           tmp_d, 1)
        # batch_lora_forward_B(delta_qA, x,
        #                     key_buffer,
        #                     a_start, a_len,
        #                     a_loc, batch_req_bins, 0, a_scaling)
        sgmm_cutlass(y_ptr, x_ptr, w_ptr, s, num_layers, N, H, 1, type_slice)
        torch.cuda.synchronize()
        #ref = x @ key_buffer[:R].reshape(-1, H).T
    def to_test_SLORA():
        # torch.cuda.synchronize()
        lora_get_qkvo_fwd_shrink(delta_qA, x,
                                 key_buffer.view(-1, R),
                                 a_loc, a_start,
                                 a_len, b_loc,
                                 b_len, batch_req_bins, H,
                                 0, R, N)
        # dispatch_bgmv(delta_qA, x,
        #               key_buffer,
        #               a_start, a_len,
        #               a_loc, batch_req_bins, 0, a_scaling)
        torch.cuda.synchronize()
    def to_test_SLORA_triton():
        torch.cuda.synchronize()
        if part == "A":
            lora_get_qkvo_fwd_shrink(x, key_buffer.view(-1, H if part=="B" else R),
                                     delta_qA,
                                     a_loc, a_start,
                                     a_len, b_loc,
                                     b_len, batch_req_bins, H if part=="B" else R,
                                     0, R if part=="B" else H, N)
        else:
            lora_get_qkvo_fwd_expand(x, key_buffer.view(-1, H if part=="B" else R),
                                     delta_qA, a_scaling,
                                     a_loc, a_start,
                                     a_len, b_loc,
                                     b_len, batch_req_bins, H if part=="B" else R,
                                     0, R if part=="B" else H, N)
        torch.cuda.synchronize()
    # Warm up
    for _ in range(10):
        if is_sgmm:
            to_test_SGMM()
        else:
            to_test_SLORA()
    run_iter = 50
    torch.cuda.synchronize()
    if is_sgmm:
        t1 = time.time()
        for _ in range(run_iter):
            to_test_SGMM()
            # torch.cuda.synchronize()
        t2 = time.time()
        print(f"SGMM time cost {((t2 - t1) / run_iter) * 1000} ms")
    else:
        t1 = time.time()
        for _ in range(run_iter):
        # for _ in range(num_layers):
            to_test_SLORA()
        torch.cuda.synchronize()
        t2 = time.time()
        print(f"SLORA time cost {((t2 - t1) / run_iter) * 1000} ms")
        # t1 = time.time()
    return ((t2 - t1) / run_iter) * 1000
        # for _ in range(run_iter):
        #     to_test_SLORA_triton()
        #     torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"SLORA triton time cost {((t2 - t1) / run_iter) * 1000} ms")


def test_dlora_mm(k, r, b, i):
    d = 4096  # output dimension

    x = torch.randn(b, k)  # shape: (b, k)
    adapter_mapping = torch.randn(b, i)  # shape: (b, i)
    lora_weight_active_lora_As = torch.randn(i, k, r)  # shape: (i, k, r)
    lora_weight_active_lora_Bs = torch.randn(i, r, d)  # shape: (i, r, d)

    # 计算过程
    for _ in range(10):
        intermediate_results = []
        for j in range(i):
            temp_result = torch.mm(x, lora_weight_active_lora_As[j])  # shape: (b, r)
            temp_result = temp_result * adapter_mapping[:, j:j+1]  # shape: (b, r)
            intermediate_results.append(temp_result)
        intermediate_result = sum(intermediate_results)  # shape: (b, r)

        final_results = []
        for j in range(i):
            temp_result = torch.mm(intermediate_result, lora_weight_active_lora_Bs[j])  # shape: (b, d)
            final_results.append(temp_result)
        result = sum(final_results)  # shape: (b, d)
        torch.cuda.synchronize()

    run_iter = 1000
    t1 = time.time()
    for _ in range(run_iter):
        intermediate_results = []
        for j in range(i):
            temp_result = torch.mm(x, lora_weight_active_lora_As[j])  # shape: (b, r)
            temp_result = temp_result * adapter_mapping[:, j:j+1]  # shape: (b, r)
            intermediate_results.append(temp_result)
        intermediate_result = sum(intermediate_results)  # shape: (b, r)

        final_results = []
        for j in range(i):
            temp_result = torch.mm(intermediate_result, lora_weight_active_lora_Bs[j])  # shape: (b, d)
            final_results.append(temp_result)
        result = sum(final_results)  # shape: (b, d)
        torch.cuda.synchronize()

    t2 = time.time()
    print(f"torch.mm DLoRA time cost {((t2 - t1) / run_iter) * 1000} ms")

def test_dlora(k, r, b):

    i = 1  # intermediate dimension (adapter mapping and lora weights)
    d = 4096  # output dimension

    x = torch.randn((b, k), dtype=torch.float16, device="cuda")  # shape: (b, k)
    adapter_mapping = torch.randn((b, i), dtype=torch.float16, device="cuda") # shape: (b, i)
    lora_weight_active_lora_As = torch.randn((i, k, r), dtype=torch.float16, device="cuda")  # shape: (i, k, r)
    lora_weight_active_lora_Bs = torch.randn((i, r, d), dtype=torch.float16, device="cuda")  # shape: (i, r, d)
    for _ in range(10):
        # 计算结果
        result = torch.einsum('bk, bi, ikr, ird->bd',
                              x,
                              adapter_mapping,
                              lora_weight_active_lora_As,
                              lora_weight_active_lora_Bs)
        torch.cuda.synchronize()
    run_iter = 50
    t1 = time.time()
    for _ in range(run_iter):
        torch.cuda.synchronize()
        result = torch.einsum('bk, bi, ikr, ird->bd',
                              x,
                              adapter_mapping,
                              lora_weight_active_lora_As,
                              lora_weight_active_lora_Bs)
        torch.cuda.synchronize()
    t2 = time.time()
    print(f"DLoRA time cost {((t2 - t1) / run_iter) * 1000} ms")
    return (t2 - t1) / (run_iter)


if __name__ == "__main__":
    torch.manual_seed(42)
    H = int(4096)
    R, N = [64], [1,2,4,8,16,32,256,512,1024,2048,4096,8192]
    t_list=[]
    t_list_d=[]
    for i in range(len(R)):
        for j in range(len(N)):
            print(H, R[i], N[j])
            t_list.append(test_bgmv(False, H, R[i], N[j]))
            test_bgmv(False, H, R[i], N[j])
            test_dlora(H, R[i], N[j])
            t_list_d.append(test_dlora(4096, R[i], N[j])*1000)
    print(t_list)
    print(t_list_d)
    # token_num = 512
    # req_num = 10
    # R, N = [64], [token_num*req*2 for req in range(1, int(req_num/2)+1)]
    # # R, N = [64], [10*512]
    # for i in range(len(R)):
    #     for j in range(len(N)):
    #         for k in range(1):
    #             print(H, R[i], N[j], k)
    #             # test_bgmv(True, H, R[i], N[j], k)
    #             test_bgmv(False, H, R[i], N[j],100)


