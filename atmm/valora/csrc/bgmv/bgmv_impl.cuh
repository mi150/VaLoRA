// adapted from https://github.com/punica-ai/punica/blob/master/punica/ops/csrc/bgmv/bgmv_impl.cuh
#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <stdio.h>
#include "vec_dtypes.cuh"

#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"


namespace cg = cooperative_groups;

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<nv_half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};
//TODO add start_ids in call
__global__ void count_consecutive_requests(const int64_t* __restrict__ lora_rank,const int64_t* __restrict__ indicies,const int64_t* c_start_ids,const int64_t* c_lora_ids,const int64_t* c_output_counts, const int64_t* c_rank_counts,int num_requests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {  // Ensure only one thread executes this kernel
        int current_count = 1;
        int output_idx = 0;
        int start_id = 0;

        int64_t*  output_counts =  const_cast<int64_t*>(c_output_counts);
        int64_t*  rank_counts =  const_cast<int64_t*>(c_rank_counts);
        int64_t*  start_ids =  const_cast<int64_t*>(c_start_ids);
        int64_t*  lora_ids =  const_cast<int64_t*>(c_lora_ids);
        for (int i = 1; i < num_requests; i++) {

            if (indicies[i] == indicies[i - 1]) {
                current_count++;
            } else {
                output_counts[output_idx++] = current_count;
                rank_counts[output_idx-1] = lora_rank[indicies[i-1]] / 4;
                lora_ids[output_idx-1] = indicies[i-1];
                start_ids[output_idx -1] = start_id;
                start_id = i;
                current_count = 1;
            }
        }
        // Store the last count

        output_counts[output_idx++] = current_count;
        rank_counts[output_idx-1] = lora_rank[indicies[num_requests-1]]/4;
        lora_ids[output_idx-1] = indicies[num_requests-1];
        start_ids[ output_idx -1] = start_id;

        // Update the output length
    }
}

//  key_buffer[m][a_loc[a_start[n]] : a_loc[a_start[n] + qkvo*a_len[n]]]；也就是size维度上获取rank长度；
//for xA,  shrink kernel ,  m = output_counts , n = feat_in  ,  k = rank_counts/4
template <int feat_in,int feat_out,typename T>
__global__ void precompute_sgmm_args_for_shrink(cutlass::gemm::GemmCoord *all_problems,
                                     T **ptr_y, T **ptr_x, T **ptr_w,
                                     int64_t *ld_y, int64_t *ld_x,
                                     int64_t *ld_w, T *y, T *x, T *w,
                                     const int64_t* __restrict__ start_indicies,const int64_t* __restrict__ lora_ranks,
                                     const int64_t* __restrict__ loc_indicies,const int64_t* __restrict__ indicies,const int64_t* output_counts,const int64_t* lora_rank_len,const int64_t* lora_ids,const int64_t* start_ids,int64_t qkvo) {
  int i = blockIdx.x;

  int m = output_counts[i] , k = feat_in, n = lora_rank_len[i];
  size_t lora_idx = lora_ids[i];
  size_t lora_rank = lora_rank_len[i]; // if j >= lora_rank, we do not need to do the computation

  size_t mem_pos = start_indicies[lora_idx] + lora_rank * qkvo;
  size_t idx = loc_indicies[mem_pos];
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  ptr_w[i] = w+idx*feat_in;
  ptr_x[i] = x+ start_ids[i]*feat_in;
  ptr_y[i] = y+start_ids[i]*feat_out;
  ld_x[i] = k;
  ld_w[i] = feat_in;
  ld_y[i] = feat_out;
  // 这里 shrink 有点问题，w 和 x 都是 row major 的，
}

//for vB expand kernel , m = output_counts, n = feat_in ?? or rank_counts / 4 　, k = feat_out
template <int feat_in,int feat_out,typename T>
__global__ void precompute_sgmm_args_for_expand(cutlass::gemm::GemmCoord *all_problems,
                                     T **ptr_y, T **ptr_x, T **ptr_w,
                                     int64_t *ld_y, int64_t *ld_x,
                                     int64_t *ld_w, T *y, T *x, T *w,
                                     const int64_t* __restrict__ start_indicies,const int64_t* __restrict__ lora_ranks,
                                     const int64_t* __restrict__ loc_indicies,const int64_t* __restrict__ indicies,const int64_t* output_counts,const int64_t* lora_rank_len,const int64_t* lora_ids,const int64_t* start_ids,int64_t qkvo) {
  int i = blockIdx.x;
  int m = output_counts[i] , k = lora_rank_len[i], n = feat_out;
  size_t lora_idx = lora_ids[i];
  size_t lora_rank = lora_rank_len[i];

  size_t mem_pos = start_indicies[lora_idx] + lora_rank * qkvo;
  size_t idx = loc_indicies[mem_pos];
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  // here expand and shrink are different
  ptr_w[i] = w+idx*feat_out;
  ptr_x[i] = x+ start_ids[i]*feat_in;
  ptr_y[i] = y+start_ids[i]*feat_out;
  ld_x[i] = feat_in;
  ld_w[i] = n;
  ld_y[i] = n;
  // printf("i %d,ld_x[i] %d,ld_y[i] %d\n", (i,ld_x[i],ld_y[i]));
}


size_t sgmm_tmp_size(int64_t num_problems) {
  constexpr auto sz = sizeof(void *) * 3 + sizeof(int64_t) * 3 +
                      sizeof(cutlass::gemm::GemmCoord);
  return sz * num_problems;
}

template <typename T>
inline T *alloc_from_buf(void **buf, int n) {
  auto *p = (T *)*buf;
  *buf = (void *)(p + n);
  return p;
}


template <typename cutlass_t, typename LinearCombination,
          typename GemmIdentityThreadblockSwizzle, typename Architecture,
          int thrd_x, int thrd_y, int thrd_z, int warp_x, int warp_y, int warp_z,
          int ins_x, int ins_y, int ins_z>
bool PerformGemmGrouped(cutlass::gemm::GemmCoord * all_problems, int num_problems, float lora_alpha,
                        cutlass_t **ptr_Y, cutlass_t **ptr_X,
                        cutlass_t **ptr_W, int64_t *ld_Y, int64_t *ld_X,
                        int64_t *ld_W, cudaStream_t stream) {
    // shrink
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_t,                                      // Element B
        cutlass::layout::ColumnMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        Architecture,                                   // Architecture
        cutlass::gemm::GemmShape<thrd_x, thrd_y, thrd_z>,          // Thread Block Shape
        cutlass::gemm::GemmShape<std::min(thrd_x, warp_x), std::min(thrd_y, warp_y), thrd_z>,           // Warp Shape
        cutlass::gemm::GemmShape<ins_x, ins_y, ins_z>,             // Instruction Shape
        LinearCombination,  // Epilogue
        GemmIdentityThreadblockSwizzle,              // Swizzling Operator
        1                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(lora_alpha, 1.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                         epilogue_op, ptr_X, ptr_W, ptr_Y,
                                         ptr_Y, ld_X, ld_W, ld_Y, ld_Y);



    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }



    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }


    return true;
}

template <typename cutlass_t, typename LinearCombination,
          typename GemmIdentityThreadblockSwizzle, typename Architecture,
          int thrd_x, int thrd_y, int thrd_z, int warp_x, int warp_y, int warp_z,
          int ins_x, int ins_y, int ins_z>
bool PerformGemmGrouped_expand(cutlass::gemm::GemmCoord * all_problems, int num_problems, float lora_alpha,
                        cutlass_t **ptr_Y, cutlass_t **ptr_X,
                        cutlass_t **ptr_W, int64_t *ld_Y, int64_t *ld_X,
                        int64_t *ld_W, cudaStream_t stream) {
    // Expand
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_t,                                      // Element B
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        Architecture,                                   // Architecture
        cutlass::gemm::GemmShape<thrd_x, thrd_y, thrd_z>,          // Thread Block Shape
        cutlass::gemm::GemmShape<std::min(thrd_x, warp_x), std::min(thrd_y, warp_y), thrd_z>,           // Warp Shape
        cutlass::gemm::GemmShape<ins_x, ins_y, ins_z>,             // Instruction Shape
        LinearCombination,  // Epilogue
        GemmIdentityThreadblockSwizzle,              // Swizzling Operator
        3                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(lora_alpha, 1.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                         epilogue_op, ptr_X, ptr_W, ptr_Y,
                                         ptr_Y, ld_X, ld_W, ld_Y, ld_Y);



    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }



    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }


    return true;
}




template <int feat_in, int feat_out,int thrd_x,int thrd_y,int thrd_z,int warp_x,int warp_y,int warp_z,typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 const T* __restrict__ W,
                 const int64_t* __restrict__ start_indicies,
                 const int64_t* __restrict__ lora_ranks,
                 const int64_t* __restrict__ loc_indicies,
                 const int64_t* __restrict__ indicies,
                 int64_t qkvo,
                 int64_t batch_size,
                 const T* __restrict__ lora_scales,
                 const int64_t* __restrict__ output_counts,const int64_t* __restrict__ rank_counts,const int64_t* __restrict__ lora_ids , const int64_t* __restrict__ start_ids, const int8_t* __restrict__ itmp_d , int64_t num_problems
                 ) {
  // printf("stage 1");

  // int* output_counts;
  // int* rank_counts;
  // int* lora_ids;
  // int* start_ids;
  float cc = float(8.0);

// cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

// printf("stage 2");
// cudaMalloc((void**)&start_ids, batch_size * sizeof(int));
// cudaMalloc((void**)&lora_ids, batch_size * sizeof(int));
// cudaMalloc((void**)&rank_counts, batch_size * sizeof(int));
// cudaMalloc((void**)&output_counts, batch_size * sizeof(int));

// // Initialize allocated memory with zeros
// cudaMemset(start_ids, 0, batch_size * sizeof(int));
// cudaMemset(lora_ids, 0, batch_size * sizeof(int));
// cudaMemset(rank_counts, 0, batch_size * sizeof(int));
// cudaMemset(output_counts, 0, batch_size * sizeof(int));  // Initialize the output counts to zero
// // Initialize the value on the device (optional, but ensures d_num_problems starts at 0)
//   cudaMemcpy(d_num_problems, &num_problems, sizeof(int), cudaMemcpyHostToDevice);
//   // Launch the counting kernel with only 1 thread
//   //for xA,  shrink kernel ,  m = output_counts , n = feat_in  ,  k = rank_counts/4 ,
//   // for vB expand kernel , m = output_counts, n = feat_in ?? or rank_counts / 4 　, k = feat_out
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // count_consecutive_requests<<<1, 1,0,stream>>>(lora_ranks,indicies,start_ids,lora_ids,output_counts,rank_counts,batch_size);

  // cudaDeviceSynchronize();
  // clock_t start1 = clock();
  // cudaMemcpy(&num_problems, d_num_problems, sizeof(int64_t), cudaMemcpyDeviceToHost);

  int tmp_size = sgmm_tmp_size(num_problems);

  int8_t * void_tmp = const_cast<int8_t*>(itmp_d);

  void* tmp_d = static_cast<void*>(void_tmp);

  using cutlass_t = typename cutlass_dtype<T>::type;
  auto ptr_Y = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_W = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ld_Y = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_X = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_W = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto all_problems = alloc_from_buf<cutlass::gemm::GemmCoord>(&tmp_d, num_problems);

  using LinearCombination = cutlass::epilogue::thread::LinearCombination<cutlass_t, 8, float, float>;
  using GemmIdentityThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  bool success = true;
  // cudaDeviceSynchronize();
  // clock_t mid = clock();

  if constexpr (feat_in < feat_out) {

    // for expand
    precompute_sgmm_args_for_expand<feat_in, feat_out><<<num_problems,1,0,stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, ld_Y, ld_X, ld_W, (cutlass_t *)Y,
      (cutlass_t *)X, (cutlass_t *)W,start_indicies,lora_ranks,
      loc_indicies,indicies,output_counts,rank_counts,lora_ids,start_ids,qkvo);
   if (cc == float(8.0)){
      // A100
      success = PerformGemmGrouped_expand<cutlass_t, LinearCombination, GemmIdentityThreadblockSwizzle, cutlass::arch::Sm80,thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z, 16, 8, 16>(
                                    all_problems, num_problems, 1.0, ptr_Y, ptr_X,
                                    ptr_W, ld_Y, ld_X, ld_W, stream);
    }
    else {
      success = PerformGemmGrouped_expand<cutlass_t, LinearCombination, GemmIdentityThreadblockSwizzle, cutlass::arch::Sm80, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z, 16, 8, 16>(
                                    all_problems, num_problems, 1.0, ptr_Y, ptr_X,
                                    ptr_W, ld_Y, ld_X, ld_W, stream);
      fprintf(stderr, "Current compute capability: %f. However, this operator is tuned for compute cability 8.0. \n", cc);
    }

  } else {
    // for shrink
    precompute_sgmm_args_for_shrink<feat_in, feat_out><<<num_problems, 1,0,stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, ld_Y, ld_X, ld_W, (cutlass_t *)Y,
      (cutlass_t *)X, (cutlass_t *)W,start_indicies,lora_ranks,
      loc_indicies,indicies,output_counts,rank_counts,lora_ids,start_ids,qkvo);
    if (cc == float(8.0)) {
      //A100
      success = PerformGemmGrouped<cutlass_t, LinearCombination, GemmIdentityThreadblockSwizzle, cutlass::arch::Sm80,thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z,16, 8, 16>(
                                    all_problems, num_problems, 1.0, ptr_Y, ptr_X,
                                    ptr_W, ld_Y, ld_X, ld_W, stream);

    }
    else {
      success = PerformGemmGrouped<cutlass_t, LinearCombination, GemmIdentityThreadblockSwizzle, cutlass::arch::Sm80, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z, 16, 8, 16>(
                                    all_problems, num_problems, 1.0, ptr_Y, ptr_X,
                                    ptr_W, ld_Y, ld_X, ld_W, stream);
      fprintf(stderr, "Current compute capability: %f. However, this operator is tuned for compute cability 8.0. \n", cc);
    }
  }
  cudaStreamDestroy(stream);
  // cudaDeviceSynchronize();
  // clock_t end = clock();
  // printf("Time for middle part: %ld clock cycles\n", (long)(start2 - start1));
  // printf("Time for first part: %ld clock cycles\n", (long)(start1 - start));
  // printf("Time for second part: %ld clock cycles\n", (long)(end - mid));
}

// thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z,ins_x,ins_y, ins_z

#define INST_BGMV(feat_in, feat_out, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z,T)                                    \
  template void bgmv_kernel<feat_in, feat_out, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z,T>(                            \
      T* __restrict__ Y, const T* __restrict__ X, const T* __restrict__ W, \
      const int64_t* __restrict__ start_indicies,                           \
      const int64_t* __restrict__ lora_ranks,                           \
      const int64_t* __restrict__ loc_indicies,                           \
      const int64_t* __restrict__ indicies, int64_t qkvo,           \
      int64_t batch_size, const T* __restrict__ lora_scales,const int64_t* __restrict__ output_counts,const int64_t* __restrict__ rank_counts,const int64_t* __restrict__ lora_ids , const int64_t* __restrict__ start_ids, const int8_t* __restrict__ itmp_d ,int64_t num_problems);

#define INST_BGMV_TWOSIDE(T, narrow, wide, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z) \
  INST_BGMV(narrow, wide, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z,T)               \
  INST_BGMV(wide, narrow, thrd_x, thrd_y, thrd_z, warp_x, warp_y, warp_z,T)
