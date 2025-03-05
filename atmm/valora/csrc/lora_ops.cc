#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <unordered_map>
#include <tuple>
#include <cstdint>

#include "bgmv/bgmv_config.h"
namespace {

//====== utils ======

inline void check_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr __uint128_t pack_u16(uint16_t a, uint16_t b) {
  return (__uint128_t(a) << 16) | __uint128_t(b);
}

inline constexpr uint64_t pack_u16_3(uint16_t a, uint16_t b,uint32_t c) {
  return (uint64_t(a) << 48) | (uint64_t(b) << 32) | (uint64_t(c));
}

inline constexpr __uint128_t pack_6_param(uint32_t a, uint32_t b, uint32_t c, uint32_t x , uint32_t y, uint32_t z){
        return (static_cast<__uint128_t>(a) << 110) |
         (static_cast<__uint128_t>(b) << 100) |
         (static_cast<__uint128_t>(c) << 90) |
         (static_cast<__uint128_t>(x) << 80) |
         (static_cast<__uint128_t>(y) << 70) |
         (static_cast<__uint128_t>(z) << 60);
}

inline constexpr __uint128_t hash_vl (uint16_t key0 , uint16_t key1 ,uint32_t a, uint32_t b, uint32_t c, uint32_t x , uint32_t y, uint32_t z) {
    // Simple hash combining function
    return  static_cast<__uint128_t>(pack_u16(key0,key1)) + pack_6_param(a,b,c,x,y,z);
}




#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

//====== bgmv ======
template <typename T>
inline bool launch_bgmv_kernel(T* Y, const T* X, const T* W,
                               const int64_t* start_indicies,
                               const int64_t* lora_ranks,
                               const int64_t* loc_indicies,
                               const int64_t* indicies,
                               uint64_t qkvo,
                               uint16_t in_features, uint16_t out_features,
                               int64_t batch_size,
                               const T* lora_scales,
                               const int64_t* output_counts,
                               const int64_t* rank_counts,
                               const int64_t* lora_ids,
                               const int64_t* start_ids,
                               const int8_t* tmp_d,
                              int64_t num_problems,
                              int32_t thd_x,
                              int32_t thd_y,
                              int32_t thd_z,
                              int32_t wrap_x,
                              int32_t wrap_y,
                              int32_t wrap_z
                               ) {
  switch( hash_vl(in_features, out_features,thd_x,thd_y,thd_z,wrap_x,wrap_y,wrap_z)) {
#define CASE_ONESIDE(_T, feat_in, feat_out, thd_x_value , thd_y_value, thd_z_value,wrap_x_value,wrap_y_value, wrap_z_value)                           \
  case hash_vl(feat_in, feat_out, thd_x_value , thd_y_value, thd_z_value,wrap_x_value,wrap_y_value, wrap_z_value):                                   \
    bgmv_kernel<feat_in, feat_out, thd_x_value , thd_y_value, thd_z_value,wrap_x_value,wrap_y_value, wrap_z_value>(Y, X, W, start_indicies, lora_ranks, loc_indicies, indicies, \
                                   qkvo, batch_size, lora_scales, output_counts, rank_counts,lora_ids,start_ids, tmp_d,num_problems);     \
    break;
#define CASE(_T, narrow, wide,tx,ty,tz,wx,wy,wz)  \
    CASE_ONESIDE(T, wide, narrow,tx,ty,tz,wx,wy,wz) \
    CASE_ONESIDE(T, narrow, wide,tx,ty,tz,wx,wy,wz) //\
//  CASE_ONESIDE(T, wide, narrow,tx,ty,tz,wx,wy,wz)

    FOR_BGMV_WIDE_NARROW(CASE, _)
#undef CASE
#undef CASE_ONESIDE
    default:
      return false;
  }

  return true;
}

//TODO add int* output_counts;
  // int* rank_counts;
  // int* lora_ids;
  // int* start_ids;
  //  tmp_d
void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w, torch::Tensor start_indicies,
                   torch::Tensor lora_ranks, torch::Tensor loc_indicies, torch::Tensor indicies,
                   int64_t qkvo, torch::Tensor lora_scales, torch::Tensor output_counts, torch::Tensor rank_counts,torch::Tensor lora_ids ,torch::Tensor start_ids, torch::Tensor tmp_d, int64_t num_problems, int32_t thrd_x,
                              int32_t thrd_y,
                              int32_t thrd_z,
                              int32_t wrap_x,
                              int32_t wrap_y,
                              int32_t wrap_z) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(start_indicies);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(loc_indicies);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(3, w); // [size, head_dim, head_size]
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t h_in = x.size(1);
  int64_t h_out = y.size(1);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  bool ok = false;
  if (h_in < 65536 && h_out < 65536) {
    switch (x.scalar_type()) {
      case at::ScalarType::Half:
        ok = launch_bgmv_kernel(static_cast<nv_half*>(y.data_ptr()),
                                static_cast<nv_half*>(x.data_ptr()),
                                static_cast<nv_half*>(w.data_ptr()),
                                start_indicies.data_ptr<int64_t>(),
                                lora_ranks.data_ptr<int64_t>(),
                                loc_indicies.data_ptr<int64_t>(),
                                indicies.data_ptr<int64_t>(), qkvo, h_in, h_out, B,
                                static_cast<nv_half*>(lora_scales.data_ptr()),
                                output_counts.data_ptr<int64_t>(),
                                rank_counts.data_ptr<int64_t>(),
                                lora_ids.data_ptr<int64_t>(),
                                start_ids.data_ptr<int64_t>(),
                                tmp_d.data_ptr<int8_t>(),
                                num_problems,
                                thrd_x, thrd_y, thrd_z, wrap_x, wrap_y, wrap_z
                                );
        break;
      case at::ScalarType::BFloat16:
        ok = launch_bgmv_kernel(static_cast<nv_bfloat16*>(y.data_ptr()),
                                static_cast<nv_bfloat16*>(x.data_ptr()),
                                static_cast<nv_bfloat16*>(w.data_ptr()),
                                start_indicies.data_ptr<int64_t>(),
                                lora_ranks.data_ptr<int64_t>(),
                                loc_indicies.data_ptr<int64_t>(),
                                indicies.data_ptr<int64_t>(), qkvo, h_in, h_out, B,
                                static_cast<nv_bfloat16*>(lora_scales.data_ptr()),
                                output_counts.data_ptr<int64_t>(),
                                rank_counts.data_ptr<int64_t>(),
                                lora_ids.data_ptr<int64_t>(),
                                start_ids.data_ptr<int64_t>(),
                                tmp_d.data_ptr<int8_t>(),
                                num_problems,
                                thrd_x, thrd_y, thrd_z, wrap_x, wrap_y, wrap_z
                                );
        break;
      default:
        break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type());
}

}  // namespace

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dispatch_bgmv", &dispatch_bgmv, "dispatch_bgmv");
}
