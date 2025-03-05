#pragma once

template <int feat_in, int feat_out, int thrd_x,int thrd_y,int thrd_z,int wrap_x,int wrap_y, int wrap_z, typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 const T* __restrict__ W, const int64_t* __restrict__ start_indicies,
                 const int64_t* __restrict__ lora_ranks, const int64_t* __restrict__ loc_indicies,
                 const int64_t* __restrict__ indicies, int64_t qkvo, int64_t batch_size,
                 const T* __restrict__ lora_scales,const int64_t* __restrict__ output_counts,const int64_t* __restrict__ rank_counts,const int64_t* __restrict__ lora_ids , const int64_t* __restrict__ start_ids, const int8_t* __restrict__ itmp_d,int64_t num_problems);

// clang-format off
#define FOR_BGMV_PARAM_wrapz(f,T,narrow,wide,thx,thy,thz,wrapx,wrapy)\
    f(T,narrow,wide,thx,thy,thz,wrapx,wrapy,32) \
    f(T,narrow,wide,thx,thy,thz,wrapx,wrapy,64)

//    f(T,narrow,wide,thx,thy,thz,wrapx,wrapy,4)\
//    f(T,narrow,wide,thx,thy,thz,wrapx,wrapy,4)\
//    f(T,narrow,wide,thx,thy,thz,wrapx,wrapy,8)\



#define FOR_BGMV_PARAM_wrapy(f,T,narrow,wide,thx,thy,thz,wrapx) \
    FOR_BGMV_PARAM_wrapz(f,T,narrow,wide,thx,thy,thz,wrapx,32) \
    FOR_BGMV_PARAM_wrapz(f,T,narrow,wide,thx,thy,thz,wrapx,64) \
    FOR_BGMV_PARAM_wrapz(f,T,narrow,wide,thx,thy,thz,wrapx,128)



#define FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,thz)\
    FOR_BGMV_PARAM_wrapy(f,T,narrow,wide,thx,thy,thz,32) \
    FOR_BGMV_PARAM_wrapy(f,T,narrow,wide,thx,thy,thz,64) \
    FOR_BGMV_PARAM_wrapy(f,T,narrow,wide,thx,thy,thz,128)

//    FOR_BGMV_PARAM_wrapy(f,T,narrow,wide,thx,thy,thz,8)\



//
//#if thy == 32
//    #define FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,thy) \
//        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,32)
//#elif thy == 64
//    #define FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,thy) \
//        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,32) \
//        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,64)
//#else
//    #define FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,thy) \
//        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,32) \
//        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,64)
//#endif

//#define FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,thy)\
//    FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,32)\
//    FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,64)\
//    FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,128)\
//    FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,256)\
//    FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,512)\

//#if narrow == 16
//    #define FOR_BGMV_PARAM_thy(f,T,narrow,wide,thx)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,16)
//#elif narrow == 32
//    #define FOR_BGMV_PARAM_thy(f,T,narrow,wide,thx)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,32)
//#elif narrow == 64
//    #define FOR_BGMV_PARAM_thy(f,T,narrow,wide,thx)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,32)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,64)
//#else
//    #define FOR_BGMV_PARAM_thy(f,T,narrow,wide,thx)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,32)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,64)\
//        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,128)
//#endif
//    FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,256)\
//    FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,512)\


#define FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,thy) \
        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,32) \
        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,64) \
//        FOR_BGMV_PARAM_wrapx(f,T,narrow,wide,thx,thy,128)



#define FOR_BGMV_PARAM_thy(f,T,narrow,wide,thx)\
        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,32)\
        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,64)\
        FOR_BGMV_PARAM_thz(f,T,narrow,wide,thx,128)



#define FOR_BGMV_PARAM_thx(f,T,narrow,wide)\
    FOR_BGMV_PARAM_thy(f,T,narrow,wide,32)\
    FOR_BGMV_PARAM_thy(f,T,narrow,wide,64)\
    FOR_BGMV_PARAM_thy(f,T,narrow,wide,128)\
//    FOR_BGMV_PARAM_thy(f,T,narrow,wide,256)\


//    FOR_BGMV_PARAM_thy(f,T,narrow,wide,256)\
//    FOR_BGMV_PARAM_thy(f,T,narrow,wide,512)\

#define FOR_BGMV_WIDE(f, T, narrow) \
    FOR_BGMV_PARAM_thx(f,T,narrow, 4096) \
    FOR_BGMV_PARAM_thx(f,T,narrow, 6144) \
    FOR_BGMV_PARAM_thx(f,T,narrow, 5120)
//    FOR_BGMV_PARAM_thx(f,T,narrow, 1664) \
//    FOR_BGMV_PARAM_thx(f,T,narrow, 2048)
//    FOR_BGMV_PARAM_thx(f,T,narrow, 2560) \
//    FOR_BGMV_PARAM_thx(f,T,narrow, 3072) \
//    FOR_BGMV_PARAM_thx(f,T,narrow, 3328) \
//    FOR_BGMV_PARAM_thx(f,T,narrow, 4096) \
//    FOR_BGMV_PARAM_thx(f,T,narrow, 5120) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 6656) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 7168) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 8192) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 9216) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 10240) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 11008) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 12288) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 13824) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 16384) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 20480) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 28672) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 36864) \
//    FOR_BGMV_PARAM_thx(f,T, narrow, 49152) \

#define FOR_BGMV_WIDE_NARROW(f, T) \
    FOR_BGMV_WIDE(f, T, 16) \
    FOR_BGMV_WIDE(f, T, 32) \
    FOR_BGMV_WIDE(f, T, 64) \
    FOR_BGMV_WIDE(f, T, 128) \
    FOR_BGMV_WIDE(f, T, 256) \
//    FOR_BGMV_WIDE(f, T, 16) \
//    FOR_BGMV_WIDE(f, T, 32) \





// clang-format on