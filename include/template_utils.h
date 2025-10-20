#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__device__ __forceinline__
T type_add(T a, T b) {
    return a + b;
}

template <>
__device__ __forceinline__
__half type_add<__half>(__half a, __half b) {
    return __hadd(a, b);
}

template <typename T>
__device__ __forceinline__
T type_max(T a, T b) {
    return a > b ? a : b;
}

template <>
__device__ __forceinline__
__half type_max(__half a, __half b) {
    return __hmax(a, b);
}

template <>
__device__ __forceinline__
__nv_bfloat16 type_max(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmax(a, b);
}

template <typename T>
struct CudaEquivalent {
    using type = T;
};

template <>
struct CudaEquivalent<c10::Half> {
    using type = __half;
};

template <>
struct CudaEquivalent<c10::BFloat16> {
    using type = __nv_bfloat16;
};
