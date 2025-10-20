#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define GET_STRIDES(tensor, prefix) \
    const int prefix##_stride_b = tensor.stride(0); \
    const int prefix##_stride_h = tensor.stride(1); \
    const int prefix##_stride_n = tensor.stride(2); \
    const int prefix##_stride_d = tensor.stride(3);

constexpr int WARP_SIZE = 32;

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

template <typename T>
__global__
void elementWiseAddKernel(
    const T* __restrict__ input_a,    // [batch_size, num_heads, seqlen, head_size]
    const T* __restrict__ input_b,    // [batch_size, num_heads, seqlen, head_size]
    T* __restrict__ output,
    const int seqlen,
    const int head_size,
    const int a_stride_b,
    const int a_stride_h,
    const int a_stride_n,
    const int a_stride_d,
    const int b_stride_b,
    const int b_stride_h,
    const int b_stride_n,
    const int b_stride_d,
    const int out_stride_b,
    const int out_stride_h,
    const int out_stride_n,
    const int out_stride_d
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int tid = threadIdx.x;
    const int num_warps = blockDim.x >> 5;
    const int warpid = tid >> 5;
    const int laneid = tid & 0x1F;

    using cuda_t = typename CudaEquivalent<T>::type;

    const cuda_t* input_a_ptr = reinterpret_cast<const cuda_t*>(input_a) + batch_idx * a_stride_b +
                                head_idx * a_stride_h;
    const cuda_t* input_b_ptr = reinterpret_cast<const cuda_t*>(input_b) + batch_idx * b_stride_b +
                                head_idx * b_stride_h;
    cuda_t* output_ptr = reinterpret_cast<cuda_t*>(output) + batch_idx * out_stride_b +
                         head_idx * out_stride_h;

    for (int i = warpid; i < seqlen; i += num_warps) {
        int input_a_offset = i * a_stride_n;
        int input_b_offset = i * b_stride_n;
        int output_offset = i * out_stride_n;

        for (int dim_idx = laneid; dim_idx < head_size; dim_idx += WARP_SIZE) {
            output_ptr[output_offset + dim_idx * out_stride_d] = type_add(input_a_ptr[input_a_offset + dim_idx * a_stride_d],
                                                                          input_b_ptr[input_b_offset + dim_idx * b_stride_d]);
        }
    }
}

torch::Tensor element_wise_add(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b
) {
    TORCH_CHECK(input_a.sizes() == input_b.sizes(), 
                "Input shapes are not the same, please check.");
    TORCH_CHECK(input_a.dtype() == input_b.dtype(), 
                "input_a and input_b must have the same dtype, but got ",
                input_a.dtype(), " and ", input_b.dtype())
    torch::Tensor output = torch::zeros_like(input_a);

    const auto [batch_size, num_heads, seqlen, head_size] = std::make_tuple(input_a.size(0),
                                                                            input_a.size(1),
                                                                            input_a.size(2),
                                                                            input_a.size(3));

    GET_STRIDES(input_a, a)
    GET_STRIDES(input_b, b)
    GET_STRIDES(output, out)

    constexpr int block_size = 128;
    const dim3 grid(batch_size, num_heads);
    const dim3 block(block_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input_a.scalar_type(),
        "Element Wise Add.",
        [&] {
            elementWiseAddKernel<<<grid, block, 0, stream>>>(
                input_a.data_ptr<scalar_t>(),
                input_b.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                seqlen,
                head_size,
                a_stride_b,
                a_stride_h,
                a_stride_n,
                a_stride_d,
                b_stride_b,
                b_stride_h,
                b_stride_n,
                b_stride_d,
                out_stride_b,
                out_stride_h,
                out_stride_n,
                out_stride_d
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
