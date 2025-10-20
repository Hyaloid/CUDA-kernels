#include "template_utils.h"

#define GET_STRIDES(tensor, prefix) \
    const int prefix##_stride_b = tensor.stride(0); \
    const int prefix##_stride_h = tensor.stride(1); \
    const int prefix##_stride_n = tensor.stride(2); \
    const int prefix##_stride_d = tensor.stride(3);

template <typename T>
__device__ __forceinline__ 
T type_zero() {
    return T(0);
}

template <>
__device__ __forceinline__ 
__half type_zero() {
    return __float2half(0.0f);
}

template <>
__device__ __forceinline__ 
__nv_bfloat16 type_zero() {
    return __float2bfloat16(0.0f);
}

template <typename T>
__global__
void naiveReLuKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int i_stride_b,
    const int i_stride_h,
    const int i_stride_n,
    const int i_stride_d,
    const int o_stride_b,
    const int o_stride_h,
    const int o_stride_n,
    const int o_stride_d,
    const int seqlen,
    const int head_size
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    using cuda_t = typename CudaEquivalent<T>::type;

    const cuda_t* input_ptr = reinterpret_cast<const cuda_t*>(input) + batch_idx * i_stride_b +
                              head_idx * i_stride_h;
    cuda_t* output_ptr = reinterpret_cast<cuda_t*>(output) + batch_idx * o_stride_b +
                         head_idx * o_stride_h;

    cuda_t base_num = type_zero<cuda_t>();

    for (int i = tid; i < seqlen; i += blockDim.x) {
        int input_offset = i * i_stride_n;
        int output_offset = i * o_stride_n;
        for (int head_idx = 0; head_idx < head_size; ++head_idx) {
            output_ptr[output_offset + head_idx * o_stride_d] = type_max(base_num, input_ptr[input_offset + head_idx * i_stride_d]);
        }
    }
}

torch::Tensor naive_relu(
    const torch::Tensor& input
) {
    torch::Tensor output = torch::zeros_like(input);

    const auto [batch_size, num_heads, seqlen, head_size] = std::make_tuple(input.size(0),
                                                                            input.size(1),
                                                                            input.size(2),
                                                                            input.size(3));

    GET_STRIDES(input, i)
    GET_STRIDES(output, o)

    constexpr int block_size = 128;
    const dim3 grid(batch_size, num_heads);
    const dim3 block(block_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "Naive ReLu.",
        [&] {
            naiveReLuKernel<<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                i_stride_b,
                i_stride_h,
                i_stride_n,
                i_stride_d,
                o_stride_b,
                o_stride_h,
                o_stride_n,
                o_stride_d,
                seqlen,
                head_size
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

