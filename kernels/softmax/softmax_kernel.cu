#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__
T warpReduceSum(T val) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset, WARP_SIZE);
    }
    return val;
}

template <typename T>
__device__ __forceinline__
T warpReduceMax(T val) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset, WARP_SIZE));
    }
    return val;
}

__global__
void naiveSoftmaxKernel(
    const float* input,
    float* output,
    const int stride_b,
    const int stride_n,
    const int stride_h,
    const int stride_d,
    const int out_stride_b,
    const int out_stride_n,
    const int out_stride_h,
    const int out_stride_d,
    const int seqlen,
    const int head_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int tid = threadIdx.x;
    const float* input_ptr = input + batch_idx * stride_b +
                             head_idx * stride_h;
    float* output_ptr = output + batch_idx * out_stride_b +
                        head_idx * out_stride_h;

    for (int i = tid; i < seqlen; i += blockDim.x) {
        int idx = i * stride_n;
        int out_idx = i * out_stride_n;

        float sum = 0.0f;
        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            sum += __expf(input_ptr[idx + dim_idx * stride_d]);
        }

        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            output_ptr[out_idx + dim_idx * out_stride_d] = __expf(input_ptr[idx + dim_idx * stride_d]) / sum;
        }
    }
}

__global__
void safeSoftmaxKernel(
    const float* input,
    float* output,
    const int stride_b,
    const int stride_n,
    const int stride_h,
    const int stride_d,
    const int out_stride_b,
    const int out_stride_n,
    const int out_stride_h,
    const int out_stride_d,
    const int seqlen,
    const int head_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int tid = threadIdx.x;
    const float* input_ptr = input + batch_idx * stride_b +
                             head_idx * stride_h;
    float* output_ptr = output + batch_idx * out_stride_b +
                        head_idx * out_stride_h;

    for (int i = tid; i < seqlen; i += blockDim.x) {
        int idx = i * stride_n;
        int out_idx = i * out_stride_n;

        float row_sum = 0.0f;
        float row_max = -FLT_MAX;
        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            float input_val = input_ptr[idx + dim_idx * stride_d];
            row_max = fmaxf(row_max, input_val);
        }
        
        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            float input_val = input_ptr[idx + dim_idx * stride_d];
            row_sum += __expf(input_val - row_max);
        }

        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            output_ptr[out_idx + dim_idx * out_stride_d] = __expf(input_ptr[idx + dim_idx * stride_d] - row_max) / row_sum;
        }
    }
}

__global__
void safeSoftmaxOptimizedKernel(
    const float* input,
    float* output,
    const int stride_b,
    const int stride_n,
    const int stride_h,
    const int stride_d,
    const int out_stride_b,
    const int out_stride_n,
    const int out_stride_h,
    const int out_stride_d,
    const int seqlen,
    const int head_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int tid = threadIdx.x;
    int num_warps = blockDim.x >> 5;
    int warpid = tid >> 5;
    int laneid = tid & 0x1F;

    const float* input_ptr = input + batch_idx * stride_b +
                             head_idx * stride_h;
    float* output_ptr = output + batch_idx * out_stride_b +
                        head_idx * out_stride_h;

    for (int i = warpid; i < seqlen; i += num_warps) {
        int idx = i * stride_n;
        int out_idx = i * out_stride_n;

        float row_sum = 0.0f;
        float row_max = -FLT_MAX;
        for (int dim_idx = laneid; dim_idx < head_size; dim_idx += WARP_SIZE) {
            float input_val = input_ptr[idx + dim_idx * stride_d];
            row_max = max(row_max, input_val);
        }

        row_max = warpReduceMax(row_max);

        for (int dim_idx = laneid; dim_idx < head_size; dim_idx += WARP_SIZE) {
            float input_val = input_ptr[idx + dim_idx * stride_d];
            row_sum += __expf(input_val - row_max);
        }

        row_sum = warpReduceSum(row_sum);

        for (int dim_idx = laneid; dim_idx < head_size; dim_idx += WARP_SIZE) {
            output_ptr[out_idx + dim_idx * out_stride_d] = __expf(input_ptr[idx + dim_idx * stride_d] - row_max) / row_sum;
        }
    }
}

__global__
void onlineSoftmaxKernel(
    const float* input,
    float* output,
    const int stride_b,
    const int stride_n,
    const int stride_h,
    const int stride_d,
    const int out_stride_b,
    const int out_stride_n,
    const int out_stride_h,
    const int out_stride_d,
    const int seqlen,
    const int head_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int tid = threadIdx.x;
    const float* input_ptr = input + batch_idx * stride_b +
                             head_idx * stride_h;
    float* output_ptr = output + batch_idx * out_stride_b +
                        head_idx * out_stride_h;

    for (int i = tid; i < seqlen; i += blockDim.x) {
        int idx = i * stride_n;
        int out_idx = i * out_stride_n;

        // sum
        float row_sum = 0.0f;
        float row_max = -FLT_MAX;
        float last_row_max = -FLT_MAX;
        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            float input_val = input_ptr[idx + dim_idx * stride_d];
            row_max = fmaxf(row_max, input_val);
            row_sum = row_sum * __expf(last_row_max - row_max) + __expf(input_val - row_max);
            last_row_max = row_max;
        }

        // exp
        for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
            output_ptr[out_idx + dim_idx * out_stride_d] = __expf(input_ptr[idx + dim_idx * stride_d] - row_max) / row_sum;
        }
    }
}

__global__
void onlineSoftmaxOptimizedKernel(
    const float* input,
    float* output,
    const int stride_b,
    const int stride_n,
    const int stride_h,
    const int stride_d,
    const int out_stride_b,
    const int out_stride_n,
    const int out_stride_h,
    const int out_stride_d,
    const int seqlen,
    const int head_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int tid = threadIdx.x;
    int num_warps = blockDim.x >> 5;
    int warpid = tid >> 5;
    int laneid = tid & 0x1F;
    const float* input_ptr = input + batch_idx * stride_b +
                             head_idx * stride_h;
    float* output_ptr = output + batch_idx * out_stride_b +
                        head_idx * out_stride_h;

    for (int i = warpid; i < seqlen; i += num_warps) {
        int idx = i * stride_n;
        int out_idx = i * out_stride_n;

        float row_sum = 0.0f;
        float row_max = -FLT_MAX;
        float last_row_max = -FLT_MAX;
        for (int dim_idx = laneid; dim_idx < head_size; dim_idx += WARP_SIZE) {
            float input_val = input_ptr[idx + dim_idx * stride_d];
            row_max = fmaxf(row_max, input_val);
            row_max = warpReduceMax(row_max);
            row_sum = row_sum * __expf(last_row_max - row_max) + __expf(input_val - row_max);
            row_sum = warpReduceSum(row_sum);
            last_row_max = row_max;
        }

        for (int dim_idx = laneid; dim_idx < head_size; dim_idx += WARP_SIZE) {
            output_ptr[out_idx + dim_idx * out_stride_d] = __expf(input_ptr[idx + dim_idx * stride_d] - row_max) / row_sum;
        }
    }
} 

template <typename KernelFunc>
torch::Tensor launch_softmax_kernel(
    const torch::Tensor& input, // [batch_size, seqlen, num_heads, head_size]
    KernelFunc kernel_func
) {
    torch::Tensor output = torch::zeros_like(input);

    const int batch_size = input.size(0);
    const int seqlen = input.size(1);
    const int num_heads = input.size(2);
    const int head_size = input.size(3);

    const int stride_b = input.stride(0);
    const int stride_n = input.stride(1);
    const int stride_h = input.stride(2);
    const int stride_d = input.stride(3);
    const int out_stride_b = output.stride(0);
    const int out_stride_n = output.stride(1);
    const int out_stride_h = output.stride(2);
    const int out_stride_d = output.stride(3);

    constexpr int block_size = 128;
    const dim3 grid(batch_size, num_heads);
    const dim3 block(block_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    kernel_func<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        stride_b, stride_n, stride_h, stride_d,
        out_stride_b, out_stride_n, out_stride_h, out_stride_d,
        seqlen, head_size
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor naive_softmax(const torch::Tensor& input) {
    return launch_softmax_kernel(input, naiveSoftmaxKernel);
}

torch::Tensor safe_softmax(const torch::Tensor& input) {
    return launch_softmax_kernel(input, safeSoftmaxKernel);
}

torch::Tensor safe_softmax_optimized(const torch::Tensor& input) {
    return launch_softmax_kernel(input, safeSoftmaxOptimizedKernel);
}

torch::Tensor online_softmax(const torch::Tensor& input) {
    return launch_softmax_kernel(input, onlineSoftmaxKernel);
}

torch::Tensor online_softmax_optimized(const torch::Tensor& input) {
    return launch_softmax_kernel(input, onlineSoftmaxOptimizedKernel);
}
