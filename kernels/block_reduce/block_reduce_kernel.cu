#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__global__
void inclusivePrefixSumKernel(const T* input, T* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    output[tid] = input[tid];
    
    __syncthreads();
    
    // up-sweep
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            output[index] += output[index - stride];
        }
        __syncthreads();
    }
    
    // down-sweep
    for (int stride = n / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < n) {
            output[index + stride] += output[index];
        }
        __syncthreads();
    }
}

torch::Tensor inclusive_prefix_sum(
    const torch::Tensor& input,
    torch::Tensor& output
) {
    const int input_len = input.size(0);

    constexpr int block_size = 128;
    const int grid_size = (input_len + block_size - 1) / block_size;

    const dim3 grid(grid_size);
    const dim3 block(block_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_ALL_TYPES(
        input.scalar_type(),
        "inclusivePrefixSum",
        [&] {
            inclusivePrefixSumKernel<scalar_t><<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                input_len
            );
        }
    );
    
    return output;
}
