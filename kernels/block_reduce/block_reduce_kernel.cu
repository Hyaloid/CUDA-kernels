#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

__device__ __host__ inline
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

template <typename T>
__global__
void inclusivePrefixSumKernel(const T* input, T* output, int n) {
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int block_len = min(n - offset, blockDim.x);
    
    if (tid >= block_len) return;

    extern __shared__ int block_sum[];
    T* shared_block_sum = reinterpret_cast<T*>(block_sum);
    
    output[tid + offset] = input[tid + offset];
    __syncthreads();
    
    // up-sweep
    for (int stride = 1; stride < block_len; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < block_len) {
            output[index + offset] += output[index - stride + offset];
        }
        __syncthreads();
    }
    
    // down-sweep
    for (int stride = block_len / 2; stride > 0; stride >>= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < block_len) {
            output[index + stride + offset] += output[index + offset];
        }
        __syncthreads();
    }
    // block sync
    block.sync();
    if (tid < gridDim.x) {
        shared_block_sum[tid] = output[(tid + 1) * blockDim.x - 1];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < blockIdx.x; ++i) {
        output[tid + offset] += shared_block_sum[i];
    }
}

torch::Tensor inclusive_prefix_sum(
    const torch::Tensor& input,
    torch::Tensor& output
) {
    /*
        This algorithm requires the input_len to be a power of two.
        input: [1, 2, 3, 4, 5, 6]
        output: [1, 3, 6, 10, 15, 21]
    */
    const int input_len = input.size(0);

    TORCH_CHECK(isPowerOfTwo(input_len), "input_len must be a power of two, but got ", input_len);

    constexpr int block_size = 128;
    const int grid_size = (input_len + block_size - 1) / block_size;

    const dim3 grid(grid_size);
    const dim3 block(block_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int shared_mem_size = grid_size * sizeof(int);
    
    AT_DISPATCH_ALL_TYPES(
        input.scalar_type(),
        "inclusivePrefixSum",
        [&] {
            inclusivePrefixSumKernel<scalar_t><<<grid, block, shared_mem_size, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                input_len
            );
        }
    );
    
    return output;
}
