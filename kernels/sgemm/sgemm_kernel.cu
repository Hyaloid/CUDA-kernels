#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define GET_STRIDES(tensor, prefix) \
    const int prefix##_stride_n = tensor.stride(0); \
    const int prefix##_stride_d = tensor.stride(1);

__host__ __device__ inline
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__global__
void naiveSgemmKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ D,
    const int a_stride_n,
    const int a_stride_d,
    const int b_stride_n,
    const int b_stride_d,
    const int d_stride_n,
    const int d_stride_d,
    const int seqlen_a,
    const int seqlen_b,
    const int head_size
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seqlen_a && col < seqlen_b) {
        float val = 0.0f;
    #pragma unroll
        for (int k = 0; k < head_size; ++k) {
            // transpose B
            val += A[row * a_stride_n + k * a_stride_d] *
                   B[col * b_stride_n + k * b_stride_d];
        }
        D[row * d_stride_n + col * d_stride_d] = val;
    }
}

template <typename KernelFunc>
torch::Tensor launch_sgemm_kernel(
    const torch::Tensor& A,   // [seqlen, head_size]
    const torch::Tensor& B,   // [seqlen, head_size]
    KernelFunc kernel_func
) {
    const int seqlen_a = A.size(0);
    const int head_size = A.size(1);

    const int seqlen_b = B.size(0);

    TORCH_CHECK(head_size == B.size(1));

    auto options = torch::TensorOptions()
                   .dtype(A.dtype())
                   .device(A.device());
    torch::Tensor output = torch::zeros({seqlen_a, seqlen_b}, options);

    constexpr int tile_row = 32;
    constexpr int tile_col = 32;
    const dim3 grid(ceil_div(seqlen_a, tile_row), ceil_div(seqlen_b, tile_col));
    const dim3 block(tile_row, tile_col);

    GET_STRIDES(A, a);
    GET_STRIDES(B, b);
    GET_STRIDES(output, d);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    kernel_func<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        a_stride_n, a_stride_d,
        b_stride_n, b_stride_d,
        d_stride_n, d_stride_d,
        seqlen_a, seqlen_b, head_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor naive_sgemm(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    return launch_sgemm_kernel(A, B, naiveSgemmKernel);
}
