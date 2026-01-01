import torch
from sgemm import naive_sgemm

from utils import CUDATimer, KernelBenchmark

def create_data():
    seqlen = 128
    head_dim = 512
    input_data = torch.randn(seqlen, head_dim, dtype=torch.float32).cuda()
    input_data_b = torch.randn(7000, head_dim, dtype=torch.float32).cuda()
    return input_data, input_data_b

def torch_naive_gemm(x, y, axis=-1):
    output = x @ y.T
    return output

def main():
    kernels = {
        "naive": (torch_naive_gemm, naive_sgemm),
    }

    kernel_bench = KernelBenchmark(
        create_data,
        CUDATimer(),
    )
    kernel_bench.create_data()
    kernel_bench.warmup(kernels,)
    kernel_bench.run(kernels, True, 50, atol=1e-1, rtol=1e-1)
    

if __name__ == "__main__":
    main()
