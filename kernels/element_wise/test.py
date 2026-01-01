import torch
from element_wise import element_wise_add
from utils import CUDATimer, KernelBenchmark

def create_data():
    batch_size = 2
    seqlen = 12
    num_heads = 2
    head_dim = 8
    input_data = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=torch.float32).cuda()
    return input_data, input_data

def element_wise_add_torch(x, axis=-1):
    output = x + x
    return output

def main():
    kernels = {
        "naive": (element_wise_add_torch, element_wise_add),
    }

    kernel_bench = KernelBenchmark(
        create_data,
        CUDATimer(),
    )

    kernel_bench.warmup(
        kernels,
        iters=50,
    )

    kernel_bench.run(
        kernels,
        check=True,
    )

if __name__ == "__main__":
    main()
