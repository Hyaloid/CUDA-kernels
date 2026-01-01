import torch
from softmax_kernel import naive_softmax, safe_softmax, safe_softmax_optimized, online_softmax, online_softmax_optimized
from utils import CUDATimer, KernelBenchmark

def create_data():
    batch_size = 2
    seqlen = 12
    num_heads = 2
    head_dim = 8
    input_data = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        dtype=torch.float32,
    ).cuda()
    return input_data

def torch_naive_softmax(x, axis=-1):
    e_x = torch.exp(x)
    return e_x / torch.sum(e_x, dim=axis, keepdim=True)

def main():
    kernels = {
        "naive": (torch_naive_softmax, naive_softmax),
        "safe": (torch_naive_softmax, safe_softmax),
        "safe_opt": (torch_naive_softmax, safe_softmax_optimized),
        "online": (torch_naive_softmax, online_softmax),
        "online_opt": (torch_naive_softmax, online_softmax_optimized)
    }

    kernel_bench = KernelBenchmark(
        create_data,
        CUDATimer(),
    )

    kernel_bench.warmup(
        kernels,
        iters=100,
    )

    kernel_bench.run(
        kernels,
        check=True,
    )

if __name__ == "__main__":
    main()
