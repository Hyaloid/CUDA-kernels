import torch

from relu_kernel import naive_relu
import torch.nn.functional as F

from utils import CUDATimer, KernelBenchmark

def create_data():
    batch_size = 2
    seqlen = 12
    num_heads = 2
    head_dim = 8
    input_data = (torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=torch.float32) + 3).cuda()
    return input_data

def relu_torch(x, axis=-1):
    output = F.relu(x)
    return output

def main():
    kernels = {"naive": (relu_torch,naive_relu,)}

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
