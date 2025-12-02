import sys
import torch

from sgemm import naive_sgemm
import torch.nn.functional as F

def create_data():
    batch_size = 1
    seqlen = 6878
    num_heads = 2
    head_dim = 512
    input_data = torch.randn(seqlen, head_dim, dtype=torch.float32).cuda()
    input_data_b = torch.randn(7000, head_dim, dtype=torch.float32).cuda()
    return input_data, input_data_b

def torch_naive_gemm(x, y, axis=-1):
    output = x @ y.T
    return output

def run_kernels(input_data, input_data_b, base_func_name, compared_func_name):
    # run torch kernel
    output_torch = base_func_name(input_data, input_data_b)
    print("Torch output:", output_torch)

    # run cuda kernel
    output_cuda = compared_func_name(input_data, input_data_b)
    print("CUDA output:", output_cuda)
    print(output_torch.shape)
    print(output_cuda.shape)
    return output_torch, output_cuda

def compare(output_torch, output_cuda):
    if torch.allclose(output_torch, output_cuda, atol=1e-1, rtol=1e-1):
        print("Correct!")
    else:
        print("MissMatch!")

def main():
    input_data, input_data_b = create_data()

    kernels = {
        "naive": (torch_naive_gemm, naive_sgemm),
    }

    for name, (torch_kernel, cuda_kernel) in kernels.items():
        print(f"Comparing {name} kernels ...")
        output_torch, output_cuda = run_kernels(input_data, input_data_b, torch_kernel, cuda_kernel)
        compare(output_torch, output_cuda)

if __name__ == "__main__":
    main()
