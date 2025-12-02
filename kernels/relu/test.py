import sys
import torch

from relu_kernel import naive_relu
import torch.nn.functional as F

def create_data():
    batch_size = 2
    seqlen = 12
    num_heads = 2
    head_dim = 8
    input_data = (torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=torch.float32) - 3).cuda()
    return input_data

def relu_torch(x, axis=-1):
    output = F.relu(x)
    return output

def run_kernels(input_data, base_func_name, compared_func_name):
    # run torch kernel
    output_torch = base_func_name(input_data)

    # run cuda kernel
    output_cuda = compared_func_name(input_data)

    return output_torch, output_cuda

def compare(output_torch, output_cuda):
    if torch.allclose(output_torch, output_cuda, atol=1e-5, rtol=1e-5):
        print("Correct!")
    else:
        print("MissMatch!")

def main():
    input_data = create_data()

    kernels = {
        "naive": (relu_torch, naive_relu),
    }

    for name, (torch_kernel, cuda_kernel) in kernels.items():
        print(f"Comparing {name} kernels ...")
        output_torch, output_cuda = run_kernels(input_data, torch_kernel, cuda_kernel)
        compare(output_torch, output_cuda)

if __name__ == "__main__":
    main()
