import sys
import torch

from softmax_kernel import naive_softmax, safe_softmax, safe_softmax_optimized, online_softmax, online_softmax_optimized
import torch.nn.functional as F

def create_data():
    batch_size = 2
    seqlen = 12
    num_heads = 2
    head_dim = 8
    input_data = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float32).cuda()
    return input_data

def torch_naive_softmax(x, axis=-1):
    e_x = torch.exp(x)
    return e_x / torch.sum(e_x, dim=axis, keepdim=True)
    
def torch_safe_softmax(x, axis=-1):
    return F.softmax(x, dim=axis)

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
        "naive": (torch_naive_softmax, naive_softmax),
        "safe": (torch_safe_softmax, safe_softmax),
        "safe_opt": (torch_safe_softmax, safe_softmax_optimized),
        "online": (torch_safe_softmax, online_softmax),
        "online_opt": (torch_safe_softmax, online_softmax_optimized)
    }

    for name, (torch_kernel, cuda_kernel) in kernels.items():
        print(f"Comparing {name} kernels ...")
        output_torch, output_cuda = run_kernels(input_data, torch_kernel, cuda_kernel)
        compare(output_torch, output_cuda)

if __name__ == "__main__":
    main()
