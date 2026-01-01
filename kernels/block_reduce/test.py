from block_reduce import (
    inclusive_prefix_sum
)

import torch

def create_data():
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # a = torch.tensor(a, dtype=torch.int32, device="cuda")
    a = tensor = torch.arange(1024, dtype=torch.int32, device="cuda")
    b = torch.zeros_like(a)
    c = a.cumsum(-1)
    print(f"c: \n{c}")
    return a, b

def run_kernels():
    input_data, output_data = create_data()
    inclusive_prefix_sum(input_data, output_data)
    print(output_data)

def main():
    run_kernels()

if __name__ == "__main__":
    main()