import sys
from utils import setup_kernel

setup_kernel("block_reduce", "block_reduce_kernel.cu", "block_reduce.cpp")
