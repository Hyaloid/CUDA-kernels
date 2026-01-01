import sys
from utils import setup_kernel

setup_kernel("softmax_kernel", "softmax_kernel.cu", "softmax.cpp")
