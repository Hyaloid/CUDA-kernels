import sys
from utils import setup_kernel

setup_kernel("relu_kernel", "relu_kernel.cu", "relu.cpp")