import sys
from utils import setup_kernel

setup_kernel("element_wise", "element_wise_kernel.cu", "element_wise.cpp")
