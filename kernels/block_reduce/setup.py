import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import setup_kernel

setup_kernel("block_reduce", "block_reduce_kernel.cu", "block_reduce.cpp")
