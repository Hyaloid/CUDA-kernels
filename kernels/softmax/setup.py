import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import setup_kernel

setup_kernel("softmax_kernel", "softmax_kernel.cu", "softmax.cpp")
