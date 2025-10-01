import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import CXX_FLAGS, NVCC_FLAGS, CUDA_INCLUDES, LINK_FLAGS, CUDA_LIBS

setup(
    name='block_reduce',
    ext_modules=[
        CUDAExtension(
            name='block_reduce',
            sources=['block_reduce_kernel.cu', 'block_reduce.cpp'],
            include_dirs=CUDA_INCLUDES,
            library_dirs=CUDA_LIBS,
            extra_compile_args={
                'cxx': CXX_FLAGS,
                'nvcc': NVCC_FLAGS,
            },
            extra_link_args=LINK_FLAGS,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)