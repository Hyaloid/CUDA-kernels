import torch
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CXX_FLAGS = []
NVCC_FLAGS = []
LINK_FLAGS = []
CUDA_INCLUDES = []
CPP_INCLUDES = []
CUDA_LIBS = []

CXX_FLAGS += ["-O3", "-std=c++17"]
NVCC_FLAGS += ["-O3", "-std=c++17"]
CUDA_LIBS += ["-L/usr/local/cuda/lib64"]
CUDA_INCLUDES += ["/usr/local/cuda/include"]
CPP_INCLUDES += [str((Path(__file__).parent / "../include").resolve())]
LINK_FLAGS += ["-lcudart"]

CXX_FLAGS += ["-Wno-sign-compare", "-Wno-unused_variable", "-Wno-unused-local-typedefs"]
NVCC_FLAGS += ["-allow-unsupported-compiler", "--forward-unknown-to-host-compiler"]

def get_compute_capability():
    if not torch.cuda.is_available():
        return "80"  # set default to Ampere architecture
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"

sm_version = get_compute_capability()
arch_flag = f'-gencode=arch=compute_{sm_version},code=sm_{sm_version}'
if arch_flag not in NVCC_FLAGS:
    NVCC_FLAGS.append(arch_flag)

NVCC_FLAGS += [
    "-O3",
    "--use_fast_math",
    "-Xcompiler",
    "-Wall",
    "--disable-warnings",       # reduce meaningless compilations of logs
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

def setup_kernel(name, *sources):
    setup(
        name=name,
        ext_modules=[
            CUDAExtension(
                name=name,
                sources=list(sources),
                include_dirs=CUDA_INCLUDES + CPP_INCLUDES,
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
