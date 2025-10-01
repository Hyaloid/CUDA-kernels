CXX_FLAGS = []
NVCC_FLAGS = []
LINK_FLAGS = []
CUDA_INCLUDES = []
CUDA_LIBS = []

CXX_FLAGS += ["-O3", "-std=c++17"]
NVCC_FLAGS += ["-O3", "-std=c++17"]
CUDA_LIBS += ["-L/usr/local/cuda/lib64"]
CUDA_INCLUDES += ["/usr/local/cuda/include"]
LINK_FLAGS += ["-lcudart"]

CXX_FLAGS += ["-Wno-sign-compare", "-Wno-unused_variable", "-Wno-unused-local-typedefs"]
NVCC_FLAGS += ["-allow-unsupported-compiler", "--forward-unknown-to-host-compiler"]

