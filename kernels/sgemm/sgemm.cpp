#include <torch/extension.h>
#include "utils.h"

torch::Tensor naive_sgemm(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_EXTENSION(naive_sgemm);
}
