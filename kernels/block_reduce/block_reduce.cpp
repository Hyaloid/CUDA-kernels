#include <torch/extension.h>
#include "utils.h"

torch::Tensor inclusive_prefix_sum(
    const torch::Tensor& input,
    torch::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_EXTENSION(inclusive_prefix_sum);
}