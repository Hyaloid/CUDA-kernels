#include <torch/extension.h>
#include "utils.h"

torch::Tensor naive_relu(
    const torch::Tensor& input
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_EXTENSION(naive_relu);
}
