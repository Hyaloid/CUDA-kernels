#include <torch/extension.h>
#include "utils.h"

torch::Tensor element_wise_add(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_EXTENSION(element_wise_add);
}
