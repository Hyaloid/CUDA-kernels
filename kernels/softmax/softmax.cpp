#include <torch/extension.h>
#include "utils.h"

torch::Tensor naive_softmax(
    const torch::Tensor& input
);

torch::Tensor safe_softmax(
    const torch::Tensor& input
);

torch::Tensor safe_softmax_optimized(
    const torch::Tensor& input
);

torch::Tensor online_softmax(
    const torch::Tensor& input
);

torch::Tensor online_softmax_optimized(
    const torch::Tensor& input
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_EXTENSION(naive_softmax);
    TORCH_BINDING_EXTENSION(safe_softmax);
    TORCH_BINDING_EXTENSION(safe_softmax_optimized);
    TORCH_BINDING_EXTENSION(online_softmax);
    TORCH_BINDING_EXTENSION(online_softmax_optimized);
}
