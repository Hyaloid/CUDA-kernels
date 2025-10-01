#include <torch/extension.h>

torch::Tensor inclusive_prefix_sum(
    const torch::Tensor& input,
    torch::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "inclusive_prefix_sum",
        &inclusive_prefix_sum,
        "Implementation of inclusive_prefix_sum."
    );
}