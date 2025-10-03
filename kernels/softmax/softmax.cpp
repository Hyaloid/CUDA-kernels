#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "naive_softmax",
        &naive_softmax,
        "Implementation of naive_softmax."
    );

    m.def(
        "safe_softmax",
        &safe_softmax,
        "Implementation of safe_softmax."
    );

    m.def(
        "safe_softmax_optimized",
        &safe_softmax_optimized,
        "Implementation of safe_softmax_optimized."
    );

    m.def(
        "online_softmax",
        &online_softmax,
        "Implementation of online_softmax."
    );
}