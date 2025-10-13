#define TORCH_BINDING_EXTENSION(func) \
    m.def(#func, &func, "Implementation of " #func ".")
