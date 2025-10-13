#define STRINGFY(str) #str
#define TORCH_BINDING_EXTENSION(func) m.def(STRINGFY(func), &func, (std::string("Implementation of ") + STRINGFY(func) + ".").c_str())
