import torch

class KernelBenchmark:
    def __init__(self, data_fn, timer=None, device="cuda"):
        self.data_fn = data_fn
        self.timer = timer
        self.device = device

    def create_data(self, *args, **kwargs):
        result = self.data_fn(*args, **kwargs)
        if isinstance(result, tuple):
            self.input_data = result[0]
            self.extra_data = result[1:]
        else:
            self.input_data = result
            self.extra_data = ()
        self.args_for_kernel = (self.input_data, *self.extra_data)

    def _run_with_timer(self, func, iters=50, *args, **kwargs):
        if self.timer is None:
            y = func(*args, **kwargs)
            return y, None

        self.timer.start()
        for _ in range(iters):
            y = func(*args, **kwargs)
        self.timer.stop()

        return y, self.timer.elapsed_time() / iters

    def _run_kernel_pair(self, name, torch_kernel, cuda_kernel, iters=50, check=False, atol=1e-5, rtol=1e-5):
        print(f"Running kernel: {name}")

        out_torch, t_torch = self._run_with_timer(torch_kernel, iters, *self.args_for_kernel)
        out_cuda, t_cuda = self._run_with_timer(cuda_kernel, iters, *self.args_for_kernel)

        if t_torch is not None:
            print(f"{torch_kernel.__name__} time: {t_torch:.6f} ms")
        if t_cuda is not None:
            print(f"{cuda_kernel.__name__} time: {t_cuda:.6f} ms")

        if check:
            self.check_correctness(out_torch, out_cuda, atol, rtol)

    @staticmethod
    def check_correctness(a, b, atol=1e-5, rtol=1e-5):
        if torch.allclose(a, b, atol=atol, rtol=rtol):
            print("Correct!")
        else:
            max_diff = (a - b).abs().max().item()
            print(f"Mismatch! max diff = {max_diff}")

    def warmup(self, kernels, iters=50):
        print("Warmup ...")
        self.create_data()
        for _ in range(iters):
            for _, (torch_kernel, cuda_kernel) in kernels.items():
                torch_kernel(*self.args_for_kernel)
                cuda_kernel(*self.args_for_kernel)
        torch.cuda.synchronize()

    def run(self, kernels, check=False, iters=50, **kwargs):
        atol = kwargs.get("atol", 1e-5)
        rtol = kwargs.get("rtol", 1e-5)

        self.create_data()

        for name, (torch_kernel, cuda_kernel) in kernels.items():
            self._run_kernel_pair(
                name, torch_kernel, cuda_kernel, iters=iters, check=check, atol=atol, rtol=rtol
            )
