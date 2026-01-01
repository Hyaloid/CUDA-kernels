import torch

class CUDATimer:
    """A simple CUDA timer using torch.cuda.Event for measuring elapsed time."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        """Start the timer."""
        self.start_event.record()

    def stop(self):
        """Stop the timer."""
        self.end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded

    def elapsed_time(self):
        """Get the elapsed time in milliseconds."""
        return self.start_event.elapsed_time(self.end_event)