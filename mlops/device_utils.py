import importlib
import os
import torch

def get_device_info():
    """Detect best device and its count."""
    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()

    # MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps", 1

    # XLA (TPU)
    if importlib.util.find_spec("torch_xla"):
        try:
            import torch_xla.core.xla_model as xm
            d = xm.get_xla_supported_devices()
            if d:
                return "xla", len(d)
        except Exception:
            pass

    # CPU
    return "cpu", (os.cpu_count() or 1)

def get_device_info_accelerate():
    from accelerate import Accelerator
    accelerator = Accelerator()
    return accelerator.device.type, accelerator.num_processes

def in_colab_or_kaggle():
    import os, sys
    if any(key.startswith("KAGGLE") for key in os.environ.keys()):
        return True
    elif "IPython" in sys.modules:
        return "google.colab" in str(sys.modules["IPython"].get_ipython())
    else:
        return False

if __name__ == "__main__":
    # from mlops.device_utils import get_device_info, in_colab_or_kaggle
    device_type, num_processes = get_device_info()
    print(f"Device: {device_type}") # "cpu", "cuda", "mps" or "xla"
    print(f"Processes: {num_processes}")
    print("Running in Colab or Kaggle:", in_colab_or_kaggle())
