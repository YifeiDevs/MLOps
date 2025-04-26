def get_device_info():
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
    print(f"Device: {device_type}") # "cpu", "cuda" or "xla"
    print(f"Processes: {num_processes}")
    print("Running in Colab or Kaggle:", is_in_colab_or_kaggle())
