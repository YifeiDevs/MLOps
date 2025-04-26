def get_device_info():
    from accelerate import Accelerator
    accelerator = Accelerator()
    return accelerator.device.type, accelerator.num_processes

if __name__ == "__main__":
    device_type, num_processes = get_device_info()
    print(f"Device: {device_type}") # "cpu", "cuda" or "xla"
    print(f"Processes: {num_processes}")
