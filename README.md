# MLOps


```shell
pip install --upgrade git+https://github.com/YifeiDevs/MLOps.git
```

## get_secret_key
```python
from mlops.secret_utils import get_secret_key
api_key = get_secret_key("WANDB_API_KEY")
print(f"{api_key = }")
```


## upload_to_huggingface
```python
from mlops.huggingface_utils import upload_to_huggingface
repo_name = "your-repo-name"
HF_TOKEN = "your-huggingface-token" # https://huggingface.co/settings/tokens
upload_to_huggingface("test.txt", repo_name, HF_TOKEN, repo_type="model" or "dataset")
```


## download_latest_file
```python
from mlops.huggingface_utils import download_latest_file
repo_name = "your-repo-name"
downloaded_file_path = download_latest_file(repo_name, repo_type="model" or "dataset", token=None)
print(f"{downloaded_file_path = }")
```

## device
```python
from mlops.device_utils import get_device_info, in_colab_or_kaggle
device_type, num_processes = get_device_info()
print(f"Device: {device_type}") # "cpu", "cuda" or "xla"
print(f"Processes: {num_processes}")
print("Running in Colab or Kaggle:", in_colab_or_kaggle())
```



