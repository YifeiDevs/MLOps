import os
from huggingface_hub import HfApi
from datetime import datetime

def upload_to_huggingface(file_path, repo_name, token, path_in_repo="auto", repo_type="model", add_timestamp_prefix=True):
    """
    Upload a file to a Hugging Face repo with optional timestamp prefix.
    - file_path: Local file path
    - repo_name: Hugging Face repo name
    - token: API token
    - path_in_repo: Repo path, "auto" adds timestamp
    - repo_type: Repo type ("model", "dataset", "space")
    """
    if path_in_repo == "auto":
        # file_name = file_path.split("/")[-1]
        file_name = os.path.basename(file_path)
        if add_timestamp_prefix:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"{timestamp}_{file_name}"
        path_in_repo = file_name

    api = HfApi()  # API instance

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_name,
        repo_type=repo_type,
        token=token
    )
    
    print(f"Uploaded '{file_path}' to '{repo_name}' at '{path_in_repo}'.")

if __name__ == "__main__":
    # %pip install git+https://github.com/YifeiDevs/MLOps.git
    # from mlops.huggingface_utils import upload_to_huggingface
    repo_name = "your-repo-name"
    HF_TOKEN = "your-huggingface-token" # https://huggingface.co/settings/tokens
    upload_to_huggingface("test.txt", repo_name, HF_TOKEN, repo_type="model" or "dataset")

# %%
import re
from huggingface_hub import HfApi, hf_hub_download
from datetime import datetime

def get_latest_file(repo_name, token, repo_type="model"):
    """
    Get the most recent timestamped file in a Hugging Face repo.
    - repo_name: Hugging Face repo name
    - token: API token
    - repo_type: Repo type
    Returns: Latest file name or None.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_name, repo_type=repo_type, token=token)
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"

    files_with_timestamps = {
        file_name: datetime.strptime(re.search(timestamp_pattern, file_name).group(1), "%Y-%m-%d_%H-%M-%S")
        for file_name in files if re.search(timestamp_pattern, file_name)
    }

    return max(files_with_timestamps, key=files_with_timestamps.get) if files_with_timestamps else None

def download_latest_file(repo_name, token=None, repo_type="model"):
    """
    Download the most recent timestamped file from a Hugging Face repo.
    - repo_name: Hugging Face repo name
    - token: API token
    - repo_type: Repo type
    Returns: Downloaded file path or None.
    """
    latest_file = get_latest_file(repo_name, token, repo_type)

    if not latest_file:
        print("No timestamped files found.")
        return None

    downloaded_file_path = hf_hub_download(
        repo_id=repo_name,
        filename=latest_file,
        repo_type=repo_type,
        token=token
    )

    print(f"Downloaded '{latest_file}' to '{downloaded_file_path}'.")
    return downloaded_file_path

if __name__ == "__main__":
    # %pip install git+https://github.com/YifeiDevs/MLOps.git
    # from mlops.huggingface_utils import download_latest_file
    repo_name = "your-repo-name"
    downloaded_file_path = download_latest_file(repo_name, repo_type="model" or "dataset", token=None)
    print(f"{downloaded_file_path = }")
