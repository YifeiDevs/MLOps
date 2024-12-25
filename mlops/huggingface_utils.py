
# %%
from huggingface_hub import HfApi
from datetime import datetime

def upload_to_huggingface(file_path, repo_name, token, path_in_repo="auto", repo_type="model"):
    # 如果 path_in_repo 为 "auto"，在文件名前加时间戳
    if path_in_repo == "auto":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 生成时间戳，格式为 YYYYMMDDHHMMSS
        file_name = file_path.split("/")[-1]
        path_in_repo = f"{timestamp}_{file_name}"  # 将时间戳加到文件名前
    # 创建 API 实例
    api = HfApi()
    # 上传文件
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_name,
        repo_type=repo_type,
        token=token
    )
    print(f"File '{file_path}' successfully uploaded to '{repo_name}' at '{path_in_repo}'.")

if __name__ == "__main__":
    repo_name = "your-repo-name"  # 替换为你的仓库名称
    HF_TOKEN = "your-huggingface-token"  # 替换为你的 Hugging Face 访问令牌
    upload_to_huggingface("test.txt", repo_name, HF_TOKEN)


# %%
import re
from huggingface_hub import HfApi, hf_hub_download
from datetime import datetime

def get_latest_file(repo_name, token, repo_type="model"):
    # 创建 API 实例
    api = HfApi()

    # 获取仓库中文件列表
    files = api.list_repo_files(repo_id=repo_name, repo_type=repo_type, token=token)

    # 匹配文件名中带有时间戳的文件
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    files_with_timestamps = {}

    for file_name in files:
        match = re.search(timestamp_pattern, file_name)
        if match:
            timestamp = match.group(1)
            # 转换时间戳为 datetime 对象
            timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
            files_with_timestamps[file_name] = timestamp_dt

    # 如果没有符合条件的文件，返回 None
    if not files_with_timestamps:
        return None

    # 找到最新的文件
    latest_file = max(files_with_timestamps, key=files_with_timestamps.get)
    return latest_file

def download_latest_file(repo_name, token, repo_type="model"):
    # 获取最新文件的名称
    latest_file = get_latest_file(repo_name, token, repo_type)

    if latest_file is None:
        print("No files with timestamps found in the repository.")
        return None

    # 使用 hf_hub_download 下载文件
    downloaded_file_path = hf_hub_download(
        repo_id=repo_name,
        filename=latest_file,
        repo_type=repo_type,
        token=token
    )

    print(f"Latest file '{latest_file}' has been downloaded to '{downloaded_file_path}'.")
    return downloaded_file_path

if __name__ == "__main__":
    repo_name = "your-repo-name"  # 替换为你的仓库名称
    HF_TOKEN = "your-hf-token"  # 替换为你的 Hugging Face Token

    downloaded_file_path = download_latest_file(repo_name, HF_TOKEN)
    print(f"Downloaded file path: {downloaded_file_path = }")
