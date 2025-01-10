import os
from pathlib import Path

def get_secret_key(secret_name: str) -> str:
    """Retrieves a secret key from various platforms.

    Attempts to get the key from:
    1. Google Colab (🔑 -> Secret -> Add secret)
    2. Kaggle (Add-ons -> Secrets -> Add secret)
    3. Local environment variables or .env file (In .env `WANDB_API_KEY=your_key_here`)

    Args:
        secret_name: The name of the secret key.

    Returns:
        The secret key value, or None if not found.
    """
    # Attempt Kaggle    
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(secret_name)
    except ImportError:
        pass

    # Attempt Google Colab
    try:
        from google.colab import userdata
        return userdata.get(secret_name)
    except ImportError:
        pass



    # Attempt local environment (and .env)
    if Path('.env').exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    return os.getenv(secret_name)

if __name__ == "__main__":
    # %pip install git+https://github.com/YifeiDevs/MLOps.git
    # from mlops.secret_utils import get_secret_key
    api_key = get_secret_key("WANDB_API_KEY")
    print(f"{api_key = }")
