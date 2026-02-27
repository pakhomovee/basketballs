import os
import zipfile
import requests
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_and_extract(url, extract_dir="."):
    """Download a zip archive from a Yandex Disk public link and extract it."""
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    final_url = base_url + "public_key=" + url

    response = requests.get(final_url)
    download_url = response.json()["href"]
    download_response = requests.get(download_url)

    zip_path = os.path.join(extract_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(download_response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)
    print("Data downloaded and extracted.")

def download(url, filename, save_dir="."):
    """
    Download a file from a Yandex Disk public link without extracting.

    Args:
        url: Yandex Disk public key (e.g. https://disk.yandex.ru/d/xxx).
        save_dir: Directory to save the file.
        filename: Name of the saved file.

    Returns:
        Path to the downloaded file.
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    final_url = base_url + "public_key=" + url

    response = requests.get(final_url)
    download_url = response.json()["href"]
    download_response = requests.get(download_url)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        f.write(download_response.content)

    print(f"Downloaded to {save_path}")
    return save_path
