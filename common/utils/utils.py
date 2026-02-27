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
