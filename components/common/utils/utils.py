import os
import zipfile
import requests
import torch
from tqdm import tqdm

_YANDEX_DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_yandex_href(url):
    """Resolve a Yandex Disk public key into a direct download URL."""
    response = requests.get(_YANDEX_DOWNLOAD_API + "public_key=" + url)
    response.raise_for_status()
    return response.json()["href"]


def _stream_to_file(download_url, dest_path, desc):
    """Stream *download_url* to *dest_path* in chunks with a tqdm progress bar.

    Streaming (rather than loading the whole response into memory via
    ``.content``) keeps memory flat regardless of archive size — essential for
    multi-GB datasets.
    """
    with requests.get(download_url, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            open(dest_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download_and_extract(url, extract_dir="."):
    """Download a zip archive from a Yandex Disk public link and extract it."""
    download_url = _resolve_yandex_href(url)

    os.makedirs(extract_dir, exist_ok=True)
    zip_path = os.path.join(extract_dir, "data.zip")
    _stream_to_file(download_url, zip_path, desc="Downloading data.zip")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting", unit="file"):
            zip_ref.extract(member, extract_dir)

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
    download_url = _resolve_yandex_href(url)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    _stream_to_file(download_url, save_path, desc=f"Downloading {filename}")

    print(f"Downloaded to {save_path}")
    return save_path
