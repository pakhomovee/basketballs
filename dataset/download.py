import kagglehub
import shutil
import os

#kagglehub.dataset_download("gabrielvanzandycke/deepsport-dataset")
from components.common.utils import download_and_extract

download_and_extract('https://disk.yandex.ru/d/kftcFbk0_XOOyg', extract_dir='dataset')
