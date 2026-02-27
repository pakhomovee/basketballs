import os
import zipfile
import requests

def download_data(self, public_key):
        if os.path.exists(self.video_path) and os.path.exists(self.gt_path):
            print("Data already exists.")
            return

        print("Downloading data...")
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
        final_url = base_url + 'public_key=' + public_key

        try:
            response = requests.get(final_url)
            download_url = response.json()['href']
            download_response = requests.get(download_url)

            with open('data.zip', 'wb') as f:
                f.write(download_response.content)

            with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Data downloaded and extracted.")
        except Exception as e:
            print(f"Error downloading data: {e}")
