import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import os
import shutil
from tqdm import tqdm
import requests
from pathlib import Path


def _make_chrome_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(options=options)


def extract_url_information(driver: webdriver.Chrome, url: str):
    """Extracts URL information from a webpage using Selenium.

    Args:
        url (str): The URL of the webpage to extract information from.

    Returns:
        list: A list of dictionaries containing URL information.
    """

    driver.get(url)

    # Wait for the page to load
    random_seconds = np.random.randint(0, 2)
    wait = WebDriverWait(driver, random_seconds)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "GamePlayByPlayRow_row__2iX_w")))
    play_rows = driver.find_elements(By.CLASS_NAME, "GamePlayByPlayRow_row__2iX_w")

    url_info = []
    pts = "0-0"
    for row in play_rows:
        # Extract text
        try:
            desc = row.text.split("\n")
            info = ""
            time = desc[0]
            if len(desc) > 2:
                pts = desc[1]
                info = desc[2]
            else:
                info = desc[1]

            # Attempt to find any link within the row
            link = row.find_element(By.TAG_NAME, "a")  # Adjust if the link tag is different
            url = link.get_attribute("href") if link else "No link found"

            # Print the results
            is_home_team = row.get_attribute("data-is-home-team")

            url_info.append({"url": url, "time": time, "pts": pts, "home": is_home_team, "info": info})

        except:
            continue

    return url_info


def extract_mp4_urls(driver: webdriver.Chrome, url: str):
    """Extracts MP4 URLs from a webpage using Selenium.

    Args:
        url (str): The URL of the webpage to extract information from.

    Returns:
        list: A list of MP4 URLs.
    """

    driver.get(url)

    # Wait for the page to load

    wait = WebDriverWait(driver, 10)

    wait.until(EC.presence_of_element_located((By.TAG_NAME, "video")))

    visible_mp4_urls = [
        a.get_attribute("src")
        for a in driver.find_elements(By.TAG_NAME, "video")
        if a.is_displayed() and a.get_attribute("src").endswith(".mp4")
    ]

    # Combine visible and hidden MP4 URLs
    all_mp4_urls = visible_mp4_urls

    return all_mp4_urls


def download_with_progress(
    url: str,
    filename: str,
    chunk_size: int = 1024 * 256,
    timeout_s: tuple[float, float] = (15.0, 300.0),  # (connect, read)
) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.nba.com/",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }

    resp = requests.get(url, stream=True, timeout=timeout_s, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    total = resp.headers.get("Content-Length")
    total_size = int(total) if total and total.isdigit() else None

    with (
        open(filename, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(filename),
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))


# Path to your CSV
csv_path = Path(__file__).parent / "dataset.csv"

# Directory to save the downloaded files
output_dir = Path(__file__).parent / "data"
tmp_dir = Path(__file__).parent / "tmp"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path, sep=";")

driver = _make_chrome_driver()
try:
    # Download each URL as an MP4 (reuse the same Chrome tab)
    for counter, url in enumerate(df["urls"], start=1):
        try:
            final_path = os.path.join(output_dir, f"{counter}.mp4")
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                print(f"Skipping (already exists): {final_path}")
                continue
            tmp_path = os.path.join(tmp_dir, f"{counter}.mp4.part")
            print(f"Downloading {url} -> {final_path} (via {tmp_path})")
            specific_url = url
            mp4_urls = extract_mp4_urls(driver, specific_url)

            mp4_urls = mp4_urls[0]
            print(f"Downloading {mp4_urls} -> {tmp_path}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                download_with_progress(mp4_urls, tmp_path)
                shutil.copy2(tmp_path, final_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            print(f"Failed to download {url}: {e}")
finally:
    driver.quit()
