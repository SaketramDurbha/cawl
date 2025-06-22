import glob
import os
import urllib
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class AISDataDownloader:
    def __init__(self):
        self.base_url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
        self.download_path = "data/"

    def download_date_range(self, date_range: pd.core.indexes.datetimes.DatetimeIndex):
        """Download daily AIS data for a given date range formatted as a pandas DatetimeIndex."""
        for date in tqdm(date_range, desc="Downloading AIS data"):
            # Check if file is downloaded before making HTML request
            url = urllib.parse.urljoin(
                self.base_url, f'{date.year}/AIS_{date.strftime("%Y_%m_%d")}.zip'
            )
            zip_path = Path(f'data/AIS_{date.strftime("%Y_%m-%d")}.zip')
            csv_path = Path(f'data/AIS_{date.strftime("%Y_%m_%d")}.csv')

            if csv_path.exists():
                print(f"File already exists: {csv_path}")
                continue
            else:
                response = requests.get(url)

                # Write to file if html response is successful
                if response.status_code == 200:
                    with open(zip_path, "wb") as f:
                        f.write(response.content)
                        for chunk in response.iter_content(chunk_size=128):
                            if chunk:  # Filter out keep-alive new chunks
                                f.write(chunk)

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(self.download_path)
                else:
                    print(
                        f"Failed to download file: {url}. Status code: {response.status_code}"
                    )

    def delete_zip_files(self):
        """Delete all zip files in the download directory."""
        zip_files = glob.glob(os.path.join(self.download_path, "*.zip"))

        for file in zip_files:
            os.remove(file)
