# Import necessary libraries
import os

from kaggle.api.kaggle_api_extended import KaggleApi

# Set the local path where you want to save the dataset
local_path = "./stead"

# Create the directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

# Initialize the Kaggle API
# You need to register kaggle and do pip install kaggle. To get kaggle.json please follow: https://www.kaggle.com/docs/api
api = KaggleApi()
api.authenticate()  # Make sure you've set up ~/.kaggle/kaggle.json

# Download the dataset
print("Downloading dataset...")
api.dataset_download_files(
    "isevilla/stanford-earthquake-dataset-stead",
    path=local_path,
    unzip=True,  # Set to True to automatically extract the downloaded zip
)

# List the files in the downloaded dataset
print(f"Dataset files: {os.listdir(local_path)}")
