# Import necessary libraries
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import h5py

# Set the local path where you want to save the dataset
local_path = "./stead"

# Create the directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()  # Make sure you've set up ~/.kaggle/kaggle.json

# Download the dataset
print("Downloading dataset...")
api.dataset_download_files(
    "isevilla/stanford-earthquake-dataset-stead", 
    path=local_path,
    unzip=True  # Set to True to automatically extract the downloaded zip
)

# List the files in the downloaded dataset
print(f"Dataset files: {os.listdir(local_path)}")

