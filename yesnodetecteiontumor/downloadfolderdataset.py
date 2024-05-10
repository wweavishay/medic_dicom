import subprocess
import os
import shutil
from zipfile import ZipFile

# Download dataset using Kaggle API
subprocess.run(["kaggle", "datasets", "download", "-d", "navoneel/brain-mri-images-for-brain-tumor-detection"])

# Unzip dataset
with ZipFile('brain-mri-images-for-brain-tumor-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('brain_mri_data')

# Move files to specified local folder
source_folder = 'brain_mri_data'
destination_folder = 'C:/Users/מיכאל/Desktop/medicproject/yesnodetecteiontumor'

for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)
    shutil.move(source_file, destination_file)
