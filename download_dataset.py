
import kagglehub

# Download Brain Tumor MRI Dataset
path = kagglehub.dataset_download(
    "masoudnickparvar/brain-tumor-mri-dataset"
)

print("Dataset downloaded successfully!")
print("Path to dataset files:", path)



import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download(
    "masoudnickparvar/brain-tumor-mri-dataset"
)

print("Downloaded location:", path)

# Create dataset folder inside project
destination = os.path.join(
    os.getcwd(),
    "brain-tumor-mri-dataset"
)

# Copy dataset into project folder
shutil.copytree(path, destination, dirs_exist_ok=True)

print("Dataset saved successfully!")
print("Dataset location:", destination)
