import os
import shutil
import random

# Define paths
parasitized_dir = 'D:\rohan\Final Year Project 7-1-2024\malaria_detection_project\malaria_detection\Parasitized'
uninfected_dir = 'D:\rohan\Final Year Project 7-1-2024\malaria_detection_project\malaria_detection\Uninfected'
train_dir = 'D:\rohan\Final Year Project 7-1-2024\malaria_detection_project\malaria_detection\path_to_train_directory'
validation_dir = 'malaria_detection_project/malaria_detection/path_to_validation_directory'

# Create train and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Function to move files from source directory to destination directory
def move_files(source_dir, dest_dir, files):
    os.makedirs(dest_dir, exist_ok=True)
    for file in files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(dest_dir, file)
        shutil.copy(src, dst)

# List files in the parasitized directory
parasitized_files = os.listdir(parasitized_dir)
# List files in the uninfected directory
uninfected_files = os.listdir(uninfected_dir)

# Shuffle the files randomly
random.shuffle(parasitized_files)
random.shuffle(uninfected_files)

# Split data into train and validation sets
split_ratio = 0.8  # 80% for training, 20% for validation
parasitized_train = parasitized_files[:int(len(parasitized_files) * split_ratio)]
parasitized_validation = parasitized_files[int(len(parasitized_files) * split_ratio):]
uninfected_train = uninfected_files[:int(len(uninfected_files) * split_ratio)]
uninfected_validation = uninfected_files[int(len(uninfected_files) * split_ratio):]

# Move files to train directory
move_files(parasitized_dir, os.path.join(train_dir, 'parasitized'), parasitized_train)
move_files(uninfected_dir, os.path.join(train_dir, 'uninfected'), uninfected_train)

# Move files to validation directory
move_files(parasitized_dir, os.path.join(validation_dir, 'parasitized'), parasitized_validation)
move_files(uninfected_dir, os.path.join(validation_dir, 'uninfected'), uninfected_validation)

print("Data splitting completed!")
