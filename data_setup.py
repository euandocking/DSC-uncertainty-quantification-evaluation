#data_setup.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from tempfile import TemporaryDirectory
import pandas as pd
import shutil
import random
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import time
from datetime import datetime
import datetime as dt
import copy
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

# function to sort the images into class folders based on given csv file
def copy_images_to_folders(csv_path, image_folder, output_folder):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        image_name = row['image'] + '.jpg'  # Assuming images have '.jpg' extension
        image_path = os.path.join(image_folder, image_name)

        # Iterate through class columns and copy images to the corresponding folders
        for class_name in df.columns[1:]:
            class_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)  # Create folder if it doesn't exist

            if row[class_name] == 1:  # Check if the image belongs to the class
                shutil.copy(image_path, class_folder)

def create_folders(*folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def remove_folders(*folders):
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)

def split_data_for_class(class_folder_input, class_folder_output, images, ratio, sample_limit=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if sample_limit is not None:
        images = images[:sample_limit]

    np.random.shuffle(images)
    num_samples = int(len(images) * ratio)
    for image_name in images[:num_samples]:
        shutil.copy(os.path.join(class_folder_input, image_name), class_folder_output)

def split_data(sorted_folder, training_folder, validation_folder, test_folder, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, sample_limit=None, seed=None):
    create_folders(training_folder, validation_folder, test_folder)

    for class_name in os.listdir(sorted_folder):
        class_folder_input = os.path.join(sorted_folder, class_name)
        if os.path.isdir(class_folder_input):
            images = [img for img in os.listdir(class_folder_input) if img.endswith('.jpg')]

            class_folder_training = os.path.join(training_folder, class_name)
            class_folder_validation = os.path.join(validation_folder, class_name)
            class_folder_test = os.path.join(test_folder, class_name)

            create_folders(class_folder_training, class_folder_validation, class_folder_test)

            split_data_for_class(class_folder_input, class_folder_training, images, train_ratio, sample_limit, seed)
            split_data_for_class(class_folder_input, class_folder_validation, images, val_ratio, sample_limit, seed)
            split_data_for_class(class_folder_input, class_folder_test, images, test_ratio, sample_limit, seed)