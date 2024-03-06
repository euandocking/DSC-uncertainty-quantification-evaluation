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

from image_check import imshow

# Define a custom neural network model based on EfficientNetB0
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5):
        super(CustomEfficientNetB0, self).__init__()
        # Load the pre-trained EfficientNetB0 model
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        # Get the number of input features to the final fully connected layer
        num_ftrs = self.model._fc.in_features
        # Replace the final fully connected layer with an identity layer
        self.model._fc = nn.Identity()  # remove the original fully connected layer
        # Define additional fully connected layers
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = dropout_prob
    
    def forward(self, x, dropout=True):
        # Forward pass through the EfficientNetB0 model
        x = self.model(x)
        # Pass through the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout regularization if specified
        if dropout:
            x = F.dropout(x, p=self.dropout)
        # Pass through the final fully connected layer
        x = self.fc2(x)
        return x

def validate_model(model, criterion, data_loader, device, num_val_mc_samples=100, num_classes=1):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_list = [model(inputs).unsqueeze(0) for _ in range(num_val_mc_samples)]
            outputs_mean = torch.cat(outputs_list, dim=0).mean(dim=0)
            
            # Normalize the loss by the number of classes
            loss = criterion(outputs_mean, labels) / num_classes
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs_mean, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    return epoch_loss, epoch_acc

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, train_losses, train_accuracies, val_losses, val_accuracies, best_epoch, num_epochs=50, num_val_mc_samples=100, loss_weight=0.5, num_classes=1):
    since = time.time()

    best_combined_metric = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = 0.0
    best_val_acc = 0.0
    
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        for epoch in range(num_epochs):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'Epoch {epoch + 1}/{num_epochs} - {current_time}')
            print('-' * 10)

            for phase in ['train', 'val']:
                model.train(phase == 'train')
                data_loader = dataloaders[phase]

                if phase == 'train':
                    running_loss = 0.0
                    running_corrects = 0
                    total_samples = 0
                    
                    for inputs, labels in data_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Apply class-wise normalization
                        loss = loss / num_classes
                        
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == labels.data)
                        total_samples += labels.size(0)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / total_samples
                    print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, {phase.capitalize()} Acc: {epoch_acc:.4f}')
                    
                    if phase == 'train':
                        train_losses.append(epoch_loss)
                        train_accuracies.append(epoch_acc)
                    
                else:
                    epoch_loss, epoch_acc = validate_model(model, criterion, data_loader, device, num_val_mc_samples, num_classes)
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc)
                    print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, {phase.capitalize()} Acc: {epoch_acc:.4f}')
                    
                    # Calculate combined metric
                    combined_metric = epoch_acc - loss_weight * epoch_loss
                    if combined_metric > best_combined_metric:
                        best_combined_metric = combined_metric
                        best_val_loss = epoch_loss
                        best_val_acc = epoch_acc
                        best_epoch = epoch + 1  # Store the epoch number
                        torch.save(model.state_dict(), best_model_params_path)

            print()

        print(f'Best combined metric: {best_combined_metric:.4f}')
        print(f'Loss associated with the best combined metric: {best_val_loss:.4f}')
        print(f'Accuracy associated with the best combined metric: {best_val_acc:.4f}')
        print(f'Epoch associated with the best model: {best_epoch}')
        model.load_state_dict(torch.load(best_model_params_path))

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, best_epoch):
    # Move accuracies to CPU
    # not needed for losses as they were never moved to GPU
    train_accuracies = [acc.cpu().numpy() for acc in train_accuracies]
    val_accuracies = [acc.cpu().numpy() for acc in val_accuracies]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, label='Train')
    plt.plot(np.arange(1, len(val_losses)+1), val_losses, label='Validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')  # Add vertical line at best_epoch
    plt.legend().remove()  # Remove legend from the first subplot
    plt.xticks(np.arange(1, len(train_losses)+1))  # Set x-axis ticks from 1 to the number of epochs

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(train_accuracies)+1), train_accuracies, label='Train')
    plt.plot(np.arange(1, len(val_accuracies)+1), val_accuracies, label='Validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Add the best epoch to the legend for the second subplot
    handles, labels = plt.gca().get_legend_handles_labels()
    best_epoch_label = f'Best Epoch: {best_epoch}'
    handles.append(plt.Line2D([], [], color='r', linestyle='--', label=best_epoch_label))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.arange(1, len(train_accuracies)+1))  # Set x-axis ticks from 1 to the number of epochs

    plt.axvline(x=best_epoch, color='r', linestyle='--')  # Add vertical line at best_epoch

    plt.show()

def visualize_model(model, dataloader, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')

                predicted_class = class_names[preds[j]]
                actual_class = class_names[labels[j]]
                title = f'Predicted: {predicted_class}\nActual: {actual_class}'
                ax.set_title(title)

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)