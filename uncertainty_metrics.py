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
import csv

def export_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Label', 'AURC'])  # Write header
        csv_writer.writerows(data)

def calculate_risks(guesses_are_correct, uncertainties, targets):
    # Error handling
    if len(guesses_are_correct) != len(uncertainties) or len(uncertainties) != len(targets):
        raise ValueError("Lengths of guesses_are_correct, uncertainties, and targets must be equal.")
    
    risks = []
    class_names = np.unique(targets)
    
    while uncertainties:
        # Accumulate correct guesses for each class
        guesses_are_correct_by_class = [[] for _ in range(len(class_names))]
        for target_index, target in enumerate(targets):
            class_index = np.where(class_names == target)[0][0]
            guesses_are_correct_by_class[class_index].append(guesses_are_correct[target_index])
        
        # Calculate class accuracies
        class_accuracies = []
        for class_correct in guesses_are_correct_by_class:
            if len(class_correct) > 0:  # Only calculate accuracy if there are samples for the class
                class_accuracies.append(np.mean(class_correct))
        
        mean_risk = 1 - np.mean(class_accuracies)
        
        # Append to risks list
        risks.append(mean_risk)
        
        # Remove results for the least certain sample
        max_uncertainty_index = np.argmax(uncertainties)
        uncertainties.pop(max_uncertainty_index)
        guesses_are_correct.pop(max_uncertainty_index)
        targets.pop(max_uncertainty_index)
    
    risks.reverse()

    return risks

def process_labels(sample_labels, guesses_are_correct, uncertainties, class_names):
    guesses_are_correct_by_class = [[] for _ in range(len(class_names))]
    uncertainties_by_class = [[] for _ in range(len(class_names))]

    for label, correct_guess, uncertainty in zip(sample_labels, guesses_are_correct, uncertainties):
        guesses_are_correct_by_class[label].append(correct_guess)
        uncertainties_by_class[label].append(uncertainty)

    return guesses_are_correct_by_class, uncertainties_by_class

def calculate_and_append_risks(risks_list, labels_list, guesses_are_correct, uncertainties, labels, targets):
    risks = calculate_risks(guesses_are_correct, uncertainties, targets)
    risks_list.append(risks)
    labels_list.append(labels)

def calculate_and_append_risks_by_class(risks_list_by_class, labels_list_by_class, guesses_are_correct_by_class, uncertainties_by_class, class_names, labels):
    for i in range(len(class_names)):
        risks = calculate_risks(guesses_are_correct_by_class[i], uncertainties_by_class[i])
        risks_list_by_class[i].append(risks)
        labels_list_by_class[i].append(labels)

def process_uncertainties(risks_list, labels_list, risks_list_by_class, labels_list_by_class, model, dataloader, class_names, device, uncertainty_function, uncertainty_name, *args, num_samples=None):
    # Calculate uncertainties using the provided function
    if num_samples is not None:
        guesses_are_correct, uncertainties, sample_labels = uncertainty_function(model, dataloader, class_names, device, num_samples)
    else:
        guesses_are_correct, uncertainties, sample_labels = uncertainty_function(model, dataloader, class_names, device)
    
    # Process labels and uncertainties by class
    guesses_are_correct_by_class, uncertainties_by_class = process_labels(sample_labels, guesses_are_correct, uncertainties, class_names)
    
    # Append risks and labels to the global lists
    calculate_and_append_risks(risks_list, labels_list, guesses_are_correct, uncertainties, uncertainty_name, sample_labels)
    calculate_and_append_risks_by_class(risks_list_by_class, labels_list_by_class, guesses_are_correct_by_class, uncertainties_by_class, class_names, uncertainty_name)
    
    print(uncertainty_name, "processed")

def calculate_softmax_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # for every batch in the dataloader
        for inputs, labels in dataloader:
            #get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            # get the outputs
            outputs = model(inputs, dropout=False).to(device)  # Move outputs to device
            # get the softmax outputs
            probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
            # get the class predictions
            _, predicted = torch.max(outputs, 1)
            # check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Compute uncertainty based on softmax probabilities
            for idx in range(len(predicted)):
                predicted_class = predicted[idx].item()
                uncertainty = 1.0 - probabilities[idx, predicted_class].item()  # Using confidence as uncertainty
                uncertainties.append(uncertainty)

    return guesses_are_correct, uncertainties, sample_labels


def calculate_top2_softmax_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # For every batch in the dataloader
        for inputs, labels in dataloader:
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            # Get the outputs
            outputs = model(inputs, dropout=False).to(device)  # Move outputs to device
            # Get the softmax outputs
            probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
            # Get the top two softmax values and their indices
            top2_probs, top2_indices = torch.topk(probabilities, k=2, dim=1)

            # Get the difference between the top two softmax values
            top2_diff = top2_probs[:, 0] - top2_probs[:, 1]

            # Get the class predictions
            _, predicted = torch.max(outputs, 1)
            # Check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Store the uncertainty as the difference between the top two softmax values
            uncertainties.extend(-top2_diff.cpu().numpy())

    return guesses_are_correct, uncertainties, sample_labels


def calculate_random_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            outputs = model(inputs, dropout=False).to(device)
            _, predicted = torch.max(outputs, 1)
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Generate a single random uncertainty value for each predicted class
            for idx in range(len(predicted)):
                uncertainty = random.random()
                uncertainties.append(uncertainty)

    return guesses_are_correct, uncertainties, sample_labels

def calculate_mc_dropout_uncertainties_by_sample(model, dataloader, class_names, device, num_samples=100):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            mean_predictions = torch.zeros(num_samples, inputs.size(0), len(class_names)).to(device)

            # Generate predictions with dropout for multiple samples
            for i in range(num_samples):
                outputs = model(inputs)
                predictions = F.softmax(outputs, dim=1).to(device)  # Move softmax operation to device
                mean_predictions[i] = predictions

            # Calculate mean prediction across samples
            mean_prediction = torch.mean(mean_predictions, dim=0)

            # Calculate uncertainty using mean prediction
            uncertainties.extend(1.0 - torch.max(mean_prediction, dim=1)[0].cpu().numpy())

            # Determine correctness of predictions
            _, predicted = torch.max(mean_prediction, 1)
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Print date/time for monitoring
            print(f"{dt.datetime.now()} - Batch {batch_idx + 1}/{len(dataloader)} processed")

    return guesses_are_correct, uncertainties, sample_labels

def calculate_mc_dropout_uncertainties_by_class(model, dataloader, class_names, device, num_samples=100):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            logits_list = []  # Initialize a list to store logits for each sample

            # Generate predictions with dropout for multiple samples
            for i in range(num_samples):
                outputs = model(inputs)
                logits_list.append(outputs)

            # Concatenate logits along a new dimension to create a tensor
            logits_tensor = torch.stack(logits_list, dim=0).to(device)

            # Calculate average output across all samples for each class
            class_avg_outputs = torch.mean(logits_tensor, dim=0)

            # Calculate uncertainty using the highest average output
            uncertainties.extend(1.0 - torch.max(class_avg_outputs, dim=1)[0].cpu().numpy())

            # Determine correctness of predictions
            _, predicted = torch.max(class_avg_outputs, 1)
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Print date/time for monitoring
            print(f"{dt.datetime.now()} - Batch {batch_idx + 1}/{len(dataloader)} processed")

    return guesses_are_correct, uncertainties, sample_labels

def calculate_variance_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # For every batch in the dataloader
        for inputs, labels in dataloader:
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            # Get the outputs
            outputs = model(inputs, dropout=False).to(device)  # Move outputs to device
            # Get the softmax outputs
            probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
            # Get the class predictions
            _, predicted = torch.max(outputs, 1)
            # Check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Compute uncertainty based on variance over classes
            variance = torch.var(probabilities, dim=1)
            uncertainties.extend(-variance.cpu().numpy())

    return guesses_are_correct, uncertainties, sample_labels

def calculate_variational_ratio_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # For every batch in the dataloader
        for inputs, labels in dataloader:
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            # Get the outputs
            outputs = model(inputs, dropout=False).to(device)  # Move outputs to device
            # Get the softmax outputs
            probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
            # Get the class predictions
            _, predicted = torch.max(outputs, 1)
            # Check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Compute uncertainty based on Variational Ratios
            mode_probabilities, _ = torch.max(probabilities, dim=1, keepdim=True)
            other_probabilities = probabilities.clone()
            other_probabilities.scatter_(1, predicted.view(-1, 1), 0)  # Set mode probabilities to 0
            max_other_probabilities, _ = torch.max(other_probabilities, dim=1, keepdim=True)
            variational_ratio = 1.0 - mode_probabilities / max_other_probabilities
            uncertainties.extend(variational_ratio.cpu().numpy())

    return guesses_are_correct, uncertainties, sample_labels

def calculate_variational_ratio_dropout_uncertainties(model, dataloader, class_names, device, num_samples=100):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # For every batch in the dataloader
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            predictions = []
            for _ in range(num_samples):
                # Get the outputs with dropout
                outputs = model(inputs, dropout=True).to(device)  # Move outputs to device
                predictions.append(F.softmax(outputs, dim=1).cpu().numpy())

            predictions = torch.tensor(np.array(predictions)).to(device)

            # Compute mean probabilities and uncertainties based on Variational Ratios
            mean_probabilities = torch.mean(predictions, dim=0)
            mode_probabilities, _ = torch.max(mean_probabilities, dim=1, keepdim=True)
            other_probabilities = mean_probabilities.clone()
            other_probabilities.scatter_(1, torch.argmax(mean_probabilities, dim=1).view(-1, 1), 0)  # Set mode probabilities to 0
            max_other_probabilities, _ = torch.max(other_probabilities, dim=1, keepdim=True)
            variational_ratio = 1.0 - mode_probabilities / max_other_probabilities
            uncertainties.extend(variational_ratio.cpu().numpy())

            # Get the class predictions
            _, predicted = torch.max(mean_probabilities, 1)
            # Check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())
            
            # Print date/time for monitoring
            print(f"{dt.datetime.now()} - Batch {batch_idx + 1}/{len(dataloader)} processed")

    return guesses_are_correct, uncertainties, sample_labels

def calculate_entropy_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # For every batch in the dataloader
        for inputs, labels in dataloader:
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            # Get the outputs
            outputs = model(inputs, dropout=True).to(device)  # Move outputs to device
            # Get the softmax outputs
            probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
            # Get the class predictions
            _, predicted = torch.max(outputs, 1)
            # Check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Compute uncertainty based on predictive entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # Adding a small value to avoid log(0)
            uncertainties.extend(entropy.cpu().numpy())

    return guesses_are_correct, uncertainties, sample_labels

def calculate_predictive_entropy_uncertainties(model, dataloader, class_names, device, num_samples=100):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            entropy_list = []  # Initialize a list to store entropy for each sample

            # Generate predictions with dropout for multiple samples
            for i in range(num_samples):
                # Get the outputs with dropout
                outputs = model(inputs, dropout=True).to(device)  # Move outputs to device
                probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # Adding a small value to avoid log(0)
                entropy_list.append(entropy)  # Store entropy for each sample

            # Concatenate entropies along a new dimension to create a tensor
            entropy_tensor = torch.stack(entropy_list, dim=0).to(device)

            # Calculate average entropy across all samples
            avg_entropy = torch.mean(entropy_tensor, dim=0)

            uncertainties.extend(avg_entropy.cpu().numpy())

            # Get the class predictions based on the maximum probability
            _, predicted = torch.max(outputs, 1)

            # Determine correctness of predictions
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Print date/time for monitoring
            print(f"{dt.datetime.now()} - Batch {batch_idx + 1}/{len(dataloader)} processed")

    return guesses_are_correct, uncertainties, sample_labels


def calculate_mutual_information_uncertainties(model, dataloader, class_names, device):
    model.eval()  # Set the model to evaluation mode
    guesses_are_correct = []
    uncertainties = []
    sample_labels = []

    with torch.no_grad():
        # For every batch in the dataloader
        for inputs, labels in dataloader:
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            # Get the outputs
            outputs = model(inputs, dropout=False).to(device)  # Move outputs to device
            # Get the softmax outputs
            probabilities = F.softmax(outputs, dim=1).to(device)  # Move probabilities to device
            # Get the class predictions
            _, predicted = torch.max(outputs, 1)
            # Check and store whether predictions are correct
            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Compute uncertainty based on mutual information
            uniform_distribution = torch.full_like(probabilities, 1.0 / probabilities.size(1)).to(device)  # Move uniform_distribution to device
            mutual_information = F.kl_div(probabilities.log(), uniform_distribution, reduction='none').sum(dim=1)
            uncertainties.extend(-mutual_information.cpu().numpy())

    return guesses_are_correct, uncertainties, sample_labels

def calculate_mutual_information_mc_dropout(model, dataloader, class_names, device, num_samples=100):
    model.eval()  # Set the model to evaluation mode
    uncertainties = []
    guesses_are_correct = []
    sample_labels = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_labels.extend(labels.cpu().numpy())

            logits_list = []  # Initialize a list to store logits for each sample

            # Generate predictions with dropout for multiple samples
            for i in range(num_samples):
                # Get the outputs with dropout
                outputs = model(inputs, dropout=True).to(device)  # Move outputs to device
                logits_list.append(outputs)  # Store logits for each sample

            # Concatenate logits along a new dimension to create a tensor
            logits_tensor = torch.stack(logits_list, dim=0).to(device)

            # Calculate softmax probabilities
            probabilities = F.softmax(logits_tensor, dim=2).to(device)

            # Compute uncertainty based on mutual information
            uniform_distribution = torch.full_like(probabilities, 1.0 / probabilities.size(2)).to(device)  # Move uniform_distribution to device
            mutual_information = F.kl_div(probabilities.log(), uniform_distribution, reduction='none').sum(dim=2)

            # Average mutual information across samples
            avg_mutual_information = mutual_information.mean(dim=0)

            # Store mutual information
            uncertainties.extend(-avg_mutual_information.cpu().numpy())

            # Determine correctness of predictions
            _, predicted = torch.max(logits_tensor.mean(dim=0), 1)

            correct_guesses = (predicted == labels)
            guesses_are_correct.extend(correct_guesses.cpu().numpy())

            # Print date/time for monitoring
            print(f"{dt.datetime.now()} - Batch {batch_idx + 1}/{len(dataloader)} processed")

    return guesses_are_correct, uncertainties, sample_labels

def smooth_calcs(risks):
    # Interpolate the data for smoother curves
    x_values = np.arange(len(risks))
    total_steps = len(x_values)
    
    # Convert x-values to percentage
    x_smooth_percentage = (x_values / total_steps)  # Already between 0 and 1
    
    # Create a smooth range for the x-axis
    x_smooth_percentage_interp = np.linspace(x_smooth_percentage.min(), x_smooth_percentage.max(), 300)

    return x_smooth_percentage_interp, x_smooth_percentage

def calculate_aurc(risks_list, labels_list, x_smooth_percentage_interp, x_smooth_percentage):
    aurc_cutoff_list = []  # For 20% cutoff
    aurc_full_list = []    # For the entire curve
    
    for risks, label in zip(risks_list, labels_list):
        risks_smooth = np.interp(x_smooth_percentage_interp, x_smooth_percentage, risks)
        
        # Find indices corresponding to values less than or equal to 20% coverage
        cutoff_index = np.argmax(x_smooth_percentage_interp >= 0.2)  # 20% coverage in normalized scale

        # Calculate area under the curve from 100% coverage to the 20% cutoff
        area_cutoff = np.trapz(risks_smooth[cutoff_index:], x=x_smooth_percentage_interp[cutoff_index:])
        aurc_cutoff_list.append((label, area_cutoff))

        # Calculate area under the entire curve
        area_full = np.trapz(risks_smooth, x=x_smooth_percentage_interp)
        aurc_full_list.append((label, area_full))

    # Sort the lists based on AURC values
    aurc_cutoff_list.sort(key=lambda x: x[1])  # Sorting based on AURC for 20% cutoff
    aurc_full_list.sort(key=lambda x: x[1])    # Sorting based on AURC for the entire curve
    
    # Modify risks_list and labels_list in place based on sorted AURC lists
    for i, (label, _) in enumerate(aurc_full_list):
        index = labels_list.index(label)
        risks_list[i], risks_list[index] = risks_list[index], risks_list[i]
        labels_list[i], labels_list[index] = labels_list[index], labels_list[i]

    # Print the AURC values
    max_label_length = max(len(item[0]) for item in aurc_cutoff_list + aurc_full_list)

    print("\nAURC (full):")
    print("Label" + " " * (max_label_length - 5) + "\tAURC")
    print("-" * (max_label_length + 15))
    for label, aurc_full in aurc_full_list:
        print(f"{label}" + " " * (max_label_length - len(label) + 5) + f"\t{aurc_full:.4f}")
    
    print("\nAURC (20% cutoff):")
    print("Label" + " " * (max_label_length - 5) + "\tAURC")
    print("-" * (max_label_length + 15))
    for label, aurc_cutoff in aurc_cutoff_list:
        print(f"{label}" + " " * (max_label_length - len(label) + 5) + f"\t{aurc_cutoff:.4f}")

    return aurc_full_list, aurc_cutoff_list

def plot_risk_coverage(risks_list, labels_list, x_smooth_percentage_interp, x_smooth_percentage, save_dir):
    max_risk_100_coverage = 0  # Initialize max risk value at 100% coverage
    
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed (width, height)

    for risks, label in zip(risks_list, labels_list):
        risks_smooth = np.interp(x_smooth_percentage_interp, x_smooth_percentage, risks)
        plt.plot(x_smooth_percentage_interp, risks_smooth, label=label, linewidth=1)
        
        # Update max risk value at 100% coverage if necessary
        max_risk_100_coverage = max(max_risk_100_coverage, risks_smooth[-1])

    # x-axis label with a percentage
    plt.xlabel('Coverage')
    plt.ylabel('1 - Mean Class Accuracy')
    plt.title(f"Risk vs. Coverage {save_dir}")

    # Add a vertical line at approximately 20% coverage
    plt.axvline(x=0.2, color='red', linestyle='--', label='20% Coverage')

    # Adjust the x-axis limits to range from 0 to 1
    plt.xlim(0, 1)

    # Adjust the y-axis limits
    plt.ylim(0, max_risk_100_coverage * 1.1)  # Set the upper limit slightly higher than the maximum risk at 100% coverage

    # Show legend outside of the graph to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust figure size to accommodate legend outside of the plot
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'risk_coverage.png'))
    
    plt.show()

# Function to select desired metrics for a given list of labels and risks
def select_desired_metrics(labels_list, risks_list, desired_metrics):
    selected_labels_list = []
    selected_risks_list = []

    for metric_name in labels_list:
        if metric_name in desired_metrics:
            selected_labels_list.append(metric_name)
            index = labels_list.index(metric_name)
            selected_risks_list.append(risks_list[index])

    return selected_labels_list, selected_risks_list