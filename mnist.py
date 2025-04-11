#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MNIST Handwritten Digit Recognition CNN Model Training Script
Implemented with PyTorch 2.x
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# Set random seed to ensure reproducible results
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
DATA_PATH = "datasets"
MODEL_TYPE = "cnn"  # Default using CNN model, options: "cnn" or "nn"

# Define global loss function
criterion = nn.CrossEntropyLoss()

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer, input channels=1 (grayscale image), output channels=32, kernel size=3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer, input channels=32, output channels=64, kernel size=3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Pooling layer, used to reduce spatial dimensions of feature maps
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer, transforms feature maps to class predictions
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))  # Output size: 32x14x14
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))  # Output size: 64x7x7
        # Flatten feature maps
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define 3-layer perceptron model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # MNIST image size is 28x28=784, as input layer
        self.fc1 = nn.Linear(28 * 28, 312)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(312, 256)      # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(256, 10)       # Second hidden layer to output layer
        self.dropout = nn.Dropout(0.2)      # Add dropout to reduce overfitting
        
    def forward(self, x):
        # Ensure input has correct shape
        x = x.view(-1, 28 * 28)  # Flatten image to vector
        # First layer: input layer to first hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Second layer: first hidden layer to second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Third layer: second hidden layer to output layer
        x = self.fc3(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation of MNIST dataset
])

# Load MNIST dataset
print("Loading MNIST dataset...")
try:
    # Try to load dataset from local directory
    train_dataset = MNIST(root=DATA_PATH, train=True, download=False, transform=transform)
    test_dataset = MNIST(root=DATA_PATH, train=False, download=False, transform=transform)
    print(f"Successfully loaded dataset, training set size: {len(train_dataset)}, test set size: {len(test_dataset)}")
except Exception as e:
    print(f"Failed to load dataset from local: {e}")
    print("Trying to download MNIST dataset...")
    # If local loading fails, try downloading
    train_dataset = MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    print("Dataset download complete")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward propagation
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward propagation and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print training progress
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Epoch: {epoch}/{EPOCHS} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_loss/(batch_idx+1):.6f} '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    return train_loss / len(train_loader), 100. * correct / total

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# Visualize some samples
def visualize_samples(dataloader, num_samples=5):
    examples = iter(dataloader)
    samples, labels = next(examples)
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(samples[i][0], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.savefig('mnist_samples.png')
    plt.close()

# Train and evaluate model
def main(model_type="cnn"):
    # Initialize model, loss function and optimizer
    if model_type == "cnn":
        model = CNN().to(DEVICE)
        model_name = "mnist_cnn.pth"
    else:  # model_type == "nn"
        model = NeuralNetwork().to(DEVICE)
        model_name = "mnist_nn.pth"
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Visualize some samples
    print("Visualizing some samples...")
    visualize_samples(train_loader)
    
    # Record training process
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"Starting training, using device: {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, DEVICE, test_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # Save model
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")
    
    # Plot training process
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_type.upper()} Model - Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{model_type.upper()} Model - Accuracy Curves')
    
    # Save different filenames based on model type
    curves_filename = f"{model_type}_training_curves.png"
    plt.tight_layout()
    plt.savefig(curves_filename)
    plt.close()
    print(f"Training curves saved as {curves_filename}")
    
    return model_name

# Prediction function
def predict(model, device, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image)
        _, predicted = output.max(1)
        return predicted.item()

# Randomly select an image from test set and predict
def predict_random_test_image(model_type="cnn"):
    print("\nRandomly selecting an image from test set for prediction...")
    # Load trained model
    if model_type == "cnn":
        model = CNN().to(DEVICE)
        model.load_state_dict(torch.load("mnist_cnn.pth"))
    else:  # model_type == "nn"
        model = NeuralNetwork().to(DEVICE)
        model.load_state_dict(torch.load("mnist_nn.pth"))
    
    # Randomly select a sample from test set
    test_data = test_dataset.data
    test_targets = test_dataset.targets
    idx = torch.randint(0, len(test_dataset), (1,)).item()
    
    # Get image and label
    image = test_data[idx].float() / 255.0  # Normalize
    image = transforms.Normalize((0.1307,), (0.3081,))(image.unsqueeze(0))  # Apply same normalization
    true_label = test_targets[idx].item()
    
    # Save image
    plt.figure(figsize=(3, 3))
    plt.imshow(test_data[idx], cmap='gray')
    plt.title(f'True Label: {true_label}')
    plt.axis('off')
    plt.savefig('predict_image.png')
    plt.close()
    
    # Predict
    predicted_label = predict(model, DEVICE, image)
    
    print("Image saved as predict_image.png")
    print(f"True label: {true_label}")
    print(f"Prediction result: {predicted_label}")
    print(f"Prediction {'correct' if predicted_label == true_label else 'incorrect'}")

if __name__ == "__main__":
    import argparse
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='MNIST Handwritten Digit Recognition Model Training')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'nn'],
                        help='Select model type: cnn (Convolutional Neural Network) or nn (Multi-layer Perceptron)')
    args = parser.parse_args()
    
    # Directly pass command line arguments to function
    print(f"Using model type: {args.model}")
    
    model_name = main(args.model)
    predict_random_test_image(args.model)  # Execute random prediction after main program runs