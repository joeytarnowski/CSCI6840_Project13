import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Optimizer
optimizer_type = 'SGD'  # 'SGD' or 'Adam'

# Dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

# Splitting dataset into training and validation sets
train_size = int(0.8 * len(cifar_dataset))
val_size = len(cifar_dataset) - train_size
train_dataset, val_dataset = random_split(cifar_dataset, [train_size, val_size])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Define the Feedforward Neural Network Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if epoch+1 % 8 == 0:
            with open('output.txt', 'a') as file:
                print(f'Epoch [{epoch+1}/{num_epochs}]:\nTraining Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%\nValidation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%\n', file=file)
            print(f'Epoch [{epoch+1}/{num_epochs}]:\nTraining Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%\nValidation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

    return train_losses, val_losses, train_accs, val_accs

# Function to evaluate the model on the test set
def evaluate_model(name, model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    with open('output.txt', 'a') as file:
        print(f'Test Accuracy ({name}): {accuracy:.2f}%', file=file)
    print(f'Test Accuracy ({name}): {accuracy:.2f}%')
    return accuracy

# Function to plot the results of the training
def plot_results(name, train_losses, val_losses, train_accs, val_accs):
    # Plotting the loss and accuracy
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss ({name})')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'Training and Validation Accuracy ({name})')

    plt.show()

def plot_hyperparams(name, lr_avg, bs_avg, ne_avg, learning_rate, batch_size, num_epochs):
    plt.figure(figsize=(15, 5))

    # Average Accuracy vs Learning Rate
    sub1 = plt.subplot(1, 3, 1)
    plt.title(f'Average Accuracy for {name}')
    sub1.set_xlabel('Learning Rate')
    sub1.set_ylabel('Average Accuracy')
    
    # Use indices for x-axis to ensure even spacing
    x_lr = np.arange(len(learning_rate))
    plt.xticks(x_lr, learning_rate)  # Set x-ticks to actual learning rates
    plt.plot(x_lr, [np.mean(lr_avg[i]) for i in range(len(lr_avg))], marker='o')
    plt.legend([f'Avg Accuracy {name}'])

    # Average Accuracy vs Batch Size
    sub2 = plt.subplot(1, 3, 2)
    sub2.set_xlabel('Batch Size')
    sub2.set_ylabel('Average Accuracy')
    
    # Use indices for x-axis
    x_bs = np.arange(len(batch_size))
    plt.xticks(x_bs, batch_size)  # Set x-ticks to actual batch sizes
    plt.plot(x_bs, [np.mean(bs_avg[i]) for i in range(len(bs_avg))], marker='o')
    plt.legend([f'Avg Accuracy {name}'])

    # Average Accuracy vs Number of Epochs
    sub3 = plt.subplot(1, 3, 3)
    sub3.set_xlabel('Number of Epochs')
    sub3.set_ylabel('Average Accuracy')
    
    # Use indices for x-axis
    x_ne = np.arange(len(num_epochs))
    plt.xticks(x_ne, num_epochs)  # Set x-ticks to actual number of epochs
    plt.plot(x_ne, [np.mean(ne_avg[i]) for i in range(len(ne_avg))], marker='o')
    plt.legend([f'Avg Accuracy {name}'])

    plt.tight_layout()
    plt.show()
    
def test_hyperparams():
    with open('output.txt', 'w') as file:
        print("Beginning Hyperparameter Test", file=file)

    # Hyperparameters
    learning_rate = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
    batch_size = [8,16,32,64,128,256]
    num_epochs = [5,10,20,35,50,100]
    
    # Adam Optimization
    # Results Dictionary
    results = {}
    lr_avg = [[] for _ in range(len(learning_rate))]
    bs_avg = [[] for _ in range(len(batch_size))]
    ne_avg = [[] for _ in range(len(num_epochs))]
    num_tests = len(learning_rate) * len(batch_size) * len(num_epochs)
    curr_test = 0
    
    # Loop through all hyperparameter possibilities
    for lr in learning_rate:
      for bs in batch_size:
        for ne in num_epochs:
            curr_test += 1
            with open('output.txt', 'a') as file:
                print(f'\nTesting {curr_test}/{num_tests}\nLearning Rate: {lr}\nBatch Size: {bs}\nNumber of Epochs: {ne}\n', file=file)
            print(f'\nTesting {curr_test}/{num_tests}\nLearning Rate: {lr}\nBatch Size: {bs}\nNumber of Epochs: {ne}\n')
            # Data loaders
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        
            # Initialize the network, loss function, and SGD/Adam optimizer
            model = CNN()
            criterion = nn.CrossEntropyLoss()
            if optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr)
        
            # Training the model
            train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, ne, criterion, optimizer)
        
            # Evaluating the model on the test set
            acc = evaluate_model(f'lr={lr}, bs={bs}, ne={ne}', model, test_loader)
        
            # Save results
            results[f'lr={lr}, bs={bs}, ne={ne}'] = {
                'accuracy': acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
            lr_avg[learning_rate.index(lr)].append(acc)
            bs_avg[batch_size.index(bs)].append(acc)
            ne_avg[num_epochs.index(ne)].append(acc)
    
    # Plotting the results of hyperparameter test
    plot_hyperparams('SGD Optimization', lr_avg, bs_avg, ne_avg, learning_rate, batch_size, num_epochs)
    
    largest_acc = [0,'none']
    for key in results:
        if results[key]['accuracy'] > largest_acc[0]:
            largest_acc[0] = results[key]['accuracy']
            largest_acc[1] = key
    with open('output.txt', 'a') as file:
        print(f'The best hyperparameters for SGD Optimization are {largest_acc[1]} with an accuracy of {largest_acc[0]}', file=file)
    print(f'The best hyperparameters for SGD Optimization are {largest_acc[1]} with an accuracy of {largest_acc[0]}')

# Call the hyperparameter test function - comment out to only test a single model
test_hyperparams()

# Train and test single model
# Data loaders
'''
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the network, loss function, and SGD/Adam optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.1)

# Training the model
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, 50, criterion, optimizer)
plot_results(optimizer_type, train_losses, val_losses, train_accs, val_accs)
evaluate_model(optimizer_type, model, test_loader)
'''