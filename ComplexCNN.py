import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 1. Dataset Loading and Visualization
def load_and_visualize_data(batch_size, valid_split=0.1):
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split training dataset into training and validation sets
    num_train = int((1 - valid_split) * len(dataset))
    num_valid = len(dataset) - num_train
    train_set, valid_set = torch.utils.data.random_split(dataset, [num_train, num_valid])
    
    # Create dataloaders
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Visualize a few images from the training set
    classes = dataset.classes
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    return trainloader, validloader, testloader, classes

# 2. Model Definitions
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Connection Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.residual_conv = nn.Conv2d(128, 256, kernel_size=1)  # For matching dimensions
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3 with Residual Connection
        residual = self.residual_conv(x)  # Match input dimensions
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = x + residual  # Add residual connection
        x = self.pool3(x)

        # Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 3. Training and Evaluation Functions
def train_model(model, trainloader, validloader, criterion, optimizer, epochs=20):
    model = model.to(device)  # Ensure model is on the correct device
    training_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        training_loss = running_loss / len(trainloader)
        training_losses.append(training_loss)
        
        # Validation Phase
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
        validation_loss = validation_loss / len(validloader)
        validation_losses.append(validation_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}")
    
    print('Finished Training')
    return training_losses, validation_losses


def test_model(model, testloader):
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 4. Hyperparameter and Model Tuning Function
def hyperparameter_tuning(param_grid):
    best_accuracy = 0.0
    best_params = None
    results = []

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in combinations:
        print(f"Testing combination: {params}")
        
        # Unpack parameters
        model_name = params['model']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        optimizer_name = params['optimizer']
        epochs = params['epochs']
        
        # Select the model
        if model_name == "SimpleCNN":
            model = SimpleCNN()
        elif model_name == "ComplexCNN":
            model = ComplexCNN()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load data
        trainloader, valloader,testloader, classes = load_and_visualize_data(batch_size)
        
        # Define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Train and evaluate
        train_model(model, trainloader, valloader, criterion, optimizer, epochs)
        accuracy = test_model(model, testloader)
        
        # Save results
        results.append({**params, "accuracy": accuracy})
        
        # Update best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"New best accuracy: {best_accuracy:.2f}% with params {best_params}")
    
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: {best_params}")
    return best_params, best_accuracy, results

# 5. Plotting Hyperparameter Tuning Results
def plot_tuning_results(results):
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        plt.scatter(subset.index, subset["accuracy"], label=model)
    
    plt.title("Hyperparameter Tuning Results")
    plt.xlabel("Combination Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_losses(training_losses, validation_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss", marker="o")
    plt.plot(validation_losses, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

# Main Script Execution
if __name__ == "__main__":
    
    param_grid = {
        'model': ["ComplexCNN", "SimpleCNN"],
        'batch_size': [32, 48, 64],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.01],
        'optimizer': ['adam', 'sgd'],
        'epochs': [20, 40, 60, 80]
    }
    
    best_params, best_accuracy, results = hyperparameter_tuning(param_grid)
    print(f"Hyperparameter Tuning Complete. Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: {best_params}")
    
    # Plot results
    plot_tuning_results(results)
    
    '''
    # Train Final Model
    final_model = ComplexCNN()
    trainloader, valloader, testloader, _ = load_and_visualize_data(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    training_losses, validation_losses = train_model(final_model, trainloader, valloader, criterion, optimizer, 10)

    plot_losses(training_losses, validation_losses)

    test_accuracy = test_model(final_model, testloader)
    '''
