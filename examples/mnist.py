import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import waddle


def main():
    # Initialize Waddle
    waddle.init(project='mnist_example')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28*28, 500)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = x.view(-1, 28*28)
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = SimpleNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_step = epoch * total_steps + i + 1

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

                # Log metrics to Waddle
                waddle.log(category='training', data={'loss': loss.item()}, step=current_step)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

        # Log test accuracy to Waddle
        waddle.log(category='test', data={'accuracy': accuracy}, step=num_epochs * total_steps)

    # Finish Waddle logging
    waddle.finish()

if __name__ == '__main__':
    main()
