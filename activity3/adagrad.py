import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# Step 1: Load pre-trained ResNet101 model
resnet101 = models.resnet101(pretrained=True)

# Step 2: Replace the final fully connected layer
num_classes = 10  # Number of classes in SVHN dataset
resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)

# Step 3: Prepare SVHN dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 4: Define training loop and evaluation loop
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    train_losses = []
    train_accs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
    return train_losses, train_accs

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Step 5: Train the model using Adagrad optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet101 = resnet101.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(resnet101.parameters(), lr=0.01)  # Use Adagrad optimizer
train_losses, train_accs = train(resnet101, train_loader, criterion, optimizer)

# Step 6: Plot curves for training loss and training accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.legend()
plt.show()

# Step 7: Evaluate the model on the test dataset
test_accuracy = evaluate(resnet101, test_loader)
print(f'Top-5 Test Accuracy: {test_accuracy:.4f}')
