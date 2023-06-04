import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 및 로드 (배치 정규화 포함)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

ensemble_size = 5  # 앙상블 모델 개수
learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4

ensemble_models = []
optimizers = []

for _ in range(ensemble_size):
    # Student Model (ResNet-18) 생성
    student_model = resnet18(pretrained=False)  # Pretrained weights not used
    num_ftrs = student_model.fc.in_features
    student_model.fc = nn.Linear(num_ftrs, 10)  # Change output size to match CIFAR-10 classes
    student_model = student_model.to(device)
    ensemble_models.append(student_model)  # 모델 추가
    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizers.append(optimizer)


def train_model(model, criterion, optimizer, scheduler, trainloader, num_epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        scheduler.step(epoch_loss)


def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


criterion = nn.CrossEntropyLoss()
ensemble_accuracy = 0.0
individual_accuracies = []

for idx, model in enumerate(ensemble_models):
    model = model.to(device)
    optimizer = optimizers[idx]
    scheduler = ReduceLROnPlateau(optimizer, patience=3)

    train_model(model, criterion, optimizer, scheduler, trainloader, num_epochs=10)
    accuracy = evaluate_model(model, testloader)
    individual_accuracies.append(accuracy)
    ensemble_accuracy += accuracy

    print(f"Model {idx + 1} Test Accuracy: {accuracy:.4f}")

ensemble_accuracy /= ensemble_size
print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")