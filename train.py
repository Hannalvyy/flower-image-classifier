import argparse
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

def load_data(data_dir):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
    return image_datasets

def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloaders['train'])}")

    return model

def save_checkpoint(model, save_dir):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }
    torch.save(checkpoint, save_dir)

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('data_directory', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Architecture of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Load data
    data = load_data(args.data_directory)
    dataloaders = {'train': torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)}

    # Load the model
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    model.classifier[6] = nn.Linear(4096, len(data.classes))
    model.class_to_idx = data.class_to_idx

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, device, args.epochs)

    # Save the checkpoint
    save_checkpoint(model, args.save_dir)

if __name__ == '__main__':
    main()
