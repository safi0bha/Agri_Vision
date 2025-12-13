import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np

# Enhanced Model Architecture
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_sample_data():
    """Create sample images for training if no real data exists"""
    sample_dir = "sample_data/train"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample colored images for different classes
    classes = ['healthy', 'powdery_mildew', 'leaf_spot', 'blight', 'rust']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(sample_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 10 sample images per class
        for i in range(10):
            # Create different colored images for each class
            if class_name == 'healthy':
                color = (100, 200, 100)  # Green
            elif class_name == 'powdery_mildew':
                color = (200, 200, 200)  # White
            elif class_name == 'leaf_spot':
                color = (150, 100, 50)   # Brown
            elif class_name == 'blight':
                color = (100, 50, 50)    # Dark red
            else:  # rust
                color = (200, 150, 50)   # Orange
            
            img = Image.new('RGB', (128, 128), color=color)
            img.save(os.path.join(class_dir, f'sample_{i}.jpg'))

def train_model():
    # Create sample data if no dataset exists
    if not os.path.exists("sample_data/train"):
        print("📁 Creating sample training data...")
        create_sample_data()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder("sample_data/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = PlantDiseaseCNN(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🎯 Training on {len(dataset)} samples, {len(dataset.classes)} classes")
    print(f"📊 Classes: {dataset.classes}")
    
    # Training loop
    model.train()
    for epoch in range(5):  # Reduced epochs for quick training
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/plant_disease_model.pt')
    print("✅ Disease model trained and saved successfully!")
    print(f"🔮 Model can predict: {dataset.classes}")

if __name__ == '__main__':
    train_model()