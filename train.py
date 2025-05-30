import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm  # Add at the top with your imports

# ========================
# 1. Dataset Class
# ========================

class GrayscaleThreatDataset(Dataset):
    def __init__(self, image_dir, annotation_file, class_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(class_file, 'r') as cf:
            self.class_names = [line.strip() for line in cf.readlines()]

        with open(annotation_file, 'r') as af:
            lines = af.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                label_ids = []
                for p in parts[1:]:
                    try:
                        label_id = int(p.split(',')[4])
                        label_ids.append(label_id)
                    except:
                        print(f"Error parsing annotation: {p}")
                        continue
                if label_ids:
                    label = max(label_ids)
                    self.samples.append((os.path.join(image_dir, img_name), label))
                else:
                    print(f"No valid label found for line: {line}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            raise
        if self.transform:
            img = self.transform(img)
        return img, label

# ========================
# 2. Transforms
# ========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========================
# 3. Load Datasets
# ========================

print("Loading datasets...")

train_dataset = GrayscaleThreatDataset(
    image_dir='train',
    annotation_file='train/_annotations.txt',
    class_file='train/_classes.txt',
    transform=transform
)
val_dataset = GrayscaleThreatDataset(
    image_dir='valid',
    annotation_file='valid/_annotations.txt',
    class_file='valid/_classes.txt',
    transform=transform
)
test_dataset = GrayscaleThreatDataset(
    image_dir='test',
    annotation_file='test/_annotations.txt',
    class_file='test/_classes.txt',
    transform=transform
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ========================
# 4. Model Setup
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale input
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(device)



# ========================
# 5. Training (with Progress Bar)
# ========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Starting training...")
for epoch in range(10):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1} started...")

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    for i, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} complete. Avg Loss = {running_loss / len(train_loader):.4f}")

# ========================
# 6. Evaluation
# ========================

def evaluate(model, loader, dataset_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"{dataset_name} Accuracy: {acc:.2f}%")

evaluate(model, val_loader, "Validation")
evaluate(model, test_loader, "Test")

# ========================
# 7. Save Model
# ========================

torch.save(model.state_dict(), "grayscale_threat_model.pt")
print("✅ Model saved to grayscale_threat_model.pt")
