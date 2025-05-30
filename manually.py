import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch import nn
from PIL import Image

# ========================
# 1. Model Setup
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # Change 3 to your number of classes if needed
model = model.to(device)

# ========================
# 2. Load Saved Weights
# ========================

model.load_state_dict(torch.load("grayscale_threat_model.pt", map_location=device))
model.eval()

# ========================
# 3. Load Class Names
# ========================

with open('valid/_classes.txt', 'r') as cf:
    class_names = [line.strip() for line in cf.readlines()]

# ========================
# 4. Image Transform
# ========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========================
# 5. Prediction Function
# ========================

def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # Grayscale
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0, predicted.item()].item()
    print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")

# ========================
# 6. Predict on a Sample Image
# ========================

# Replace with your image path
image_path = "waking_1.jpg"
predict_image(image_path)
