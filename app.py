from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import os

app = Flask(__name__)

# -------------------------------
# Model Setup
# -------------------------------
class_names = ['person', 'person hold weapon', 'weapon']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("grayscale_threat_model.pt", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Transform for incoming image
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == "":
            return "No selected file"

        img = Image.open(file.stream).convert("L")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            class_name = class_names[predicted.item()]

        return render_template("index.html", prediction=class_name)

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def webcam_or_upload_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        class_name = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    return jsonify({
        "prediction": class_name,
        "confidence": round(confidence * 100, 2)
    })




if __name__ == "__main__":
    app.run(debug=True)
