import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import logging

# Logging for model.py
model_logger = logging.getLogger("model_logger")
model_logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler("logs/model.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Adding handlers
if not model_logger.handlers:
    model_logger.addHandler(console_handler)
    model_logger.addHandler(file_handler)

# Setting device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading class_to_idx mapping
with open("models/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

# Reverse mapping: idx to wnid
idx_to_wnid = {v: k for k, v in class_to_idx.items()}

# Loading words.txt for wnid to human-readable name
label_dict = {}
with open("models/words.txt", "r") as f:
    for line in f:
        wnid, name = line.strip().split("\t") if "\t" in line else line.strip().split(" ", 1)
        label_dict[wnid] = name

# idx to human-readable class
idx_to_class = {i: label_dict[wnid] for i, wnid in idx_to_wnid.items()}

# Preprocessing
transform_infer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Loading model
def load_model(model_path="models/efficientnet_b0_tinyimagenet.pth"):
    num_classes = len(class_to_idx)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# Loading model globally
model = load_model()

# Predict function
def predict(image: Image.Image):
    image = image.convert("RGB")
    model_logger.debug("Converted image to RGB.")
    input_tensor = transform_infer(image).unsqueeze(0).to(device)
    model_logger.debug("Converted image to tensor.")

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    wnid = idx_to_wnid[pred_idx]
    class_name = idx_to_class[pred_idx]
    
    model_logger.info(f"Prediction:\nwnid: {wnid}, class_name: {class_name}, confidence: {round(confidence, 4)}")

    return {
        "class_name": class_name,
        "wnid": wnid,
        "confidence": round(confidence, 4)
    }

def get_model_info(model=model):
    param_count = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / (1024**2)
    os.remove("temp.pth")

    return {
        "num_parameters": param_count,
        "model_size_mb": round(size_mb, 2)
    }

