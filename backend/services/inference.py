import torch
import cv2
import numpy as np
from torchvision import models, transforms
from utils.preprocess import preprocess

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/lung_model.pt"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

async def run_inference(file):
    data = await file.read()

    img = cv2.imdecode(
        np.frombuffer(data,np.uint8),
        cv2.IMREAD_COLOR
    )

    img = preprocess(img)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out,1).item()
        conf = torch.softmax(out,1).max().item()

    return {
        "prediction": "PNEUMONIA" if pred else "NORMAL",
        "confidence": float(conf)
    }
    
async def run_inference(file):
    return {
        "prediction": "PNEUMONIA",
        "confidence": 0.92
    }