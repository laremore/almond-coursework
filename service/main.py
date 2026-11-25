from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
from torchvision import transforms
import io

import sys

sys.path.append(".")
from src.model import get_base_resnet18

app = FastAPI()

DEVICE = torch.device("cpu")
CLASS_NAMES = ["Ak", "Nurlu", "Sıra", "Kapadokya"]
MODEL_PATH = "best_resnet18_focal_almond.pth"

model = None


@app.on_event("startup")
def load_model():
    global model
    model = get_base_resnet18(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    return {
        "class": CLASS_NAMES[pred_idx],
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }