from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import io
import numpy as np
from PIL import Image

# ML Imports
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO

from rag import process_query

app = FastAPI(title="Textile Dept RAG API", description="Project Chatbot Backend")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "textile-heritage")

# Mount the frontend directory for CSS, JS, and local images
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Keep the original static images for RAG results
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=STATIC_DIR), name="images")

# Configure Paths for Models
RESNET_CKPT = os.path.join(os.path.dirname(BASE_DIR), "resnet50_4saree_best.pt")
YOLO_CKPT = os.path.join(os.path.dirname(BASE_DIR), "runs/classify/YOLO11m_4class_v4/weights/best.pt")
CLASS_NAMES = ["baluchari", "maheshwari", "negammam", "phulkari"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_transform = transforms.Compose([
    transforms.Resize(672),
    transforms.CenterCrop(640),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("[API] Loading ResNet50 model (Eager, FP16)...")
resnet_model = models.resnet50(weights=None)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model.load_state_dict(torch.load(RESNET_CKPT, map_location=DEVICE, weights_only=True))
resnet_model = resnet_model.to(DEVICE).half().eval()

print("[API] Loading YOLO11m model (Eager, FP16)...")
yolo_model = YOLO(YOLO_CKPT)

class ChatRequest(BaseModel):
    query: str
    language: str = "English"

@app.get("/")
def read_root():
    """Serve the frontend index.html from the heritage directory."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    print(f"\n[API] Received query: '{req.query}' in {req.language}")
    result = process_query(req.query, req.language)
    return result

@app.post("/identify")
async def identify_design(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ResNet prediction (FP16)
        tensor = resnet_transform(img).unsqueeze(0).to(DEVICE).half()
        with torch.no_grad():
            logits = resnet_model(tensor)
            resnet_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        # YOLO prediction (FP16)
        results = yolo_model(img, half=True, verbose=False)[0]
        yolo_probs = results.probs.data.cpu().numpy()
        
        # Weighted Ensemble
        w_resnet = 0.15
        w_yolo = 0.85
        combined_probs = w_resnet * resnet_probs + w_yolo * yolo_probs
        
        pred_class_idx = int(np.argmax(combined_probs))
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(combined_probs[pred_class_idx])
        
        return {"class": pred_class, "confidence": confidence}
    except Exception as e:
        print(f"[API] Error in /identify: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run via: uvicorn app:app --host 0.0.0.0 --port 8000
