from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import torch
from PIL import Image
import io
from app.model.grad_cam import predict_and_visualize_gradcam
from app.model.predict import predict
from fastapi.middleware.cors import CORSMiddleware
from app.model.grad_cam import ViTBinaryClassifier
from app.model.predict import CNNBinaryClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,   
    allow_methods=["*"],      
    allow_headers=["*"],      
)
# Bỏ image_path cũ, chỉ giữ model path thôi
model_path = os.getenv("MODEL_PATH")
model_vit_path = os.getenv("MODEL_VIT")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_vit = ViTBinaryClassifier()
checkpoint = torch.load(model_vit_path, map_location=device)
model_vit.load_state_dict(checkpoint)
model_vit.to(device)
model_vit.eval()
print(f"Loaded model from {model_vit_path}")

model = CNNBinaryClassifier(model_type='resnet18', pretrained=False).to(device)

load_dotenv()

@app.get('/')
def home():
    return {'msg': 'Welcome in my api'}

@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh upload về PIL Image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Gọi hàm predict với img truyền vào thay vì image_path
        # Bạn cần chỉnh hàm predict nhận PIL Image hoặc tensor (nếu chưa, mình có thể giúp)
        result = predict(
            image=img,
            model_path=model_path,
            model=model,
            device=device
        )
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_gradcam")
async def predict_and_visualize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = predict_and_visualize_gradcam(
            image=img,
            model=model_vit,
            device=device
        )
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})