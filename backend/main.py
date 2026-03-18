from fastapi import FastAPI, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Medical AI API running"}

@app.post("/predict")
async def predict(file: UploadFile):

    data = await file.read()

    img = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    # dummy result ก่อน (ยังไม่ใช้ AI)
    return {
        "prediction": "PNEUMONIA",
        "confidence": 0.85
    }