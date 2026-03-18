from fastapi import APIRouter, UploadFile
from services.inference import run_inference

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile):
    return await run_inference(file)