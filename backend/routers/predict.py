from fastapi import APIRouter, UploadFile
from services.inference import run_inference

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile):
    result = await run_inference(file)
    return result