from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import predict

app = FastAPI(
    title="NeuroSight AI",
    description="AI Medical Imaging System (Lung X-ray)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "NeuroSight AI"
    }

@app.get("/test")
def test():
    return {"message": "API working"}