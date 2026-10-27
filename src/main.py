from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .predict import FraudPredictor

app = FastAPI(
    title="U-SAFE Fraud Intelligence API",
    description="Local demonstration API for the IEEE-CIS fraud detection project.",
    version="1.0.0",
)

# تحميل الموديل عبر predict.py الأصلي
predictor = FraudPredictor()

@app.post("/api/predict")
def predict_fraud(user_input: dict):
    # تشغيل نفس اللوجيك بدون تعديل
    return predictor.predict(user_input)

# ربط مجلد frontend لفتح واجهة HTML مباشرة عند تشغيل السيرفر
PROJECT_ROOT = Path(__file__).resolve().parents[1]
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
