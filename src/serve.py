from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tải file model.joblib từ cloud storage về máy khi server khởi động.
    """
    # TODO 1 -> 4: Tạo client, lấy bucket/blob và tải về
    client = storage.Client()
    bucket = client.bucket(ARTIFACT_BUCKET)
    blob   = bucket.blob(MODEL_KEY)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    blob.download_to_filename(MODEL_PATH)
    print("Model da duoc tai xuong tu cloud storage.")


# Chỉ tải và nạp model nếu có cấu hình ARTIFACT_BUCKET (tránh lỗi khi chạy test cục bộ)
if os.environ.get("ARTIFACT_BUCKET"):
    download_model()
    model = joblib.load(MODEL_PATH)
else:
    model = None


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    # TODO 5: Trả về dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    # TODO 6: Kiểm tra phải có đúng 10 đặc trưng
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400, 
            detail="Expected 10 features (adult income)"
        )

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # TODO 7: Dự đoán
    pred = int(model.predict([req.features])[0])

    # TODO 8: Trả về kết quả dự đoán và nhãn
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
