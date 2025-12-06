from typing import List
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from api_models import Applicant, PredictionResponse, BatchPredictionResponse


MODEL_PATH = "./models/german_credit_rf.joblib"
model_obj = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once on startup, and once on shutdown.
    Replaces deprecated @app.on_event("startup").
    """
    global model_obj
    print("Loading model...")

    def load_model_sync():
        return joblib.load(MODEL_PATH)

    # Load model in threadpool (non-blocking)
    model_obj = await run_in_threadpool(load_model_sync)
    print("Model loaded successfully.")

    yield  # App runs here

    # Optional: cleanup on shutdown
    print("Shutting down...")


# FastAPI app with lifespan
app = FastAPI(
    title="German Credit Scoring API",
    version="1.0.0",
    lifespan=lifespan
)


def get_model():
    if model_obj is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return model_obj["model"], model_obj["metadata"]


# ==========================================================
# Sync prediction helpers (run inside threadpool)
# ==========================================================

def _predict_single_sync(applicant: Applicant, threshold: float = 0.5) -> PredictionResponse:
    model, metadata = get_model()

    feature_order = metadata["feature_order"]
    df = pd.DataFrame(
        [[applicant.model_dump()[col] for col in feature_order]],
        columns=feature_order
    )

    proba_good = float(model.predict_proba(df)[:, 1][0])
    label = int(proba_good >= threshold)

    return PredictionResponse(
        predicted_label=label,
        proba_good=proba_good,
        threshold=threshold,
    )


def _predict_batch_sync(applicants: List[Applicant], threshold: float = 0.5) -> BatchPredictionResponse:
    model, metadata = get_model()

    feature_order = metadata["feature_order"]
    df = pd.DataFrame([a.model_dump() for a in applicants])[feature_order]

    probs = model.predict_proba(df)[:, 1]
    labels = (probs >= threshold).astype(int)

    preds = [
        PredictionResponse(
            predicted_label=int(lbl),
            proba_good=float(p),
            threshold=threshold,
        )
        for lbl, p in zip(labels, probs)
    ]

    return BatchPredictionResponse(predictions=preds)


# ==========================================================
# Async Endpoints (CPU-heavy work in threadpool)
# ==========================================================

@app.get("/health")
def health_check():
    try:
        get_model()
        return {"status": "ok"}
    except:
        return {"status": "model_not_loaded"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(applicant: Applicant, threshold: float = 0.5):
    return await run_in_threadpool(_predict_single_sync, applicant, threshold)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(applicants: List[Applicant], threshold: float = 0.5):
    return await run_in_threadpool(_predict_batch_sync, applicants, threshold)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)