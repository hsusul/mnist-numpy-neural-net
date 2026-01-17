from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import sys

# allow importing your src/ modules
sys.path.append("src")

from model import TwoLayerNet  # noqa: E402

MODEL_PATH = "artifacts/model_mnist.npz"

app = FastAPI(title="MNIST Draw Demo", version="1.0.0")

# serve static frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

model = None


@app.on_event("startup")
def load_model():
    global model
    model = TwoLayerNet.load(MODEL_PATH)


@app.get("/")
def home():
    return FileResponse("app/static/index.html")



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    """
    Expects JSON:
      { "pixels": [784 floats in [0,1]] }
    Returns:
      pred, probs[10], hidden[128]
    """
    if "pixels" not in payload:
        return JSONResponse({"error": "Missing 'pixels' field."}, status_code=400)

    pixels = payload["pixels"]
    if not isinstance(pixels, list) or len(pixels) != 784:
        return JSONResponse({"error": "pixels must be a list of length 784."}, status_code=400)

    try:
        x = np.array(pixels, dtype=np.float32).reshape(1, 784)
    except Exception:
        return JSONResponse({"error": "Could not parse pixels into float array."}, status_code=400)

    if not np.isfinite(x).all():
        return JSONResponse({"error": "pixels contains NaN/Inf."}, status_code=400)

    # forward with hidden activations
    z1 = model.dense1.forward(x)     # (1,128)
    a1 = model.relu.forward(z1)      # (1,128) ReLU activations
    logits = model.dense2.forward(a1)  # (1,10)

    # stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    pred = int(np.argmax(probs, axis=1)[0])

    return {
        "pred": pred,
        "probs": probs[0].astype(float).tolist(),
        "hidden": a1[0].astype(float).tolist(),
    }
