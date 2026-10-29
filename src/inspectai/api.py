from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import cv2
from inspectai.classical import inspect

app = FastAPI(title="InspectAI CPU Baseline", version="0.1.0")
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000

async def decode_image(file: UploadFile) -> np.ndarray:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 12MB limit")
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    if raw.shape[0] * raw.shape[1] > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Too many image pixels")
    return raw

@app.get("/health")
def health():
    return {"status": "ok", "model": "classical-cv-v0"}

@app.post("/inspect")
async def inspect_images(reference: UploadFile = File(...), candidate: UploadFile = File(...)):
    ref = await decode_image(reference)
    cand = await decode_image(candidate)
    try:
        result = inspect(ref, cand)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"anomaly_score": result.anomaly_score, "score_definition": "fraction_of_nonzero_difference_mask_pixels_not_probability", "boxes_xywh": result.boxes, "alignment_succeeded": result.alignment_succeeded, "warning": result.warning, "requires_review": bool(result.boxes) or not result.alignment_succeeded}
