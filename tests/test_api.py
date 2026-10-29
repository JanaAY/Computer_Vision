import cv2
import numpy as np
from fastapi.testclient import TestClient
from inspectai.api import app

client = TestClient(app)

def as_png(a):
    ok, buf = cv2.imencode(".png", a)
    assert ok
    return buf.tobytes()

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_inspect_upload():
    im = np.zeros((100, 100, 3), np.uint8)
    res = client.post("/inspect", files={"reference": ("ref.png", as_png(im), "image/png"), "candidate": ("cand.png", as_png(im), "image/png")})
    assert res.status_code == 200
    assert res.json()["anomaly_score"] == 0.0
    assert res.json()["alignment_succeeded"] is False
    assert res.json()["requires_review"] is True

def test_invalid_upload():
    res = client.post("/inspect", files={"reference": ("ref.png", b"not-image", "image/png"), "candidate": ("cand.png", b"not-image", "image/png")})
    assert res.status_code == 400
