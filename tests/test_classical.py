import numpy as np
import cv2
from inspectai.classical import inspect

def test_identical_images_have_zero_score():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    out = inspect(image, image, register=False)
    assert out.anomaly_score == 0
    assert out.boxes == []

def test_visible_difference_detected():
    a = np.full((150, 150, 3), 90, dtype=np.uint8)
    b = a.copy()
    cv2.rectangle(b, (40, 40), (90, 90), (255, 255, 255), -1)
    out = inspect(a, b, register=False)
    assert out.anomaly_score > 0
    assert len(out.boxes) >= 1

def test_invalid_image_raises():
    import pytest
    with pytest.raises(ValueError):
        inspect(np.zeros((4, 4)), np.zeros((4, 4)))
