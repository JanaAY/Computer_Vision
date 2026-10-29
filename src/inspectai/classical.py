"""An intentionally simple, CPU-only classical-CV starter, not production inspection."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class InspectionResult:
    anomaly_score: float
    boxes: list[tuple[int, int, int, int]]
    alignment_succeeded: bool
    warning: str | None
    overlay: np.ndarray
    mask: np.ndarray


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return candidate warped into reference coordinates, else fallback resized image."""
    h, w = reference.shape[:2]
    candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_AREA)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1200)
    kp1, des1 = orb.detectAndCompute(cand_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return candidate, False
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    matches = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(matches) < 8:
        return candidate, False
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inliers is None or int(inliers.sum()) < 6:
        return candidate, False
    warped = cv2.warpPerspective(candidate, H, (w, h))
    return warped, True


def inspect(reference: np.ndarray, candidate: np.ndarray, *, diff_threshold: int = 35, min_area: int = 50, register: bool = True) -> InspectionResult:
    if reference is None or candidate is None or reference.ndim != 3 or candidate.ndim != 3 or reference.shape[2] != 3 or candidate.shape[2] != 3:
        raise ValueError("Reference and candidate must be BGR color images")
    if not 0 <= diff_threshold <= 255 or min_area < 1:
        raise ValueError("Invalid inspection parameters")
    h, w = reference.shape[:2]
    if h * w > 16_000_000:
        raise ValueError("Image exceeds 16 megapixels")
    if register:
        cand, did_align = align(reference, candidate)
        warning = None if did_align else "Alignment failed; comparison used resized unregistered image. Manual review required."
    else:
        cand = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_AREA)
        did_align = False
        warning = "Registration disabled; results assume images already aligned."
    r = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    c = cv2.GaussianBlur(cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    delta = cv2.absdiff(r, c)
    _, mask = cv2.threshold(delta, diff_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_area], key=lambda b: (b[1], b[0]))
    overlay = cand.copy()
    for x, y, bw, bh in boxes:
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    score = float(np.count_nonzero(mask)) / mask.size
    return InspectionResult(score, boxes, did_align, warning, overlay, mask)
