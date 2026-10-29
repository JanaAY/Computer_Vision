"""Generate synthetic images for testing pipeline mechanics, not model quality."""
from pathlib import Path
import cv2
import numpy as np

folder = Path("demo")
folder.mkdir(exist_ok=True)
base = np.full((480, 640, 3), (35, 70, 35), dtype=np.uint8)
for row in range(5):
    for col in range(7):
        x, y = 35 + col * 82, 35 + row * 82
        cv2.rectangle(base, (x, y), (x+52, y+38), (175, 175, 175), -1)
        cv2.circle(base, (x+10, y+10), 4, (10, 10, 10), -1)
        cv2.line(base, (x+8, y+30), (x+44, y+30), (12, 12, 12), 2)
candidate = base.copy()
cv2.rectangle(candidate, (35 + 3*82, 35 + 2*82), (35+3*82+52, 35+2*82+38), (35, 70, 35), -1)
cv2.imwrite(str(folder / "reference.png"), base)
cv2.imwrite(str(folder / "candidate.png"), candidate)
print("Created demo/reference.png and demo/candidate.png")
