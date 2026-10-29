import argparse
import json
import cv2
from inspectai.classical import read_image, inspect

def main():
    parser = argparse.ArgumentParser(description="Compare PCB candidate to reference (CPU baseline)")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--no-registration", action="store_true")
    args = parser.parse_args()
    result = inspect(read_image(args.reference), read_image(args.candidate), register=not args.no_registration)
    if not cv2.imwrite(args.output, result.overlay):
        raise OSError("Failed writing overlay")
    print(json.dumps({"anomaly_score": result.anomaly_score, "boxes_xywh": result.boxes, "alignment_succeeded": result.alignment_succeeded, "warning": result.warning, "output": args.output}, indent=2))
if __name__ == "__main__":
    main()
