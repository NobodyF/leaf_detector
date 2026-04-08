import re
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH    = Path(r"runs\detect\runs\leaves_fruits10\weights\best.pt")
SOURCE_FOLDER = Path(r"D:\LeavesDatasetClassic\new_images")
OUTPUT_FOLDER = Path(r"D:\LeavesDatasetClassic\predicted_output")

CONF     = 0.05 # lower = more detections
IOU      = 0.80
IMG_SIZE = 1536
MAX_DET  = 2500   # default was 300, raised to avoid capping

BOX_COLOR = (180, 0, 220)  # purple (BGR)
BOX_THICK = 3
# ─────────────────────────────────────────────────────────────────────────────


def extract_plant_name(img_path: Path) -> str:
    return img_path.stem.replace('_', ' ')  # '02_Plant6_Top_25062025' → '02 Plant6 Top 25062025'


def draw_info_panel(img: np.ndarray, plant_name: str, leaf_count: int) -> np.ndarray:
    """Draw a single-line info panel in the top-left corner."""
    h, w = img.shape[:2]

    scale      = w / 800
    font       = cv2.FONT_HERSHEY_DUPLEX
    text_scale = 0.65 * scale
    thickness  = max(1, int(scale))
    pad        = int(18 * scale)
    margin     = int(20 * scale)

    text = f"{plant_name}   |   {leaf_count} leaves"

    (tw, th), _ = cv2.getTextSize(text, font, text_scale, thickness)

    x1, y1 = margin, margin
    x2, y2 = x1 + tw + pad * 2, y1 + th + pad * 2

    # Semi-transparent dark purple background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 10, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Purple border
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thickness + 1, cv2.LINE_AA)

    # White text
    cv2.putText(img, text, (x1 + pad, y1 + pad + th),
                font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def process_image(img_path: Path, results) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    boxes = results.boxes

    # Draw purple boxes only — no labels
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK, cv2.LINE_AA)

    plant_name = extract_plant_name(img_path)
    draw_info_panel(img, plant_name, len(boxes))

    return img


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SOURCE_FOLDER.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        p for p in SOURCE_FOLDER.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        print("No images found in source folder.")
        return

    print(f"Processing {len(image_paths)} images...\n")

    for img_path in image_paths:
        result_list = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,  # ← no more 300 cap
            save=False,
            verbose=False,
        )

        annotated = process_image(img_path, result_list[0])

        out_path = OUTPUT_FOLDER / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  ✓  {img_path.name}  →  {len(result_list[0].boxes)} leaves detected")

    print(f"\nDone. {len(image_paths)} images saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()