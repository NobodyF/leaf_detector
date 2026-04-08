from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH    = Path(r"runs\detect\runs\leaves_fruits10\weights\best.pt")
SOURCE_FOLDER = Path(r"D:\LeavesDatasetClassic\test")
OUTPUT_FOLDER = Path(r"D:\LeavesDatasetClassic\predictions")

CONF     = 0.10
IOU      = 0.80
IMG_SIZE = 1536
MAX_DET  = 2500

CLASS_COLORS = {
    0: (0, 165, 255),   # fruits  → orange
    1: (180, 0, 220),   # leaves  → purple
}
DEFAULT_COLOR = (0, 255, 0)
BOX_THICK = 3
# ─────────────────────────────────────────────────────────────────────────────


def draw_boxes_with_labels(img: np.ndarray, boxes, class_names: dict) -> np.ndarray:
    """Draw coloured boxes with class name + confidence on the image."""
    out = img.copy()
    h, w = out.shape[:2]

    scale       = w / 1000
    font        = cv2.FONT_HERSHEY_DUPLEX
    text_scale  = max(0.4, 0.55 * scale)
    thickness   = max(1, int(scale))

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])

        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICK, cv2.LINE_AA)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, font, text_scale, thickness)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(out,
                      (x1, label_y - th - baseline - 4),
                      (x1 + tw + 6, label_y + baseline - 2),
                      color, -1)
        cv2.putText(out, label,
                    (x1 + 3, label_y - baseline),
                    font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def draw_info_panel(img: np.ndarray, plant_name: str, counts: dict, class_names: dict) -> np.ndarray:
    """Top-left panel showing plant name and per-class counts."""
    h, w = img.shape[:2]
    scale      = w / 800
    font       = cv2.FONT_HERSHEY_DUPLEX
    text_scale = 0.65 * scale
    thickness  = max(1, int(scale))
    pad        = int(18 * scale)
    margin     = int(20 * scale)

    lines = [plant_name] + [
        f"  {class_names.get(cid, str(cid))}: {cnt}"
        for cid, cnt in sorted(counts.items())
    ]

    sizes = [cv2.getTextSize(l, font, text_scale, thickness)[0] for l in lines]
    panel_w = max(s[0] for s in sizes) + pad * 2
    line_h  = max(s[1] for s in sizes) + pad
    panel_h = line_h * len(lines) + pad

    x1, y1 = margin, margin
    x2, y2 = x1 + panel_w, y1 + panel_h

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 10, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 0, 220), thickness + 1, cv2.LINE_AA)

    for i, (line, (tw, th)) in enumerate(zip(lines, sizes)):
        cy = y1 + pad + i * line_h + th
        cv2.putText(img, line, (x1 + pad, cy),
                    font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def save_yolo_labels(txt_path: Path, boxes, img_w: int, img_h: int):
    """Save predictions as YOLO .txt  (cls cx cy w h  — normalised 0-1)."""
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(lines))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SOURCE_FOLDER.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_FOLDER}")

    annotated_dir = OUTPUT_FOLDER / "annotated"
    labels_dir    = OUTPUT_FOLDER / "labels"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    class_names = model.names  # {0: 'fruits', 1: 'leaves'}

    image_paths = sorted(
        p for p in SOURCE_FOLDER.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        print("No images found in source folder.")
        return

    print(f"Processing {len(image_paths)} images...\n")

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            save=False,
            verbose=False,
        )

        r = results[0]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Could not read {img_path.name}")
            continue

        h, w = img.shape[:2]
        boxes = r.boxes if r.boxes is not None else []

        # Count per class
        counts = {}
        for box in boxes:
            cid = int(box.cls[0])
            counts[cid] = counts.get(cid, 0) + 1

        # Draw annotated image
        annotated = draw_boxes_with_labels(img, boxes, class_names)
        plant_name = img_path.stem.replace("_", " ")
        annotated = draw_info_panel(annotated, plant_name, counts, class_names)

        cv2.imwrite(str(annotated_dir / img_path.name), annotated)

        # Save YOLO labels
        save_yolo_labels(labels_dir / (img_path.stem + ".txt"), boxes, w, h)

        count_str = "  |  ".join(
            f"{class_names.get(cid, cid)}: {cnt}" for cid, cnt in sorted(counts.items())
        )
        print(f"  ✓  {img_path.name:<40}  {count_str}")

    print(f"\nDone.")
    print(f"  Annotated images → {annotated_dir}")
    print(f"  YOLO label files → {labels_dir}")
    print(f"\n  Point annotator.py at these folders to review & fix.")


if __name__ == "__main__":
    main()
