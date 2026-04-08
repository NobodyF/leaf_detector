# 🍃 Leaves & Fruits YOLO Detector

A computer vision pipeline for detecting and counting **leaves** and **fruits** on potted plants using a custom-trained YOLO model. Built for field use — run inference on new plant photos, review and correct predictions visually, then fine-tune the model with new data.

---

## 📁 Project Structure

```
project/
├── dataset.yaml          # Dataset config — paths, class names
├── train.py              # Full training from a base YOLO model
├── train_adding.py       # Fine-tune an existing model with new images
├── predict.py            # Run inference, draw boxes, save annotated images
├── predict_and_save.py   # Run inference + save YOLO .txt label files for annotator
├── annotator.py          # Web-based annotation correction tool
├── results/              # Output folder — annotated plant images
└── yolo26n.pt / yolo26s.pt  # Base YOLO weights (starting point for training)
```

---

## 📄 File Descriptions

### `dataset.yaml`
Defines where the training data lives and what classes the model detects.

```yaml
path: D:/LeavesDatasetClassic
train: train/images
val: valid/images

names:
  0: fruits
  1: leaves
```

YOLO automatically finds label `.txt` files by replacing `images` with `labels` in the path. Edit this file if your dataset is in a different location.

---

### `train.py`
**Full training from scratch** using a base YOLO model (`yolo26s.pt`).

- Trains for 150 epochs on the full dataset
- Saves weights to `runs/detect/runs/leaves_fruits/weights/`
- Runs validation automatically after training

Use this when building the model for the first time or when the dataset has changed significantly.

```bash
python train.py
```

---

### `train_adding.py`
**Fine-tuning** — continues training from an already-trained `best.pt` instead of starting from zero.

- Loads your existing best model as the starting point
- Trains for ~20–40 epochs with a lower learning rate (`1e-4`)
- Freezes the first 10 backbone layers so core features aren't disrupted
- Saves results to a separate folder (`leaves_fruits_ft`) — original run is untouched

Use this when you have added new images to the dataset and want to update the model without retraining from scratch.

```bash
python train_adding.py
```

> ⚠️ Before running: drop new images into `train/images/` and their labels into `train/labels/`, then confirm the `PRETRAINED` path in the CONFIG block points to your current `best.pt`.

---

### `predict.py`
**Quick inference visualizer.** Runs the model on a folder of images and saves annotated copies with purple bounding boxes and a plant info panel (plant name + leaf count).

- Reads images from `SOURCE_FOLDER`
- Saves annotated images to `OUTPUT_FOLDER`
- Does **not** save `.txt` label files
- Configurable confidence, IoU, image size, max detections

```bash
python predict.py
```

Use this for a fast visual check of how the model performs on new photos.

---

### `predict_and_save.py`
**Inference + label export.** Same as `predict.py` but additionally saves YOLO-format `.txt` prediction files alongside the annotated images.

Outputs to two subfolders:
```
predictions/
├── annotated/    ← images with coloured boxes and class labels drawn on them
└── labels/       ← YOLO .txt files (one per image) ready for annotator.py
```

- Orange boxes = **fruits**, purple boxes = **leaves**
- Per-class counts shown in an info panel on each image
- The `labels/` folder feeds directly into `annotator.py` for correction

```bash
python predict_and_save.py
```

---

### `annotator.py`
**Web-based annotation editor.** Loads predicted labels from `predict_and_save.py` and lets you correct them in the browser — then saves corrected labels to a separate folder.

**Features:**
- 🔍 Zoom with mouse wheel, pan with right-click drag
- ↔️ Move boxes by dragging
- ↕️ Resize boxes by dragging corner handles
- ➕ Draw new boxes on empty canvas
- 🗑️ Delete wrong boxes (`Del` key or button)
- 🏷️ Toggle class labels on/off
- 💾 Save corrected labels (`Ctrl+S`) → writes to `corrected/labels/`
- ✓ Corrected images marked with a green badge in the sidebar

```bash
pip install flask
python annotator.py
# then open http://127.0.0.1:5000
```

Paths are configured at the top of the file:
```python
IMAGES_FOLDER    = ...   # your original images
LABELS_FOLDER    = ...   # predicted .txt files (read)
CORRECTED_FOLDER = ...   # corrected .txt files (write)
```

---

### `results/`
Output folder containing **annotated plant images** — one image per plant, with all detected leaves and fruits marked with coloured bounding boxes and a count panel. These are the final visual outputs of the pipeline.

---

## 🔄 Full Pipeline

```
New plant photos
      │
      ▼
predict_and_save.py   →   annotated images + .txt label files
      │
      ▼
annotator.py          →   corrected .txt label files
      │
      ▼
Add corrected images + labels to dataset  (train/ folder)
      │
      ▼
train_adding.py       →   updated best.pt model
```

---

## ⚙️ Requirements

```bash
pip install ultralytics flask
```

- Python 3.8+
- CUDA-capable GPU recommended for training
- YOLO base weights: `yolo26n.pt` or `yolo26s.pt`
