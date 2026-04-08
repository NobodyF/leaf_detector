from pathlib import Path
from ultralytics import YOLO

def main():
    model = YOLO("yolo26s.pt")
    data_yaml = Path(__file__).with_name("dataset.yaml")

    results = model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=150,
        batch=16,
        device=0,
        workers=0,
        cache=False,
        project="runs",
        name="leaves_fruits",
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    YOLO(str(best_model)).val(data=str(data_yaml))

if __name__ == "__main__":
    main()