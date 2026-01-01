from ultralytics import YOLO
import os

# =========================
# CONFIGURATION — V3
# =========================
DATA_CONFIG = "configs/rdd.yaml"
MODEL_TYPE = "yolov8l.pt"

EPOCHS = 150
IMAGE_SIZE = 1024
BATCH_SIZE = 4   # lower batch for high-res
PROJECT_DIR = os.getcwd()
RUN_NAME = "RDD_V3_HIGHRES"

# =========================
# TRAINING
# =========================
def main():
    if not os.path.exists(DATA_CONFIG):
        raise FileNotFoundError("Dataset config not found")

    print(f"🚀 V3 TRAINING: {RUN_NAME}")

    model = YOLO(MODEL_TYPE)

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,

        device=0,
        pretrained=True,
        exist_ok=True,

        # 🔥 V3 TUNING
        lr0=0.005,
        lrf=0.01,
        patience=50,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,

        fliplr=0.5,
        scale=0.5,
        translate=0.1,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        workers=8,
        verbose=True
    )

    print(f"✅ V3 Training finished: {PROJECT_DIR}/{RUN_NAME}")

if __name__ == "__main__":
    main()
