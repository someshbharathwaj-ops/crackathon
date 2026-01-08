# ============================================================
#  RDD COMPETITION TRAINING SCRIPT — FINAL VERSION
# ============================================================

from ultralytics import YOLO
import os

# -----------------------------
# PATHS
# -----------------------------
DATA_CONFIG = "configs/rdd.yaml"   # dataset yaml
PROJECT_DIR = os.getcwd()

# -----------------------------
# MODEL SETTINGS
# -----------------------------
MODEL_TYPE = "yolov8l.pt"          # use large model for best mAP
RUN_NAME = "RDD_V3_CHAMPION"

# -----------------------------
# TRAINING HYPERPARAMETERS
# -----------------------------
EPOCHS = 150
IMG_SIZE = 1024                   # high-res for crack detection
BATCH_SIZE = 4                    # safe for 16GB GPU
DEVICE = 0                        # GPU

# -----------------------------
# ADVANCED AUGMENTATION
# -----------------------------
AUGMENT = dict(
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.1,
    erasing=0.4,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

# -----------------------------
# OPTIMIZATION
# -----------------------------
OPTIMIZER = "AdamW"
LR0 = 0.005
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
PATIENCE = 50                     # early stopping
WORKERS = 8

# ============================================================
def main():

    # -----------------------------
    # SAFETY CHECK
    # -----------------------------
    if not os.path.exists(DATA_CONFIG):
        raise FileNotFoundError(f"❌ Dataset config not found: {DATA_CONFIG}")

    print(f"\n🚀 FINAL TRAINING RUN: {RUN_NAME}")
    print(f"📦 Model: {MODEL_TYPE}")
    print(f"🖼 Image size: {IMG_SIZE}")
    print(f"🔁 Epochs: {EPOCHS}\n")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = YOLO(MODEL_TYPE)

    # -----------------------------
    # TRAIN
    # -----------------------------
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        optimizer=OPTIMIZER,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,

        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        workers=WORKERS,

        pretrained=True,
        cache=False,

        # -------- AUGMENTATION --------
        mosaic=AUGMENT["mosaic"],
        mixup=AUGMENT["mixup"],
        copy_paste=AUGMENT["copy_paste"],
        erasing=AUGMENT["erasing"],
        fliplr=AUGMENT["fliplr"],
        hsv_h=AUGMENT["hsv_h"],
        hsv_s=AUGMENT["hsv_s"],
        hsv_v=AUGMENT["hsv_v"],

        # -------- STABILITY --------
        amp=True,
        deterministic=True,
        seed=42,
        plots=True,
    )

    print("\n🏁 TRAINING COMPLETE!")
    print(f"📁 Results: {PROJECT_DIR}/{RUN_NAME}")
    print(f"🏆 Best weights: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


# ============================================================
if __name__ == "__main__":
    main()
