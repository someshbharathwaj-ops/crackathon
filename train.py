# train.py — RDD Improved Training Script (v2)

from ultralytics import YOLO
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Dataset config
DATA_CONFIG = "configs/rdd.yaml"

# Model choice (upgrade later to yolov8l.pt if GPU allows)
MODEL_TYPE = "yolov8m.pt"

# Training parameters
EPOCHS = 150              # ⬆️ More training
IMAGE_SIZE = 768          # ⬆️ Better small-crack detection
BATCH_SIZE = 4            # Reduced for higher resolution

# Output
PROJECT_DIR = os.getcwd()
RUN_NAME = "RDD_V2_IMPROVED"

# Device
DEVICE = 0  # Kaggle GPU = 0, CPU = "cpu"

# ============================================================
# TRAINING
# ============================================================

def main():

    if not os.path.exists(DATA_CONFIG):
        print(f"❌ Dataset config not found: {DATA_CONFIG}")
        return

    print(f"\n🚀 INITIALIZING TRAINING: {RUN_NAME}")
    print(f"📦 Model: {MODEL_TYPE}")
    print(f"🖼️ Image size: {IMAGE_SIZE}")
    print(f"⏱️ Epochs: {EPOCHS}\n")

    # Load model
    model = YOLO(MODEL_TYPE)

    # Train
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Output
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # Hardware
        device=DEVICE,
        workers=8,
        pretrained=True,

        # 🔥 OPTIMIZATION FOR CRACKS 🔥
        cls=1.5,                 # Emphasize class learning
        box=7.5,
        dfl=1.5,

        # 🔁 DATA AUGMENTATION
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        # Training stability
        patience=50,
        deterministic=True,
        seed=0,

        # Logging
        verbose=True,
        plots=True
    )

    print(f"\n✅ TRAINING COMPLETE")
    print(f"📁 Results saved to: {PROJECT_DIR}/{RUN_NAME}")


if __name__ == "__main__":
    main()
