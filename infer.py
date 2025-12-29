from ultralytics import YOLO
import os
from pathlib import Path

MODEL_PATH = "/kaggle/working/crackathon/RDD_BASELINE/weights/best.pt"
TEST_IMAGES_DIR = "/kaggle/input/crackathon/test/images"
OUTPUT_DIR = "/kaggle/working/crackathon/submissions"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    print(f"🔍 Found {len(image_files)} images for inference.")

    model.predict(
        source=TEST_IMAGES_DIR,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        save=False,
        save_txt=True,
        save_conf=True,   # ✅ REQUIRED
        project=OUTPUT_DIR,
        name="labels",
        exist_ok=True
    )

    print(f"✅ Inference complete. Predictions saved to {OUTPUT_DIR}/labels")


if __name__ == "__main__":
    main()
