# infer.py - RDD2022 Submission Generator
from ultralytics import YOLO
import os
from pathlib import Path

# ==========================================
#  CONFIGURATION (The "Settings" Object)
# ==========================================
MODEL_PATH = "/kaggle/working/crackathon/RDD_BASELINE/weights/best.pt"
TEST_IMAGES_DIR = "/kaggle/input/crackathon/test/images"
OUTPUT_DIR = "/kaggle/working/crackathon/submissions"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640


def main():
    # 1. Initialize System
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model weights not found at {MODEL_PATH}")
        return

    # 2. Load Model (Java: Model model = new Model(weights))
    model = YOLO(MODEL_PATH)

    # 3. Create Output Directory (Java: new File(dir).mkdirs())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Get List of Test Images
    image_files = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    print(f"🔍 Found {len(image_files)} images for inference.")

    # 5. Prediction Loop (The "Batch Processor")
    # stream=True handles memory efficiently for large datasets
    results = model.predict(
        source=TEST_IMAGES_DIR,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        save=False,  # We don't need to save visual JPEGs for submission
        save_txt=True,  # YOLO generates the .txt files for us!
        project=OUTPUT_DIR,
        name="labels",
        exist_ok=True
    )

    print(f" Inference complete. Predictions saved to: {OUTPUT_DIR}/labels")
    print(" Next Step: Verify these .txt files match the competition format.")


if __name__ == "__main__":
    main()
