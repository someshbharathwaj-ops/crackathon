## importations ----------------------------
## importing the YOLO from the ultralytics
from ultralytics import YOLO
import os


## --------------------------
# DATA CONFIGURATIONS
##----------------------------

DATA_CONFIG="configs/rdd.yaml"
MODEL_TYPE="yolov8m.pt"
EPOCHS=50
IMAGE_SIZE=640
BATCH_SIZE=8
PROJECT_DIR=os.getcwd()
RUN_NAME="RDD_BASELINE"

##------------------------------------------
##RUNING
##--------------------------------------

def main():
    if not os.path.exists(DATA_CONFIG):
        print(f"{DATA_CONFIG} does not exist")
        print("CHECK YOUR FOLDER STRUCTURE BEFORE RUNNING")
        return

    print(f"INITIALIZING TRAINING : {RUN_NAME} using {MODEL_TYPE} model")

    ###----------------------------------------
    ##MODEL INITILAIZATION
    ##----------------------------------------
    model = YOLO(MODEL_TYPE)

model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    device=0,
    exist_ok=True,
    pretrained=True,
    cache="ram"
   # 🔥 THIS LINE IS MANDATORY
)

    print(f"✅ Training complete! Results saved to {PROJECT_DIR}/{RUN_NAME}")
 if __name__ == "__main__":
        main()
