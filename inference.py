from ultralytics import YOLO

# Load a pretrained YOLO11n model
model_path = "/DATA/jhlee_temp/pjw/ultralytics/runs/detect/train6/weights/best.pt"
model = YOLO(model_path)
source = "/DATA_17/DATASET/Competition_Dataset/CytologIA/images/test/"
# Run inference on 'bus.jpg' with arguments
model.predict(source=source, save_txt=True, imgsz=320, batch=4, conf=0.15, device=3, project="./run", name="predict")