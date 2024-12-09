from ultralytics import YOLO


# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")



results = model.train(data="config.yaml",batch=16, epochs=1, imgsz=640,device="0",workers=0)