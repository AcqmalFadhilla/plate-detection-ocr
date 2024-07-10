from ultralytics import YOLO

model = YOLO("yolov9n.yaml")
results = model.train(data="percobaan-1/data.yaml", epochs=100, imgsz=640)