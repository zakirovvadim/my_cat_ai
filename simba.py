from ultralytics import YOLO

model = YOLO("runs/detect/my_cat_v111/weights/best.pt")

results = model.predict(source="img_1.png", save=True, conf=0.5)