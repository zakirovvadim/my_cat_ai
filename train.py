from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cuda:0",
        name="my_cat_v1",
        optimizer="AdamW",
        lr0=0.0005,
        augment=True
    )
