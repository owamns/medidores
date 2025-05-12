from ultralytics import YOLO
model = YOLO('yolo11n-obb.pt')

results = model.train(
    data='./analog_digit_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='0',
    cache=False
)