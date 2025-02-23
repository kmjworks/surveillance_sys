from ultralytics import YOLO
import torch 

MODEL_PATH = '../models/yolov11_trained.pt'
EXPORT_FORMAT = 'onnx'

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

model = YOLO(MODEL_PATH)

print(f"Exporting model to {EXPORT_FORMAT} format with input size {INPUT_HEIGHT}x{INPUT_WIDTH}...")

try:
    exported_path = model.export(
        format=EXPORT_FORMAT,
        imgsz=[INPUT_HEIGHT, INPUT_WIDTH]
    )
    print(f"Model exported successfully to: {exported_path}")

except Exception as e:
    print(f"Error during model exporting: {e}")