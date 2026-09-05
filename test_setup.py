import torch
from ultralytics import YOLO
import cv2
import numpy as np

print("Testing setup...")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")
model = YOLO('yolov8m.pt')
print("SUCCESS!")
