import torch
from ultralytics import YOLO

model = YOLO('./yolo11n.pt')

train_path = './datasets/train/images/'
val_path = './datasets/val/images/'

result = model.train(
    data='./data.yaml',
    epochs=10,
    batch=16,
    device='cpu'
)

model.save('./model2.pt')

print(result)
