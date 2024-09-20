from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

model.train(
    data='path_to_data.yaml',
    epochs=100,
    batch=32,
    img_size=512,
    workers=0,
    optimizer='AdamW'
)

model.save('path_to_trained_model.pt')

