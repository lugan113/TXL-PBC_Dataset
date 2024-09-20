from ultralytics import YOLO

model = YOLO('path_to_trained_model.pt')

model.train(
    data='path_to_data.yaml',
    epochs=100,
    batch=16,
    img_size=320,
    workers=0,
    optimizer='AdamW',
    cache=True,
    lr0=0.001
)

results = model.val()

precision = results.metrics['precision']
recall = results.metrics['recall']
map50 = results.metrics['map50']
map50_95 = results.metrics['map50_95']

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"mAP@0.5: {map50}")
print(f"mAP@0.5:0.95: {map50_95}")
