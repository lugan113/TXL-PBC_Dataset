import os
import cv2
from ultralytics import YOLO


def apply_gaussian_smoothing(image_path, ksize=(5, 5)):
    image = cv2.imread(image_path)
    smoothed_image = cv2.GaussianBlur(image, ksize, 0)
    return smoothed_image


def automated_annotation(model, image_dir, output_dir, conf_thresh=0.5, iou_thresh=0.4, smoothing=False):
    images = os.listdir(image_dir)

    for image_file in images:
        image_path = os.path.join(image_dir, image_file)

        if smoothing:
            image = apply_gaussian_smoothing(image_path)
            smoothed_path = os.path.join(output_dir, f"smoothed_{image_file}")
            cv2.imwrite(smoothed_path, image)
            image_path = smoothed_path

        results = model(image_path, conf=conf_thresh)
        results.nms(iou=iou_thresh)
        results.save(save_dir=output_dir)


model = YOLO('path_to_trained_model.pt')

image_dir = 'path_to_unannotated_images'
output_dir = 'path_to_save_annotated_images'
os.makedirs(output_dir, exist_ok=True)

automated_annotation(model, image_dir, output_dir, conf_thresh=0.5, iou_thresh=0.4, smoothing=True)

