from ultralytics import YOLO
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import yaml
import cv2
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision import transforms as T
import torch.optim as optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def load_ yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_image_and_annotations(idx, data):

    image_info = data['train'][idx]
    image_path = image_info['image']


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    annotations = image_info['annotations']
    boxes = []
    labels = []

    for ann in annotations:
        bbox = ann['bbox']
        label = ann['class']
        boxes.append(bbox)
        labels.append(label)


    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)


    target = {'boxes': boxes, 'labels': labels}
    return image, target

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform


    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        image, target = load_image_and_annotations(idx)
        if self.transform:
            image = self.transform(image)
        return image, target



def train_yolo(model_name, data_path):
    model = YOLO(model_name)
    model.train(
        data=data_path,
        epochs=100,
        batch=16,
        img_size=320,
        workers=0,
        optimizer='AdamW',
        cache=True,
        lr0=0.001
    )
    results = model.val()
    print(f"Results for {model_name}:")
    print(f"Precision: {results.metrics['precision']}")
    print(f"Recall: {results.metrics['recall']}")
    print(f"mAP@0.5: {results.metrics['map50']}")
    print(f"mAP@0.5:0.95: {results.metrics['map50_95']}")
    return results

train_yolo('yolov3.yaml', 'path_to_data.yaml')
train_yolo('yolov5s.yaml', 'path_to_data.yaml')
train_yolo('yolov8s.yaml', 'path_to_data.yaml')

def train_ssd300(data_path):

    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    num_classes = 3
    model.head.classification_head.num_classes = num_classes

    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
    ])

    train_dataset = CustomDataset(transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


    val_dataset = CustomDataset(transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            running_loss += losses.item()



    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])
    for images, targets in val_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        formatted_outputs = []
        for i in range(len(outputs)):
            formatted_output = {
                'boxes': outputs[i]['boxes'].cpu(),
                'scores': outputs[i]['scores'].cpu(),
                'labels': outputs[i]['labels'].cpu(),
            }
            formatted_outputs.append(formatted_output)

        metric.update(formatted_outputs, targets)

    eval_result = metric.compute()

    print(f"Precision: {eval_result['map']}")
    print(f"Recall: {eval_result['map_50']}")
    print(f"mAP@0.5: {eval_result['map_50']}")
    print(f"mAP@0.5:0.95: {eval_result['map_95']}")





def train_faster_rcnn(data_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
    ])

    train_dataset = CustomDataset(transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,
                                               collate_fn=lambda x: tuple(zip(*x)))


    val_dataset = CustomDataset(transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False,
                                             collate_fn=lambda x: tuple(zip(*x)))


    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.95])
    for images, targets in val_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)


        formatted_outputs = []
        for i in range(len(outputs)):
            formatted_output = {
                'boxes': outputs[i]['boxes'].cpu(),
                'scores': outputs[i]['scores'].cpu(),
                'labels': outputs[i]['labels'].cpu(),
            }
            formatted_outputs.append(formatted_output)

        metric.update(formatted_outputs, targets)

    eval_result = metric.compute()


    print(f"Precision: {eval_result['map']}")
    print(f"Recall: {eval_result['map_50']}")
    print(f"mAP@0.5: {eval_result['map_50']}")
    print(f"mAP@0.5:0.95: {eval_result['map_95']}")




train_ssd300('path_to_data.yaml')
train_faster_rcnn('path_to_data.yaml')




