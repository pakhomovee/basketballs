import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

COCO_PERSON_ID = 1
CONF_THRESHOLD = 0.5

def detect_persons_torch(frame, conf_threshold=CONF_THRESHOLD):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    persons = []
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()
    for i in range(len(labels)):
        if labels[i] == COCO_PERSON_ID and scores[i] > conf_threshold:
            x1, y1, x2, y2 = map(int, boxes[i])
            persons.append((x1, y1, x2, y2, scores[i]))
    return persons
