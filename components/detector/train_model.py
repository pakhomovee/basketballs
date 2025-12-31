from common.utils.datasets import TeamTrackDataset
from torch.utils.data import DataLoader
from common.utils.utils import get_dir
import torch
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def train_model(model, device, train_loader, val_loader, epochs, learning_rate):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        model.train()
        train_loss, val_loss = 0, 0
        for i, (images, targets, _) in tqdm(enumerate(train_loader)):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss += loss_value

            losses.backward()
            optimizer.step()
        for i, (images, targets, _) in tqdm(enumerate(val_loader)):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        print(f"Epoch {epoch} completed, Train Loss: {train_loss}, Val Loss: {val_loss}")

def collate_fn(batch):
    return tuple(zip(*batch))

def finetune_model(model, device, epochs=10, batch_size=2, learning_rate=0.005):
    dataset1 = TeamTrackDataset(root_dir=get_dir("dataset/teamtrack-mot/teamtrack-mot/basketball_side/train"))
    dataset2 = TeamTrackDataset(root_dir=get_dir("dataset/teamtrack-mot/teamtrack-mot/basketball_side/val"))
    train_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_classes = 2 (background + person)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    for param in model.backbone.parameters():
        param.requires_grad = False

    train_model(model, device,train_loader, val_loader, epochs, learning_rate)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device)
    model.train()
    finetune_model(model, device, epochs=10, batch_size=2, learning_rate=0.005)
