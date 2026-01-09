import os
from common.utils.datasets import TeamTrackDataset
from common.utils.utils import get_dir
import torch
from ultralytics import YOLO
import yaml

def train_yolo(model, yaml_path, epochs=3, batch_size=32, device="cpu"):
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        device=device,
    )


def create_yolo_dataset_yaml(output_dir):
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "person"
        }
    }
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return yaml_path

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    model = YOLO(get_dir("models/yolo11n.pt"))  
    yolo_dataset_dir = get_dir("dataset/yolo_format")
    
    if not os.path.exists(yolo_dataset_dir) or not os.listdir(yolo_dataset_dir):
        print("Dataset not found")
        exit(1)

    yaml_path = create_yolo_dataset_yaml(yolo_dataset_dir)    
    print("Starting YOLO training...")
    train_yolo(model, yaml_path, epochs=3, batch_size=32, device=device)
    model('components/detector/tests/nba.png', save=True, show_labels=False)
