from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import configparser
import torch

class TeamTrackDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_files = []
        self.annotations = []
        self.video_metadata = []
        self.items = []

        files = os.listdir(os.path.join(self.root_dir))
        for file in files:
            video_path = os.path.join(self.root_dir, file, "img1.mp4")
            self.video_files.append(video_path)

            file_annotation = os.path.join(self.root_dir, file, "gt/gt.txt")
            df = pd.read_csv(file_annotation, sep=",", header=None)
            df.columns = ["frame", "obj_id", "x", "y", "width", "height", "1_", "2_", "3_", "4_"]

            file_metadata = os.path.join(self.root_dir, file, "seqinfo.ini")
            config = configparser.ConfigParser()
            config.read(file_metadata)
            metadata = {k: v for k, v in config['Sequence'].items()}
            self.video_metadata.append(metadata)

            seq_length = int(metadata["seqlength"])
            meta_idx = len(self.video_metadata) - 1
            grouped = df.groupby('frame')
            for frame_id in range(1, seq_length + 1):
                if frame_id in grouped.groups:
                    ann = grouped.get_group(frame_id)
                else:
                    ann = pd.DataFrame(columns=df.columns)
                self.annotations.append(ann)
                self.items.append((video_path, frame_id - 1, len(self.annotations) - 1, meta_idx))


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_path, frame_idx, ann_idx, meta_idx = self.items[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

        ann_df = self.annotations[ann_idx]
        boxes = ann_df[['x', 'y', 'width', 'height']].values.astype("float32")
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame, target, self.video_metadata[meta_idx]
