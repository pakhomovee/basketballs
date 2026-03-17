from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.utils import img2label_paths, exif_size
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import colorstr, RANK
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.data.augment import RandomFlip
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from court_detector.court_constants import SYMMETRIC_MAPPING


class CenterBboxLoss(nn.Module):
    """Replaces CIoU box loss with MSE on bbox centers; zeroes out DFL loss."""

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_c = (pred_bboxes[fg_mask][:, :2] + pred_bboxes[fg_mask][:, 2:]) / 2
        tgt_c = (target_bboxes[fg_mask][:, :2] + target_bboxes[fg_mask][:, 2:]) / 2
        loss_center = ((pred_c - tgt_c) ** 2).sum(-1, keepdim=True)
        loss_center = (loss_center * weight).sum() / target_scores_sum
        loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_center, loss_dfl


class CenterDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with CenterBboxLoss instead of the default BboxLoss."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.bbox_loss = CenterBboxLoss().to(self.device)


class CenterDetectionModel(DetectionModel):
    """DetectionModel that uses CenterDetectionLoss instead of default v8DetectionLoss."""

    def init_criterion(self):
        return E2ELoss(self, CenterDetectionLoss) if getattr(self, "end2end", False) else CenterDetectionLoss(self)


class CourtDetectionDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.flip_transform = RandomFlip(p=1)

    def get_labels(self) -> list[dict]:
        """
        Custom get_lables method to deal with custom label file format.
        """
        print("Starting loading images labels...")
        self.label_files = img2label_paths(self.im_files)
        assert len(self.label_files) == len(self.im_files)
        labels = []
        datasets_to_labels = {}
        for im_file, lb_file in tqdm(zip(self.im_files, self.label_files), total=len(self.label_files)):
            im = Image.open(im_file)
            im.verify()
            shape = exif_size(im)

            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                assert len(lb[-1]) == 1
                dataset_id = lb[-1][0]
                lb = lb[:-1]
                lb = np.array(lb, dtype=np.float32)

            assert lb.shape[1] == 5

            if dataset_id not in datasets_to_labels:
                datasets_to_labels[dataset_id] = []

            labels.append(
                {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    # "segments": [],
                    # "keypoints": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

            datasets_to_labels[dataset_id].append(len(labels) - 1)
        max_size = 0
        for dataset_id in datasets_to_labels:
            max_size = max(max_size, len(datasets_to_labels[dataset_id]))
        self.images_sequence = []
        for dataset_id in datasets_to_labels:
            cur_labels = datasets_to_labels[dataset_id]
            num_added = 0
            while num_added < max_size:
                np.random.shuffle(cur_labels)
                for label_id in cur_labels:
                    assert 0 <= label_id and label_id < len(labels)
                    self.images_sequence.append(label_id)
                    num_added += 1
                    if num_added >= max_size:
                        break
        np.random.shuffle(self.images_sequence)
        print("Labels loaded!")
        return labels

    def __len__(self):
        return len(self.images_sequence)

    def __getitem__(self, index):
        index = self.images_sequence[index]
        make_flip = not self.train_mode and np.random.choice([True, False])
        if make_flip:
            old_label = deepcopy(self.labels[index])
            for i in range(len(self.labels[index]["cls"])):
                self.labels[index]["cls"][i] = SYMMETRIC_MAPPING[self.labels[index]["cls"][i].item()]
        img_and_label = self.get_image_and_label(index)
        if make_flip:
            self.labels[index] = old_label
            img_and_label = self.flip_transform(img_and_label)
        return self.transforms(img_and_label)


def build_court_detection_dataset(
    cfg,
    img_path,
    batch,
    data,
    mode,
    rect,
    stride,
):
    return CourtDetectionDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


class CourtDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CenterDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_court_detection_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
