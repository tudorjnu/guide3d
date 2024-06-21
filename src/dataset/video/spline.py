from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision import io

import annot_parser


def process_data(
    data: List[List[Dict[str, Union[str, np.ndarray]]]],
) -> List[List[Dict[str, Union[str, np.ndarray]]]]:
    videos = []
    for video_pair in data:
        video1 = []
        video2 = []
        for frame in video_pair:
            video1.append(
                dict(
                    img=frame["img1"]["path"],
                    pts=frame["img1"]["points"],
                    reconstruction=frame["reconstruction"],
                )
            )
            video2.append(
                dict(
                    img=frame["img2"]["path"],
                    pts=frame["img2"]["points"],
                    reconstruction=frame["reconstruction"],
                )
            )
        videos.append(video1)
        videos.append(video2)

    return videos


class Guide3D(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations.xml",
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
        seg_len: int = 3,
        max_len: int = 150,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
    ):
        self.root = Path(root)
        self.annotations_file = annotations_file
        self.data = annot_parser.parse_to_video_separated(
            self.root / self.annotations_file, seg_len
        )
        self.data = process_data(self.data)

        assert split in [
            "train",
            "val",
            "test",
        ], "Split should be one of 'train', 'val', 'test'"

        if split == "train":
            self.data = self.data[: int(split_ratio[0] * len(self.data))]
        elif split == "val":
            self.data = self.data[
                int(split_ratio[0] * len(self.data)) : int(
                    split_ratio[0] * len(self.data) + split_ratio[1] * len(self.data)
                )
            ]
        elif split == "test":
            self.data = self.data[
                int(split_ratio[0] * len(self.data))
                + int(split_ratio[1] * len(self.data)) :
            ]

        self.image_transform = image_transform
        self.pts_transform = pts_transform

        self.seg_len = seg_len
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]
        imgs_paths = [self.root / img_path for img_path in data_entry["imgs1_paths"]]
        imgs = [io.read_image(img_path.as_posix()) for img_path in imgs_paths]

        if self.image_transform:
            imgs = [self.image_transform(img) for img in imgs]

        imgs = torch.stack(imgs, dim=0)

        pts = data_entry["pts_reconstructed"]
        if self.pts_transform:
            pts = [self.pts_transform(pt) for pt in pts]

        lengths = [torch.tensor(len(points), dtype=torch.int32) for points in pts]
        lengths = torch.stack(lengths, dim=0)

        pts = [torch.tensor(pt, dtype=torch.float32) for pt in pts]
        pts = [torch.cat((pt, torch.zeros((self.max_len - len(pt), 3)))) for pt in pts]
        pts = torch.stack(pts, dim=0)

        return imgs, pts, lengths
