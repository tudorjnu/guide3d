import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision import io


def preprocess_tck(
    tck: Dict,
) -> List:
    t = tck["t"]
    c = tck["c"]
    k = tck["k"]

    t = np.array(t)
    c = [np.array(c_i) for c_i in c]
    k = int(k)

    return t, c, k


def process_data(
    data: Dict,
    seq_len: int = 3,
    cameras: str = "A",
) -> List:
    videos = []
    for video_pair in data:
        videoA = []
        videoB = []
        for frame in video_pair["frames"]:
            imageA = frame["cameraA"]["image"]
            imageB = frame["cameraB"]["image"]

            tckA = preprocess_tck(frame["cameraA"]["tck"])
            tckB = preprocess_tck(frame["cameraB"]["tck"])

            uA = np.array(frame["cameraA"]["u"])
            uB = np.array(frame["cameraB"]["u"])

            tck3d = preprocess_tck(frame["3d"]["tck"])
            u3d = np.array(frame["3d"]["u"])

            videoA.append(
                dict(
                    image=imageA,
                    tck=tckA,
                    u=uA,
                    tck3d=tck3d,
                    u3d=u3d,
                )
            )
            videoB.append(
                dict(
                    image=imageB,
                    tck=tckB,
                    u=uB,
                    tck3d=tck3d,
                    u3d=u3d,
                )
            )

        if "A" in cameras:
            videos.append(videoA)
        if "B" in cameras:
            videos.append(videoB)

    new_videos = []
    for video in videos:
        new_video = []
        for i in range(0, len(video) - seq_len + 1):
            new_video.append(video[i : i + seq_len])
        new_videos.append(new_video)

    return new_videos


def split_video_data(
    data: List,
    split: tuple = (0.8, 0.1, 0.1),
) -> List:
    train_data = []
    val_data = []
    test_data = []

    for video in data:
        train_idx = int(split[0] * len(video))
        val_idx = int(split[1] * len(video))
        train_data.extend(video[:train_idx])
        val_data.extend(video[train_idx : train_idx + val_idx])
        test_data.extend(video[train_idx + val_idx :])
    return train_data, val_data, test_data


class Guide3D(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "sphere.json",
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
        seg_len: int = 3,
        max_len: int = 150,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
    ):
        self.root = Path(root)
        self.annotations_file = annotations_file
        raw_data = json.load(open(self.root / self.annotations_file))
        data = process_data(raw_data)
        train_data, val_data, test_data = split_video_data(data, split_ratio)
        assert split in [
            "train",
            "val",
            "test",
        ], "Split should be one of 'train', 'val', 'test'"

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

        self.image_transform = image_transform
        self.pts_transform = pts_transform

        self.seg_len = seg_len
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]
        print(data_entry)
        exit()
        imgs_paths = [self.root / img_path for img_path in data_entry["image"]]
        imgs = [io.read_image(img_path.as_posix()) for img_path in imgs_paths]
        exit()

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


def main():
    import vars

    dataset = Guide3D(root=vars.dataset_path, split="train")
    print(len(dataset))
    # sample = dataset[0]


if __name__ == "__main__":
    main()
