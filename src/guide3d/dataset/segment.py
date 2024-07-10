import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from guide3d.representations import curve
from guide3d.utils.utils import preprocess_tck
from torch.utils import data
from torchvision import transforms

image_transforms = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]
)


def process_data(
    data: Dict,
) -> List:
    video_pairs = []
    for video_pair in data:
        videoA = []
        videoB = []
        for frame in video_pair["frames"]:
            imageA = frame["cameraA"]["image"]
            imageB = frame["cameraB"]["image"]

            tckA = preprocess_tck(frame["cameraA"]["tck"])
            tckB = preprocess_tck(frame["cameraB"]["tck"])

            uA = np.array(frame["cameraA"]["u"]).astype(np.float32)
            uB = np.array(frame["cameraB"]["u"]).astype(np.float32)

            videoA.append(
                dict(
                    image=imageA,
                    mask=make_mask(tckA, uA),
                )
            )
            videoB.append(
                dict(
                    image=imageB,
                    mask=make_mask(tckB, uB),
                )
            )
        video_pairs.append(videoA)
        video_pairs.append(videoB)

    return video_pairs


def make_mask(tck, u, delta=0.1):
    """make a segmentation mask from a polyline"""
    pts = curve.sample_spline(tck, u, delta=delta).astype(np.int32)

    mask = np.zeros((1024, 1024), dtype=np.uint8)
    pts = np.array(pts, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
    mask = mask // 255
    return mask


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
        mask_transform: callable = None,
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
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = cv2.imread(str(self.root / sample["image"]), cv2.IMREAD_GRAYSCALE)
        mask = sample["mask"]
        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask


def visualize_mask(img, mask):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(mask, cmap="gray")
    plt.show()


def test_dataset():
    import guide3d.vars as vars

    dataset_path = vars.dataset_path
    dataset = Guide3D(dataset_path, "sphere_wo_reconstruct.json")
    print(len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataloader:
        img, mask = batch
        print(img.shape, mask.shape)
        break


if __name__ == "__main__":
    test_dataset()
