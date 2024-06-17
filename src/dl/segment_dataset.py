from pathlib import Path

import torch
from torch.utils import data
from torchvision import transforms, io
import cv2
import numpy as np


import xml.etree.ElementTree as ET
from typing import List, Dict, Union


def parse_polyline(polyline_element):
    """Parse a polyline element to extract points."""
    points_str = polyline_element.get("points")
    points = np.array(
        [list(map(float, point.split(","))) for point in points_str.split(";")]
    )
    return points


def parse_image(image_element):
    """Parse an image element to extract its name and associated polylines."""
    name = image_element.get("name")
    polyline = parse_polyline(image_element.find("polyline"))
    return name, polyline


def pair_camera_annotations(annotations):
    """Pair camera annotations based on naming convention and extract first polyline points."""

    # Group annotations by task identifier (excluding camera number)

    grouped_annotations = {}
    for image_path, polyline in annotations:
        task_identifier = image_path.split("/")[0]
        sample_identifier = image_path.split("/")[1].split(".")[0]
        camera_number = task_identifier[-1]
        task_name = task_identifier[:-2] + sample_identifier

        if task_name not in grouped_annotations:
            grouped_annotations[task_name] = {}

        if camera_number == "1":
            grouped_annotations[task_name]["image1"] = image_path
            grouped_annotations[task_name]["points1"] = polyline
        elif camera_number == "2":
            grouped_annotations[task_name]["image2"] = image_path
            grouped_annotations[task_name]["points2"] = polyline

    # get the values and flatten the dictionary
    structured_dataset = [v for v in grouped_annotations.values()]

    # remove incomplete pairs
    structured_dataset = [pair for pair in structured_dataset if len(pair) == 4]

    return structured_dataset


def parse_annotation_file(file_path) -> List[Dict[str, Union[str, np.ndarray]]]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    annotations = [parse_image(image) for image in root.findall("image")]
    paired_annotations = pair_camera_annotations(annotations)
    return paired_annotations


image_transforms = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]
)


def polyline_to_mask(polyline):
    """make a segmentation mask from a polyline"""
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    pts = np.array(polyline, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
    # mask should be binary
    mask = mask // 255
    return mask


def split_annotations(annotations):
    splitted_annotations = []
    for annot in annotations:
        pts1 = annot["points1"]
        pts2 = annot["points2"]
        img1 = annot["image1"]
        img2 = annot["image2"]
        splitted_annotations.append({"img": img1, "pts": pts1})
        splitted_annotations.append({"img": img2, "pts": pts2})
    return splitted_annotations


class BESTDataset(data.Dataset):
    def __init__(self, dataset_path: Path, annotation_file: str, image_transform=None):
        self.dataset = parse_annotation_file(dataset_path / annotation_file)
        self.dataset = split_annotations(self.dataset)
        self.dataset_path = dataset_path
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_entry = self.dataset[index]
        img_path = self.dataset_path / data_entry["img"]
        pts = data_entry["pts"]
        mask = polyline_to_mask(pts)

        img = io.read_image(img_path.as_posix())
        if self.image_transform:
            img = self.image_transform(img)

        pts = torch.tensor(pts, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.int8)

        return {"img": img, "mask": mask}


def visualize_mask(img, mask):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(mask, cmap="gray")
    plt.show()


def test_dataset():
    dataset_path = Path.cwd() / "data"
    dataset = BESTDataset(
        dataset_path, "annotations.xml", image_transform=image_transforms
    )

    sample = dataset[0]
    img = sample["img"]
    mask = sample["mask"]
    print(img.shape, mask.shape)
    print(img.dtype, mask.dtype)
    print("min", img.min(), mask.min())
    print("max", img.max(), mask.max())

    visualize_mask(img.permute(1, 2, 0), mask)


def test_full_dataset():
    dataset_path = Path.home() / "data" / "segment-real"
    dataset = BESTDataset(
        dataset_path, "annotations.xml", image_transform=image_transforms
    )
    print(len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataloader:
        print(batch["img"].shape, batch["mask"].shape)
        break


if __name__ == "__main__":
    test_dataset()
    test_full_dataset()
