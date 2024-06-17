from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision import io

import annot_parser
import dl.dl_utils as dl_utils


class Video2Reconstruction(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations_3.xml",
        download: bool = False,
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
        seg_len: int = 3,
        max_len: int = 150,
    ):
        self.root = Path(root)
        self.annotations_file = annotations_file
        self.data = annot_parser.parse_to_video_separated(
            self.root / self.annotations_file, seg_len
        )

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


class Video2ReconstructionSpherical(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations_3.xml",
        download: bool = False,
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
        seg_len: int = 3,
        max_len: int = 150,
    ):
        self.root = Path(root)
        self.annotations_file = annotations_file
        self.data = annot_parser.parse_to_video_separated(
            self.root / self.annotations_file, seg_len
        )
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

        pts = [torch.tensor(pt, dtype=torch.float32) for pt in pts]
        tips = [pt[0] for pt in pts]
        offsets = [pt[1:] for pt in pts]
        offsets = [dl_utils.cartesian_to_spherical_chain(offset) for offset in offsets]

        lengths = [torch.tensor(len(points), dtype=torch.int32) for points in pts]
        lengths = torch.stack(lengths, dim=0).unsqueeze(1)

        tips = torch.stack(tips, dim=0)

        pts = [torch.cat((pt, torch.zeros((self.max_len - len(pt), 3)))) for pt in pts]
        pts = torch.stack(pts, dim=0)

        offsets = [torch.clone(offset).detach() for offset in offsets]
        offsets = [
            torch.cat((offset, torch.zeros((self.max_len - 1 - len(offset), 2))))
            for offset in offsets
        ]
        offsets = torch.stack(offsets, dim=0)
        return imgs, tips, offsets, lengths


def _test_video2reconstruction():
    import vars

    dataset = Video2Reconstruction(
        root=vars.dataset_path, annotations_file="annotations_3.xml"
    )

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    imgs, pts, seq_len = next(iter(dataloader))
    print("Images shape:", imgs.shape)
    print("Points shape:", pts.shape)
    print("Sequence Lengths shape:", seq_len.shape)


def _test_video2reconstruction_spherical():
    import vars

    dataset = Video2ReconstructionSpherical(
        root=vars.dataset_path, annotations_file="annotations_3.xml"
    )

    imgs, tips, offsets, lengths = dataset[0]
    print("Images shape:", imgs.shape)
    print("Tips shape:", tips.shape)
    print("Offsets shape:", offsets.shape)
    print("Lengths shape:", lengths.shape)

    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)

    imgs, tips, offsets, lengths = next(iter(dataloader))
    # assert (imgs.shape == (2, 3, 1))
    print("Images shape:", imgs.shape)
    print("Tips shape:", tips.shape)
    print("Offsets shape:", offsets.shape)
    print("Lengths shape:", lengths.shape)


def get_distance_between_points(pts):
    return torch.norm(pts[1:] - pts[:-1], dim=1)


if __name__ == "__main__":
    # _test_video2reconstruction()
    _test_video2reconstruction_spherical()
