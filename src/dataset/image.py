import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision import io

import annot_parser
import calibration
import dl.dl_utils as dl_utils


class Img2Reconstruction(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations_3.xml",
        download: bool = False,
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
    ):
        self.root = root
        self.annotations_file = annotations_file
        self.data = annot_parser.parse_xml_file_flattened(
            self.root / self.annotations_file
        )

        self.image_transform = image_transform
        self.pts_transform = pts_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]

        img_path = self.root / data_entry["img"]
        img = io.read_image(img_path.as_posix())

        if self.image_transform:
            img = self.image_transform(img)

        pts = data_entry["pts_reconstructed"]
        pts = torch.tensor(pts, dtype=torch.float32)

        if self.pts_transform:
            pts = self.pts_transform(pts)

        seq_len = len(pts)
        seq_len = torch.tensor(seq_len, dtype=torch.int32)

        return img, pts, seq_len


class Img2ReconstructionWSpherical(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations_3.xml",
        download: bool = False,
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
    ):
        self.root = root
        self.annotations_file = annotations_file
        self.data = annot_parser.parse_xml_file_flattened(
            self.root / self.annotations_file
        )

        self.image_transform = image_transform
        self.pts_transform = pts_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]

        img_path = self.root / data_entry["img"]
        img = io.read_image(img_path.as_posix())

        if self.image_transform:
            img = self.image_transform(img)

        pts = data_entry["pts_reconstructed"]
        pts = torch.tensor(pts, dtype=torch.float32)

        if self.pts_transform:
            pts = self.pts_transform(pts)

        tip = pts[0]
        body = dl_utils.cartesian_to_spherical_chain(pts)

        seq_len = len(body)
        seq_len = torch.tensor(seq_len, dtype=torch.int32)

        return img, tip, body, seq_len


def _test_img2reconstruction():
    import dl.dl_utils as dl_utils
    import vars

    dataset = Img2Reconstruction(
        root=vars.dataset_path, annotations_file="annotations_3.xml"
    )

    assert len(dataset) == 4100
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    imgs, pts, seq_len = next(iter(dataloader))
    assert imgs.shape == (1, 1, 1024, 1024), f"Shape mismatch:{imgs.shape}"
    assert pts.shape[0] == 1, f"Shape mismatch:{pts.shape}"
    assert pts.shape[2] == 3, f"Shape mismatch:{pts.shape}"
    assert seq_len.shape == (1,), f"Shape mismatch:{seq_len.shape}"

    imgs, pts, seq_len = dataset[0]
    # non batched operations
    pts = pts[:3]
    tip = pts[0]
    body = pts[1:]
    print("Original Points:", pts)
    spherical_pts = dl_utils.cartesian_to_spherical_chain(pts)
    print("Spherical Points:", spherical_pts)
    cartesian_pts = dl_utils.spherical_offsets_to_cartesian_chain(
        tip, spherical_pts, 0.2
    )
    print("Cartesian Points:", cartesian_pts)


def get_distance_between_points(pts):
    return torch.norm(pts[1:] - pts[:-1], dim=1)


def _test_image2reconstructionSpherical():
    import dl.dl_utils as dl_utils
    import fn
    import vars

    dataset = Img2ReconstructionWSpherical(
        root=vars.dataset_path, annotations_file="annotations_3.xml"
    )

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    img, tip, pts, seq_len = next(iter(dataloader))
    print("Images shape:", img.shape)
    print("Tip shape:", tip.shape)
    print("Points shape:", pts.shape)
    chain = dl_utils.spherical_offsets_to_cartesian_chain(
        tip.squeeze(0), pts.squeeze(0), 0.002
    )
    print(get_distance_between_points(chain))
    projected_pts = fn.project_points(chain, calibration.P1)
    # print(projected_pts)
    print(get_distance_between_points(pts.squeeze(0)))
    print(seq_len)


if __name__ == "__main__":
    _test_img2reconstruction()
    _test_image2reconstructionSpherical()
