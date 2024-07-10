import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torchvision import io

import annot_parser
from dl.networks import feature_extractors

_image_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Guide3D(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations.xml",
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


def collate_fn(batch):
    imgs, pts, lengths = zip(*batch)

    imgs = torch.stack(imgs, dim=0)

    padded_pts = pad_sequence(pts, batch_first=True, padding_value=0)

    lengths = torch.tensor(lengths)

    return imgs, padded_pts, lengths


class Model(pl.LightningModule):
    def __init__(self, feature_dim, max_len):
        super(Model, self).__init__()
        self.feature_dim = feature_dim
        self.max_len = max_len
        self.feature_extractor = feature_extractors.ResNet50(
            output_features=feature_dim
        )

        self.pts_pred = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * max_len),
        )
        self.stop_pred = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_len),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        point_pred = self.pts_pred(features)
        point_pred = point_pred.view(-1, self.max_len, 3)
        stop_pred = self.stop_pred(features)

        return point_pred, stop_pred


def _test_dataset():
    import vars

    root_path = vars.dataset_path
    dataset = Guide3D(root_path)
    print(len(dataset))


def _test_loader():
    import vars

    root_path = vars.dataset_path
    dataset = Guide3D(root_path)
    dataloader = data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    for i, (img, pts, stop_targets) in enumerate(dataloader):
        print(img.shape, pts.shape, stop_targets.shape)


def _test_model():
    import vars

    root_path = vars.dataset_path
    dataset = Guide3D(root_path)
    dataloader = data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    model = Model(feature_dim=512, max_len=150)

    with torch.no_grad():
        for i, (img, pts, stop_targets) in enumerate(dataloader):
            print(img.shape, pts.shape, stop_targets.shape)
            pts_pred, stop_pred = model(img)
            print(pts_pred.shape, stop_pred.shape)
            if i == 10:
                break


def _test():
    _test_dataset()
    # _test_loader()
    # _test_model()


if __name__ == "__main__":
    _test()
