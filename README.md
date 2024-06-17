# Shape Prediction

## Initial Results

| Model | MERS (mm) | MaxED (mm) | METE (mm) |
|-------------- | -------------- | -------------- | --|
| Linear | 3.2 ± 0.07 | 4.4 ± 0.92|2.7 ± 0.56|
| M2M | | | |

## Preparing the Dataset

I decided to go for multiple datasets for multiple use cases (i.e. segmentation, shape prediction, video vs image based).
Considerations include parsing the dataset in an appropriate format.

### Shape prediction image based

The image case is simpler, so I started with this.

**Parser:**

For the parser, I gathered all images and reconstruction points and returned them in
a flatten way.

```python
def parse_xml_file_flattened(file_path: Path) -> list:
    tree = ET.parse(file_path)
    root = tree.getroot()

    flattened_data = []

    for video_pair in root.findall("VideoPair"):
        for frame in video_pair.findall("Frame"):
            img1_path = None
            img1_pts = None

            for camera in frame.findall("Camera"):
                if camera.get("number") == "1":
                    img1_path = camera.get("image")
                    img1_pts = camera.get("points")
                    img1_pts = parse_points(img1_pts, dtype=np.int32)

            reconstruction = frame.find("Reconstruction")
            pts_reconstructed = parse_points(reconstruction.get("points"), np.float32)

            flattened_data.append(
                dict(
                    img=img1_path,
                    pts=img1_pts,
                    pts_reconstructed=pts_reconstructed,
                )
            )

    return flattened_data
```

**Dataset:**

Due to padding constraints and in an attempt to maintain the original size information,
I decided to directly output the `seq_len` along with the images and the reconstructed points.

```python
class Guide3D(data.Dataset):
    def __init__(
        self,
        root: str,
        annotations_file: str = "annotations.xml",
        download: bool = False,
        image_transform: transforms.Compose = _image_transforms,
    ):
        self.root = root
        self.annotations_file = annotations_file
        self.data = annot_parser.parse_xml_file_flattened(
            self.root / self.annotations_file
        )
        self.image_transform = image_transform

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

        seq_len = len(pts)

        return img, pts, seq_len

```

**Collating function:**

In order to have batch processing, The batch has to be padded:

```python
def collate_fn(batch):
    imgs, pts, lengths = zip(*batch)

    imgs = torch.stack(imgs, dim=0)

    padded_pts = pad_sequence(pts, batch_first=True, padding_value=0)

    lengths = torch.tensor(lengths)

    return imgs, padded_pts, lengths
```

#### Model

I ended up creating a simple model that predicts a set of points along with the
stop predictions.

```python
class Model(pl.LightningModule):
    def __init__(self, hidden_dim, max_len):
        super(Model, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.feature_extractor = feature_extractors.ResNet50(output_features=hidden_dim)

        self.pts_pred = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * max_len),
        )
        self.stop_pred = nn.Sequential(
            nn.Linear(hidden_dim, 512),
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
```
