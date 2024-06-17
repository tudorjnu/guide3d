from typing import Dict, List, Union

import numpy as np
from lxml import etree


def parse_points(points: str, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        convert = int
    elif np.issubdtype(dtype, np.floating):
        convert = float
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    parsed_points = [
        tuple(map(convert, point.split(","))) for point in points.split(";") if point
    ]

    return np.array(parsed_points, dtype=dtype)


def parse(annotation_file: str) -> List[List[Dict[str, Union[str, np.ndarray]]]]:
    annotations = []

    tree = etree.parse(annotation_file)
    root = tree.getroot()

    for video_pair in root.findall("VideoPair"):
        frames = []
        for frame in video_pair.findall("Frame"):
            frame_data = dict(
                img1=dict(
                    path=None,
                    points=None,
                ),
                img2=dict(
                    path=None,
                    points=None,
                ),
                reconstruction=None,
            )

            for camera in frame.findall("Camera"):
                if camera.get("number") == "1":
                    frame_data["img1"]["path"] = camera.get("image")
                    frame_data["img1"]["points"] = parse_points(
                        camera.get("points"), dtype=np.int32
                    )
                if camera.get("number") == "2":
                    frame_data["img2"]["path"] = camera.get("image")
                    frame_data["img2"]["points"] = parse_points(
                        camera.get("points"), dtype=np.int32
                    )

            reconstruction = frame.find("Reconstruction")
            frame_data["reconstruction"] = parse_points(
                reconstruction.get("points"), np.float32
            )
            frames.append(frame_data)
        annotations.append(frames)
    return annotations


def flatten(
    annotations: List[List[Dict[str, Union[str, np.ndarray]]]],
) -> List[Dict[str, Union[str, np.ndarray]]]:
    flattened_annotations = []
    for video_pair in annotations:
        for frame in video_pair:
            flattened_annotations.append(
                dict(
                    img=frame["img1"]["path"],
                    pts=frame["img1"]["points"],
                    reconstruction=frame["reconstruction"],
                )
            )
            flattened_annotations.append(
                dict(
                    img=frame["img2"]["path"],
                    pts=frame["img2"]["points"],
                    reconstruction=frame["reconstruction"],
                )
            )
    return flattened_annotations


def parametrize_curve(pts: np.ndarray, s: float = 0.5, k: int = 3):
    from scipy.interpolate import splprep

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    tck, u = splprep([x, y, z], s=s, k=k, u=cumulative_distances)
    return tck, u


if __name__ == "__main__":
    import vars

    data_path = vars.dataset_path

    parsed_data = parse(data_path / "annotations.xml")
    print(len(parsed_data))
    flattened_data = flatten(parsed_data)
    print(len(flattened_data))
