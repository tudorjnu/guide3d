import json
from typing import Dict, List

import numpy as np


def parse(annotation_file: str) -> List:
    json_file = json.load(open(annotation_file))
    return json_file


def get_data(
    annotation_file: str = "data/annotations/raw.json", frames: List = [70, 100, 150]
) -> List:
    import json

    dummy_data = []
    annotations = json.load(open(annotation_file))
    for video in annotations:
        for frame in video["frames"]:
            if frame["frame_number"] in frames:
                dummy_data.append(
                    {
                        "img1": frame["camera1"]["image"],
                        "img2": frame["camera2"]["image"],
                        "pts1": np.array(frame["camera1"]["points"]),
                        "pts2": np.array(frame["camera2"]["points"]),
                    }
                )
    return dummy_data


def decompose_path(img_path: str) -> str:
    path = img_path.split("/")[0]
    camera = path.split("-")[-2]
    img_number = img_path.split("/")[-1].split(".")[0]
    return path, camera, img_number


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


if __name__ == "__main__":
    import vars

    data_path = vars.dataset_path

    parsed_data = parse(data_path / "annotations.xml")
    print(len(parsed_data))
