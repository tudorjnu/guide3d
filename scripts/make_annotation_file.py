"""This script makes a JSON file with the annotations"""

import json

from parse_cvat import get_structured_dataset

import calibration as cal
import fn
from reconstruct import get_points


def extract_properties(name: str) -> (str, str):
    folder, frame_number = name.split("/")
    fluid, task, guidewire_type, video_number, camera_number = folder.split("-")

    frame_number = frame_number.split(".")[0]

    return dict(
        fluid=int(fluid),
        task=task,
        guidewire_type=guidewire_type,
        video_number=int(video_number),
        camera_number=int(camera_number),
        frame_number=int(frame_number),
    )


def parse_into_dict(annotations: list) -> dict:
    dataset = {}
    for annotation in annotations:
        properties = extract_properties(annotation["image1"])
        properties.pop("camera_number")

        frame_number = properties.pop("frame_number")

        dataset_key = "-".join(str(value) for value in properties.values())
        dataset.setdefault(dataset_key, {})
        dataset[dataset_key][frame_number] = dict(
            frame_number=frame_number, **annotation
        )

    return dataset


def parse_key(key: str) -> dict:
    fluid, task, guidewire_type, video_number = key.split("-")
    return dict(
        fluid=int(fluid),
        guidewire_type=guidewire_type,
        video_number=int(video_number),
    )


def make_json(dataset: dict, with_reconstruction: bool = False):
    root = []

    for key, value in dataset.items():
        properties = parse_key(key)

        video_pair = {}
        video_pair["fluid"] = properties["fluid"]
        video_pair["guidewire_type"] = properties["guidewire_type"]
        video_pair["video_number"] = properties["video_number"]
        video_pair["frame_count"] = len(value)
        video_pair["task"] = key

        frames = []
        for frame_number, annotation in value.items():
            frame = {
                "frame_number": frame_number,
                "camera2": {
                    "image": annotation["image1"],
                    "points": annotation["points1"].tolist(),
                },
                "camera1": {
                    "image": annotation["image2"],
                    "points": annotation["points2"].tolist(),
                },
            }

            if with_reconstruction:
                _, _, pts3D = get_points(
                    annotation["points1"], annotation["points2"], cal.P1, cal.P2
                )
                pts3D = fn.interpolate_even_spacing(pts3D, 0.2)
                frame["reconstruction"] = pts3D.tolist()

            frames.append(frame)
        video_pair["frames"] = frames
        root.append(video_pair)

    return root


def main():
    dataset = get_structured_dataset("data/annotations/cvat.xml")

    dataset = parse_into_dict(dataset)

    with open("data/annotations/raw.json", "w") as f:
        json_data = make_json(dataset)
        json.dump(json_data, f, indent=2)

    with open("data/annotations/3d.json", "w") as f:
        json_data = make_json(dataset)
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
