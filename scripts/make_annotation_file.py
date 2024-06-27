"""This script makes a JSON file with the annotations"""

import json

import numpy as np
from parse_cvat import get_structured_dataset
from reconstruction import reconstruct
from representations import curve
from tqdm import tqdm


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


def reorder_points(points: list) -> list:
    pt0 = points[0]
    pt_last = points[-1]
    if pt0[0] < pt_last[0]:
        points = points[::-1]
    return points


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
                    "points": reorder_points(annotation["points1"].tolist()),
                },
                "camera1": {
                    "image": annotation["image2"],
                    "points": reorder_points(annotation["points2"].tolist()),
                },
            }

            frames.append(frame)
        video_pair["frames"] = frames
        root.append(video_pair)

    return root


def decompose_tck(tck):
    assert isinstance(tck, list) or isinstance(
        tck, tuple
    ), f"tck should be a tuple or list, but got {type(tck)}\n"

    t, c, k = tck
    assert isinstance(t, np.ndarray), f"t should be a numpy array, but got {type(t)}\n"
    assert isinstance(c, list), f"c should be a list, but got {type(c)}\n"
    assert isinstance(k, int), f"k should be an integer, but got {type(k)}\n"

    t = t.tolist()
    c = [c_i.tolist() for c_i in c]

    return t, c, k


def make_json_spherical(dataset: dict, with_reconstruction: bool = False):
    root = []

    for key, value in tqdm(dataset.items()):
        properties = parse_key(key)

        video_pair = {}
        video_pair["fluid"] = properties["fluid"]
        video_pair["guidewire_type"] = properties["guidewire_type"]
        video_pair["video_number"] = properties["video_number"]
        video_pair["frame_count"] = len(value)
        video_pair["task"] = key

        frames = []
        for frame_number, annotation in tqdm(value.items()):
            imageA = annotation["image1"]
            imageB = annotation["image2"]
            ptsA = reorder_points(annotation["points1"].tolist())
            ptsB = reorder_points(annotation["points2"].tolist())

            ptsA = np.array(ptsA)
            ptsB = np.array(ptsB)

            tckA, uA = curve.fit_spline(ptsA, s=0.3)
            tckB, uB = curve.fit_spline(ptsB, s=0.3)

            # needed for JSON
            tA, cA, kA = decompose_tck(tckA)
            tB, cB, kB = decompose_tck(tckB)

            frame = {
                "frame_number": frame_number,
                "cameraA": {
                    "image": imageA,
                    "tck": {
                        "t": tA,
                        "c": cA,
                        "k": kA,
                    },
                    "u": uA.tolist(),
                },
                "cameraB": {
                    "image": imageB,
                    "tck": {
                        "t": tB,
                        "c": cB,
                        "k": kB,
                    },
                    "u": uB.tolist(),
                },
            }

            if with_reconstruction:
                u_max = min(uA[-1], uB[-1])
                tck3d, u3d = reconstruct(tckA, tckB, u_max)
                if tck3d is None:
                    continue
                t3d, c3d, k3d = decompose_tck(tck3d)
                u3d = u3d.tolist()
                frame["3d"] = {
                    "tck": {
                        "t": t3d,
                        "c": c3d,
                        "k": k3d,
                    },
                    "u": u3d,
                }

            frames.append(frame)
        video_pair["frames"] = frames
        root.append(video_pair)

    return root


def main():
    dataset = get_structured_dataset("data/annotations/cvat.xml")

    dataset = parse_into_dict(dataset)

    with open("data/annotations/sphere.json", "w") as f:
        json_data = make_json_spherical(dataset, with_reconstruction=True)
        json.dump(json_data, f, indent=2)

    with open("data/annotations/raw.json", "w") as f:
        json_data = make_json(dataset)
        json.dump(json_data, f, indent=2)

    with open("data/annotations/3d.json", "w") as f:
        json_data = make_json(dataset)
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
