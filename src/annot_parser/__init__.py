import json
from typing import Dict, List, Union

import numpy as np


def parse(annotation_file: str) -> List[List[Dict[str, Union[str, np.ndarray]]]]:
    json_file = json.load(open(annotation_file))

    annotations = []

    for video in json_file:
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
                    frame_data["img1"]["points"] = np.array(
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


def main():
    pass


if __name__ == "__main__":
    main()
