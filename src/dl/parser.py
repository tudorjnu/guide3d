import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def parse_points(pts: str, dtype: np.dtype = np.int32) -> np.ndarray:
    pts = pts.split(";")
    pts = [tuple(map(float, pt.split(","))) for pt in pts if pt]
    pts_array = np.array(pts, dtype=dtype)
    return pts_array


def parse_xml_file_flattened(file_path: Path) -> list:
    tree = ET.parse(file_path)
    root = tree.getroot()

    flattened_data = []

    for video_pair in root.findall("VideoPair"):
        for frame in video_pair.findall("Frame"):
            img1_path = None
            img1_pts = None

            img2_path = None
            img2_pts = None

            for camera in frame.findall("Camera"):
                if camera.get("number") == "1":
                    img1_path = camera.get("image")
                    img1_pts = camera.get("points")
                    img1_pts = parse_points(img1_pts, dtype=np.int32)
                elif camera.get("number") == "2":
                    img2_path = camera.get("image")
                    img2_pts = camera.get("points")
                    img2_pts = parse_points(img2_pts, dtype=np.int32)

            reconstruction = frame.find("Reconstruction")
            reconstructed_points = parse_points(
                reconstruction.get("points"), np.float32
            )

            flattened_data.append(
                dict(
                    img=img1_path,
                    pts=img1_pts,
                    reconstructed_pts=reconstructed_points,
                    camera_number=1,
                )
            )
            flattened_data.append(
                dict(
                    img=img2_path,
                    pts=img2_pts,
                    reconstructed_pts=reconstructed_points,
                    camera_number=2,
                )
            )

    return flattened_data


def parse_xml_file(file_path: Path) -> list:
    tree = ET.parse(file_path)
    root = tree.getroot()

    dataset = []

    for video_pair_element in root.findall("VideoPair"):
        fluid = video_pair_element.get("fluid")
        guidewire_type = video_pair_element.get("guidewire_type")
        video_number = video_pair_element.get("video_number")

        video_pair_data = {
            "fluid": fluid,
            "guidewire_type": guidewire_type,
            "video_number": video_number,
            "frames": [],
        }

        for frame in video_pair_element.findall("Frame"):
            frame_number = frame.get("number")
            frame_data = {
                "number": frame_number,
                "cameras": [],
                "reconstruction": None,
            }

            for camera in frame.findall("Camera"):
                camera_number = camera.get("number")
                image = camera.get("image")
                points = camera.get("points", np.int32)
                points_array = parse_points(points)
                frame_data["cameras"].append(
                    {
                        "number": camera_number,
                        "image": image,
                        "points": points_array,
                    }
                )

            reconstruction = frame.find("Reconstruction")
            if reconstruction is not None:
                points = reconstruction.get("points")
                points_array = parse_points(points, np.float32)
                frame_data["reconstruction"] = points_array

            video_pair_data["frames"].append(frame_data)

        dataset.append(video_pair_data)

    return dataset


def test_flatten():
    import vars

    root_path = vars.dataset_path / "annotations.xml"
    annotations = parse_xml_file_flattened(root_path.as_posix())
    print(len(annotations))
    for annotation in annotations:
        print(annotation["img"])


def test():
    import vars

    root_path = vars.dataset_path / "annotations.xml"
    annotations = parse_xml_file(root_path.as_posix())
    for annotation in annotations:
        frames = annotation["frames"]
        frame = frames[0]
        for k, v in frame.items():
            print("key", k)
        print(annotation["video_number"])
        exit()


if __name__ == "__main__":
    test_flatten()
