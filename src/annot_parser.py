import xml.etree.ElementTree as ET
from pathlib import Path

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


def parse(annotation_file: Path) -> dict:
    annotations = []
    tree = etree.parse(annotation_file)
    root = tree.getroot()

    for video_pair in root.findall("VideoPair"):
        video_pair_dict = {}

        video_pair_dict["fluid"] = video_pair.get("fluid")
        video_pair_dict["task"] = video_pair.get("task")
        video_pair_dict["guidewire_type"] = video_pair.get("guidewire_type")
        video_pair_dict["frames"] = []

        for frame in video_pair.findall("Frame"):
            frame_dict = {}
            for camera in frame.findall("Camera"):
                frame_dict[camera.get("number")] = {
                    "image": camera.get("image"),
                    "points": parse_points(camera.get("points"), np.int32),
                }

            frame_dict["reconstruction"] = parse_points(
                frame.find("Reconstruction").get("points"), np.float32
            )
            video_pair_dict["frames"].append(frame_dict)
        annotations.append(video_pair_dict)
    return annotations


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
                if camera.get("number") == "2":
                    img2_path = camera.get("image")
                    img2_pts = camera.get("points")
                    img2_pts = parse_points(img2_pts, dtype=np.int32)

            reconstruction = frame.find("Reconstruction")
            pts_reconstructed = parse_points(reconstruction.get("points"), np.float32)

            flattened_data.append(
                dict(
                    img=img1_path,
                    pts=img1_pts,
                    pts_reconstructed=pts_reconstructed,
                )
            )
            # NOTE: Commenting as the projection is not saved, therefore cannot
            # visualize the projection
            # flattened_data.append(
            #     dict(
            #         img=img2_path,
            #         pts=img2_pts,
            #         pts_reconstructed=pts_reconstructed,
            #     )
            # )

    return flattened_data


def _test_parse_xml_file_flattened():
    import vars

    root_path = vars.dataset_path / "annotations.xml"
    annotations = parse_xml_file_flattened(root_path.as_posix())
    print(len(annotations))


def parse_to_video(file_path: Path, seg_len: int) -> list:
    tree = ET.parse(file_path)
    root = tree.getroot()

    video_segments = []

    for video_pair in root.findall("VideoPair"):
        frames = list(video_pair.findall("Frame"))
        for i in range(len(frames) - seg_len + 1):
            segment = frames[i : i + seg_len]
            segment_data = []

            for frame in segment:
                frame_data = {
                    "img1_path": None,
                    "img2_path": None,
                    "pts1": None,
                    "pts2": None,
                    "pts_reconstructed": None,
                }

                for camera in frame.findall("Camera"):
                    if camera.get("number") == "1":
                        frame_data["img1_path"] = camera.get("image")
                        frame_data["pts1"] = parse_points(
                            camera.get("points"), np.int32
                        )
                    elif camera.get("number") == "2":
                        frame_data["img2_path"] = camera.get("image")
                        frame_data["pts2"] = parse_points(
                            camera.get("points"), np.int32
                        )

                reconstruction = frame.find("Reconstruction")
                frame_data["pts_reconstructed"] = parse_points(
                    reconstruction.get("points"), np.float32
                )

                segment_data.append(frame_data)

            video_segments.append(segment_data)

    return video_segments


def parse_to_video_separated(
    file_path: Path,
    seg_len: int,
    pts_transform: callable = None,
) -> list:
    tree = ET.parse(file_path)
    root = tree.getroot()

    video_segments = []

    for video_pair in root.findall("VideoPair"):
        frames = list(video_pair.findall("Frame"))
        for i in range(len(frames) - seg_len + 1):
            segment = frames[i : i + seg_len]
            segment_data = dict(
                imgs1_paths=[],
                imgs2_paths=[],
                pts1=[],
                pts2=[],
                pts_reconstructed=[],
            )

            for frame in segment:
                for camera in frame.findall("Camera"):
                    if camera.get("number") == "1":
                        segment_data["imgs1_paths"].append(camera.get("image"))
                        segment_data["pts1"].append(
                            parse_points(camera.get("points"), np.int32)
                        )
                    elif camera.get("number") == "2":
                        segment_data["imgs2_paths"].append(camera.get("image"))
                        segment_data["pts2"].append(
                            parse_points(camera.get("points"), np.int32)
                        )

                reconstruction = frame.find("Reconstruction")
                reconstructed_points = parse_points(
                    reconstruction.get("points"), np.float32
                )
                if pts_transform:
                    reconstructed_points = pts_transform(reconstructed_points)
                segment_data["pts_reconstructed"].append(reconstructed_points)

            video_segments.append(segment_data)

    return video_segments


def _test_parse_to_video():
    import vars

    root_path = vars.dataset_path / "annotations.xml"
    annotations = parse_to_video(root_path.as_posix(), 3)
    print(len(annotations))


def main():
    import utils

    file_path = Path.cwd() / "annotations_flat.xml"
    annotations = parse(file_path)
    for video in annotations:
        utils.visualize_video(video)
    # frame = annotations[0]["frames"][0]
    # print(frame["1"]["points"].shape)
    # print(frame["2"]["points"].shape)
    # print(frame["reconstruction"].shape)


def _test_parse_flattened():
    import vars

    root_path = vars.dataset_path / "annotations_3.xml"
    annotations = parse_xml_file_flattened(root_path)

    print(len(annotations))


if __name__ == "__main__":
    _test_parse_flattened()
    # _test_parse_to_video()
    # main()
