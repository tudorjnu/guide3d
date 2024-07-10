import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def parse_polyline(polyline_element: ET.Element) -> np.ndarray:
    points_str: str = polyline_element.get("points", "")
    polyline = np.array(
        [list(map(float, point.split(","))) for point in points_str.split(";")],
        dtype=np.int32,
    )
    return polyline


def extract_properties(name: str) -> dict:
    path = name
    sample_number = name.split("/")[1].split(".")[0]
    name = name.split("/")[0]

    fluid, target, guidewire, sample_id, camera = name.split("-")
    camera = int(camera)

    return dict(
        sample_number=sample_number,
        fluid=int(fluid),
        target=target,
        guidewire=guidewire,
        sample_id=int(sample_id),
        camera=int(camera),
        path=path,
    )


def parse_image(image_element: ET.Element) -> dict:
    name = image_element.get("name", "")
    properties = extract_properties(name)
    properties["task_id"] = image_element.get("task_id", "")
    properties["polyline"] = parse_polyline(image_element.find("polyline"))
    return properties


def parse_annotation_file(file_path: str) -> list:
    tree = ET.parse(file_path)
    root = tree.getroot()
    images = [parse_image(image) for image in root.findall("image")]
    return images


def build_xml(annotated_parser):
    et_root = ET.Element("root")
    for annotation in annotated_parser:
        et_image = ET.SubElement(et_root, "image")
        et_image.set("name", annotation["path"])
        et_image.set("task_id", annotation["task_id"])
        et_polyline = ET.SubElement(et_image, "polyline")
        et_polyline.set(
            "points",
            ";".join([",".join(map(str, point)) for point in annotation["polyline"]]),
        )
    pass


if __name__ == "__main__":
    annotation_path = Path.cwd() / "data" / "annotations.xml"
    # annotation_path = Path.home() / "data" / "segment-real" / "annotations.xml"
    parsed_data = parse_annotation_file(annotation_path.as_posix())
    build_xml(parsed_data)
