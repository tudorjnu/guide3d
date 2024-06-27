import json
from typing import List

import numpy as np


def parse(annotation_file: str) -> List:
    json_file = json.load(open(annotation_file))
    return json_file


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
