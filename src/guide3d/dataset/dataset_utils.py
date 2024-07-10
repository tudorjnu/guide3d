from typing import Dict, List, Union

import numpy as np


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
