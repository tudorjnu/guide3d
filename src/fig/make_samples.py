from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from utils import viz


def parse_frame(frame: Dict):
    cameraA = frame["cameraA"]
    cameraB = frame["cameraB"]

    imgA = cameraA["image"]
    imgB = cameraB["image"]

    tckA = preprocess_tck(cameraA["tck"])
    tckB = preprocess_tck(cameraB["tck"])
    tck3d = preprocess_tck(frame["3d"]["tck"])

    uA = np.array(cameraA["u"])
    uB = np.array(cameraB["u"])
    u3d = np.array(frame["3d"]["u"])

    return dict(
        imgA=imgA,
        imgB=imgB,
        tckA=tckA,
        tckB=tckB,
        uA=uA,
        uB=uB,
        tck3d=tck3d,
        u3d=u3d,
    )


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


def get_data():
    import json

    dummy_data = []
    annotations = json.load(open("data/annotations/sphere.json"))
    for video in annotations:
        for frame in video["frames"]:
            dummy_data.append(parse_frame(frame))
    return dummy_data


def main():
    import vars

    save_path = vars.viz_dataset_path / "outlined_guidewire"
    if not save_path.exists():
        save_path.mkdir()

    data = get_data()
    for frame in data:
        imgA = frame["imgA"]
        imgB = frame["imgB"]

        imgA = plt.imread(vars.dataset_path / imgA)
        imgB = plt.imread(vars.dataset_path / imgB)

        imgA = viz.convert_to_color(imgA)
        imgB = viz.convert_to_color(imgB)

        tckA = frame["tckA"]
        tckB = frame["tckB"]
        tck3d = frame["tck3d"]

        uA = frame["uA"]
        uB = frame["uB"]
        u3d = frame["u3d"]

        imgA = viz.draw_curve(imgA, tckA, uA, color="white")
        imgB = viz.draw_curve(imgB, tckB, uB, color="white")

        imgA_path = save_path / frame["imgA"]
        imgB_path = save_path / frame["imgB"]

        if not imgA_path.parent.exists() or not imgB_path.parent.exists():
            imgA_path.parent.mkdir(parents=True)
            imgB_path.parent.mkdir(parents=True)

        plt.imsave(imgA_path, imgA)
        plt.imsave(imgB_path, imgB)

        # plt.imshow(imgA)
        # plt.show()
        # plt.close()

        # pts3d = np.column_stack(splev(u3d, tck3d))
        # plot.make_3d_plot_with_mesh(pts3d)
        # plt.show()
        # plt.close()


if __name__ == "__main__":
    main()
