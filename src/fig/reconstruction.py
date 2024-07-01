from typing import Dict, List

import calibration
import matplotlib.pyplot as plt
import numpy as np
import representations.curve.viz as curve_viz
import vars
from representations import curve
from scipy.interpolate import splev
from utils import fn, viz


def parse_frame(frame: Dict):
    imgA = frame["cameraA"]["image"]
    imgB = frame["cameraB"]["image"]

    tckA = preprocess_tck(frame["cameraA"]["tck"])
    tckB = preprocess_tck(frame["cameraB"]["tck"])
    tck3d = preprocess_tck(frame["3d"]["tck"])

    uA = np.array(frame["cameraA"]["u"])
    uB = np.array(frame["cameraB"]["u"])
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


def get_errors(data: Dict, camera: str = "A", delta: float = 15):
    def process_frame(frame: Dict):
        if camera == "A":
            img = frame["imgA"]
            tck = frame["tckA"]
            u = frame["uA"]
        elif camera == "B":
            img = frame["imgB"]
            tck = frame["tckB"]
            u = frame["uB"]
        tck3d = frame["tck3d"]
        u3d = frame["u3d"]

        camera_matrix = calibration.P1 if camera == "A" else calibration.P2
        pts3d = np.column_stack(splev(u3d, tck3d))
        pts3d_reprojected = fn.project_points(pts3d, camera_matrix)
        img = plt.imread(vars.dataset_path / img)
        img = viz.convert_to_color(img)
        img = curve_viz.draw_curve(img, tck, u[-1])

        s = 0.5
        success = False
        while not success:
            try:
                tck_reprojected, u_reprojected = curve.fit_spline(pts3d_reprojected, s)
                success = True
            except RuntimeError as e:
                print("RuntimeError", e)
                s += 0.1

        img = curve_viz.draw_curve(
            img,
            tck_reprojected,
            u_reprojected[-1],
            color=(255, 0, 0),
        )
        plt.imshow(img)
        plt.show()
        plt.close()
        u_max = min(u[-1], u_reprojected[-1])
        ground_truth = curve.sample_spline(tck, u_max, delta)
        reprojection = curve.sample_spline(tck_reprojected, u_max, delta)

        return ground_truth, reprojection

    def pad_with_nan(arrays):
        max_length = max(len(arr) for arr in arrays)
        padded_arrays = []
        for arr in arrays:
            padding_length = max_length - len(arr)
            for i in range(padding_length):
                arr = np.vstack([arr, np.nan * np.ones(arr.shape[1])])
            padded_arrays.append(arr)
        return padded_arrays

    ground_truth = []
    reprojected = []
    for frame in data:
        try:
            gt, rp = process_frame(frame)
        except Exception as e:
            print(e)
            continue
        ground_truth.append(gt)
        reprojected.append(rp)

    ground_truth, reprojected = (
        pad_with_nan(ground_truth),
        pad_with_nan(reprojected),
    )

    ground_truth = np.array(ground_truth)
    reprojected = np.array(reprojected)
    print("Ground truth shape:", ground_truth.shape)
    print("Reprojected shape:", reprojected.shape)

    # calculate the euclidean distance between the ground truth and reprojected points
    distance = np.linalg.norm(ground_truth - reprojected, axis=2)
    print("Distance shape:", distance.shape)
    mean_errors = np.nanmean(distance, axis=0)
    std_errors = np.nanstd(distance, axis=0)

    print("Mean errors shape:", mean_errors.shape)
    print("Std errors shape:", std_errors.shape)

    print(mean_errors[0])
    exit()

    return mean_errors, std_errors


def make_reprojection_error_plot(data):
    def setup_fig():
        fig, ax = plt.subplots(1, 1, figsize=(2, 1))
        ax.set_ylabel("Error (pixels)")
        ax.set_xlabel("Point Index")
        ax.grid(
            True,
            which="both",
            axis="both",
            linestyle="--",
            linewidth=0.5,
            color="black",
            alpha=0.2,
        )
        return fig, ax

    data = clean_data(data)
    mean_errorsA, std_errorsA = get_errors(data, "A")
    mean_errorsB, std_errorsB = get_errors(data, "B")

    fig, ax = setup_fig()

    ax_props = dict(
        capsize=1,
        capthick=1,
        elinewidth=1,
        markersize=2,
        label="Reprojection error",
        fmt="o",
    )

    ax.errorbar(
        np.arange(len(mean_errorsA)),
        mean_errorsA,
        yerr=std_errorsA,
        **ax_props,
    )
    ax.errorbar(
        np.arange(len(mean_errorsB)),
        mean_errorsB,
        yerr=std_errorsB,
        **ax_props,
    )
    plt.tight_layout()
    fig.savefig("figs/reconstruction_reprojection_error.png")
    plt.close()


def clean_data(data: List[Dict]):
    for frame in data:
        frame["tckA"] = preprocess_tck(frame["tckA"])
        frame["tckB"] = preprocess_tck(frame["tckB"])
        frame["tck3d"] = preprocess_tck(frame["tck3d"])
        frame["uA"] = np.array(frame["uA"])
        frame["uB"] = np.array(frame["uB"])
        frame["u3d"] = np.array(frame["u3d"])

        ptsA = np.column_stack(splev(frame["uA"], frame["tckA"]))
        if np.any(ptsA > 1024) or np.any(ptsA < 0):
            print("Invalid ptsA")
            exit()
        ptsB = np.column_stack(splev(frame["uB"], frame["tckB"]))
        if np.any(ptsB > 1024) or np.any(ptsB < 0):
            print("Invalid ptsB")
            exit()


def main():
    data = get_data()
    print(len(data))
    make_reprojection_error_plot(data)
    pass


if __name__ == "__main__":
    main()
