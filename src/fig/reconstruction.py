from typing import Dict, List

import calibration
import matplotlib.pyplot as plt
import numpy as np
import vars
from representations import curve
from scipy.interpolate import splev
from utils import fn


def plot_reprojected(img, pts, pts_reprojected, show=False):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.plot(
        pts[:, 0],
        pts[:, 1],
        "ro-",
        markersize=1,
        linewidth=0.5,
        alpha=0.7,
        label="Ground Truth",
    )
    plt.plot(
        pts_reprojected[:, 0],
        pts_reprojected[:, 1],
        "bo-",
        markersize=1,
        linewidth=0.5,
        alpha=0.7,
        label="Reprojected",
    )

    if show:
        plt.show()
    plt.close()


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


def get_errors(data: Dict, camera: str = "A", delta: float = 15):
    def is_valid_frame(pts_reproj):
        if np.any(pts_reproj > 1024) or np.any(pts_reproj < 0):
            return False
        return True

    def process_frame(frame: Dict):
        if camera == "A":
            img = frame["imgA"]
            tck = frame["tckA"]
            u = frame["uA"]
            camera_matrix = calibration.P1
        elif camera == "B":
            img = frame["imgB"]
            tck = frame["tckB"]
            u = frame["uB"]
            camera_matrix = calibration.P2
        else:
            raise ValueError("Invalid camera")
        tck3d = frame["tck3d"]
        u3d = frame["u3d"]

        pts3d = np.column_stack(splev(u3d, tck3d))
        pts3d_reprojected = fn.project_points(pts3d, camera_matrix)
        tck_reprojected, u_reprojected = curve.fit_spline(pts3d_reprojected)

        pts = curve.sample_spline(tck, u, delta=delta)
        pts_reprojected = curve.sample_spline(
            tck_reprojected, u_reprojected, delta=delta
        )

        min_len = min(len(pts), len(pts_reprojected))
        pts = pts[:min_len]
        pts_reprojected = pts_reprojected[:min_len]

        return img, pts, pts_reprojected

    def pad_with_nan(gts, pds):
        def pad_to_len(arr, length):
            padding_length = length - len(arr)
            for i in range(padding_length):
                arr = np.vstack([arr, np.nan * np.ones(arr.shape[1])])
            return arr

        all_arrays = gts + pds
        max_length = max(len(arr) for arr in all_arrays)
        padded_gts = [pad_to_len(gt, max_length) for gt in gts]
        padded_pds = [pad_to_len(pd, max_length) for pd in pds]

        return padded_gts, padded_pds

    imgs = []
    ground_truth = []
    reprojected = []
    for frame in data:
        img, gt, rp = process_frame(frame)
        if not is_valid_frame(rp):
            continue
        imgs.append(img)
        ground_truth.append(gt)
        reprojected.append(rp)

    ground_truth, reprojected = pad_with_nan(ground_truth, reprojected)

    ground_truth = np.array(ground_truth)
    reprojected = np.array(reprojected)
    print("Ground truth shape:", ground_truth.shape)
    print("Reprojected shape:", reprojected.shape)

    # Calculate the Euclidean distance between the ground truth and reprojected points
    distance = np.linalg.norm(ground_truth - reprojected, axis=2)
    print("Distance shape:", distance.shape)
    #
    # Calculate the mean and standard deviation of these distances pointwise
    mean_errors = np.nanmean(distance, axis=0)
    std_errors = np.nanstd(distance, axis=0)

    print("Mean errors shape:", mean_errors.shape)
    print("Std errors shape:", std_errors.shape)

    # Find the maximum error and its index for each point
    max_errors = np.nanmax(distance, axis=1)
    max_error_indices = np.nanargmax(distance, axis=1)

    print("Max errors shape:", max_errors.shape)
    print("Max error indices shape:", max_error_indices.shape)

    # Sort the maximum errors in descending order and get the sorted indices
    sorted_indices = np.argsort(-max_errors)
    sorted_max_errors = max_errors[sorted_indices]
    sorted_max_error_indices = max_error_indices[sorted_indices]

    print("Sorted max errors shape:", sorted_max_errors.shape)
    print("Sorted max error indices shape:", sorted_max_error_indices.shape)
    # exit()

    for i in sorted_indices[:5]:
        print(f"Index {i}, max error: {max_errors[i]}")
        gt = ground_truth[i]
        rp = reprojected[i]
        print(f"Max error for frame {i}: {max_errors[i]}")
        print("img path:", vars.dataset_path / imgs[i])
        img = plt.imread(vars.dataset_path / imgs[i])
        plot_reprojected(img, gt, rp, show=True)

    # print("Mean errors:", mean_errors.shape)
    # print("Std errors:", std_errors.shape)
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

    max_len = min(len(mean_errorsA), len(mean_errorsB))
    mean_errorsA = mean_errorsA[:max_len]
    mean_errorsB = mean_errorsB[:max_len]
    std_errorsA = std_errorsA[:max_len]
    std_errorsB = std_errorsB[:max_len]

    fig, ax = setup_fig()

    ax_props = dict(
        capsize=1,
        capthick=1,
        elinewidth=1,
        markersize=2,
        fmt="o",
    )

    ax.errorbar(
        np.arange(max_len),
        mean_errorsA,
        yerr=std_errorsA,
        label="Camera A",
        **ax_props,
    )
    ax.errorbar(
        np.arange(max_len),
        mean_errorsB,
        yerr=std_errorsB,
        label="Camera B",
        **ax_props,
    )
    plt.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.show()
    fig.savefig("figs/reconstruction_reprojection_error.png")
    plt.close()


def clean_data(data: List[Dict]):
    invalid = 0
    cleaned_data = []
    for frame in data:
        ptsA = np.column_stack(splev(frame["uA"], frame["tckA"])).astype(np.int32)
        if np.any(ptsA > 1024) or np.any(ptsA < 0):
            invalid += 1
            continue
        ptsB = np.column_stack(splev(frame["uB"], frame["tckB"])).astype(np.int32)
        if np.any(ptsB > 1024) or np.any(ptsB < 0):
            invalid += 1
            continue
        cleaned_data.append(frame)
    print("Invalid frames:", invalid)
    return cleaned_data


def main():
    data = get_data()
    print(len(data))
    make_reprojection_error_plot(data)
    pass


if __name__ == "__main__":
    main()
