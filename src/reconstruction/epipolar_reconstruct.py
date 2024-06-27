from typing import List, Tuple

import calibration
import cv2
import numpy as np
import representations.curve as curve
import utils.fn as fn
import utils.plot as plot
import utils.viz as viz
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator, splev
from scipy.optimize import minimize_scalar

import reconstruction.epipolar.fig as ep_fig
import reconstruction.epipolar.plot as ep_plot
import reconstruction.epipolar.utils as ep_utils
import reconstruction.epipolar.viz as ep_viz
from reconstruction import triangulation

ep_viz = ep_viz


def get_data():
    import json

    frames = [247]
    dummy_data = []
    annotations = json.load(open("data/annotations/raw.json"))
    for video in annotations:
        for frame in video["frames"]:
            if frame["frame_number"] in frames:
                dummy_data.append(
                    {
                        "img1": frame["camera1"]["image"],
                        "img2": frame["camera2"]["image"],
                        "pts1": frame["camera1"]["points"],
                        "pts2": frame["camera2"]["points"],
                    }
                )
    return dummy_data


def interpolate_matches(u, u_match):
    assert isinstance(u, np.ndarray) and isinstance(
        u_match, np.ndarray
    ), f"u and u_match must be a numpy array, but got {type(u)}\n"

    # Filter out NaN values in u_match
    u_valid = u[~np.isnan(u_match)]
    u_match_valid = u_match[~np.isnan(u_match)]

    # Fit a monotonic function using the valid points
    monotonic_function = PchipInterpolator(u_valid, u_match_valid)

    # Interpolate the missing values
    u_match_interpolated = np.copy(u_match)
    for i, x in enumerate(u_match):
        if np.isnan(x):
            u_match_interpolated[i] = monotonic_function(u[i])

    return u_match_interpolated


def find_match(pt, tck, u_min, u_max, F):
    # Compute the epipolar line
    ln = (
        cv2.computeCorrespondEpilines(pt.reshape(-1, 1, 2), 1, F)
        .reshape(-1, 3)
        .squeeze()
    )

    # Find line extremities
    extremities = fn.find_line_extremities(ln, 1024, 1024)

    # Generate spline segments
    u_values = np.linspace(u_min, u_max, 1000)
    segments = np.column_stack(splev(u_values, tck))

    # Find intersections
    intersections = ep_utils.find_intersections(extremities, segments, verbose=False)

    if not intersections:
        return None

    # Take the first intersection point
    intersection = intersections[0]

    # Function to compute the squared distance between a point and the spline at parameter u
    def distance_squared(u, point):
        x, y = splev(u, tck)
        return (x - point[0]) ** 2 + (y - point[1]) ** 2

    # Find the best u for the intersection point using minimize_scalar with bounded method
    res = minimize_scalar(
        distance_squared, args=(intersection,), bounds=(u_min, u_max), method="bounded"
    )

    u_intersection = res.x

    return u_intersection


def match_points(tck1: tuple, tck2: tuple, u_max: float, n: int, F: np.ndarray):
    u_min = 0
    us = np.linspace(u_min, u_max, n)
    u_pairs = [0]
    u_matches = [0]
    for u in us[1:]:
        if u_min >= u_max:
            break
        pt = np.column_stack(splev(u, tck1))
        u_match = find_match(pt, tck2, u_min, u_max, F)
        u_matches.append(u_match)
        u_pairs.append(u)
        if u_match is not None:
            u_min = u_match

    # go in reverse order and eliminate None values at the end
    # these are outside the image
    for i in range(len(u_matches) - 1, 0, -1):
        if u_matches[i] is None:
            u_matches.pop(i)
            u_pairs.pop(i)
        else:
            break
    return np.array(u_pairs, dtype=float), np.array(u_matches, dtype=float)


def remove_duplicates(pts):
    # Use a set to track seen rows and maintain order
    seen = set()
    unique_rows = []
    for row in pts:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)

    # Convert the list of unique rows back to a numpy array
    unique_array = np.array(unique_rows)
    return unique_array


def get_curve_3d(
    tckA: Tuple[np.ndarray, List[np.ndarray], int],
    tckB: Tuple[np.ndarray, List[np.ndarray], int],
    u_max: float,
    n: int = 50,
    others: dict = {},
    visualize: bool = False,
):
    try:
        # find matching u values
        uA, uA_matches = match_points(tckA, tckB, u_max, n, calibration.F_A_B)

        # interpolate_matches
        uA_matches = interpolate_matches(uA, uA_matches)

        # sample points at u values
        ptsA = np.column_stack(splev(uA, tckA))
        ptsA_matches = np.column_stack(splev(uA_matches, tckB))

        ptsA = ptsA
        ptsB = ptsA_matches

        if visualize:
            img1 = others["img1"]
            img2 = others["img2"]
            u2, u2_mathces = match_points(tckB, tckA, u_max, n, calibration.F_B_A)
            u2_mathces = interpolate_matches(u2, u2_mathces)
            pts2 = np.column_stack(splev(u2, tckB))
            pts2_matches = np.column_stack(splev(u2_mathces, tckA))
            ep_plot.plot_with_matches(
                ptsA, pts2, ptsA_matches, pts2_matches, img1, img2
            )
            ep_fig.make_epilines_plot(
                ptsA,
                ptsA_matches,
                img1,
                img2,
                calibration.F_A_B,
                tckA,
                others["u1"],
                tckB,
                others["u2"],
            )

        reconstructed_pts = triangulation.reconstruct(
            ptsA, ptsB, calibration.P1, calibration.P2, calibration.F_A_B
        )

        reconstructed_pts = remove_duplicates(reconstructed_pts)

        tck3d, u3D = curve.fit_spline(reconstructed_pts, s=0.5)

        if visualize:
            reconstructed_pts = np.column_stack(splev(u3D, tck3d))
            # ep_plot.plot_reprojection(img1, tckA, u_max, tck3d, u3D[-1], calibration.P1)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                reconstructed_pts[:, 0],
                reconstructed_pts[:, 1],
                reconstructed_pts[:, 2],
                marker="o",
                markersize=1,
                linestyle="-",
            )
            ax.view_init(5, 180)
            plot.plot_mesh(plot.get_mesh(), ax)
            # plt.show()
            plt.close()

        return tck3d, u3D
    except Exception as e:
        print("Error in get_curve_3d:", e)
        return None, None


reconstruct = get_curve_3d


def main():
    import vars

    dataset_path = vars.dataset_path

    samples = get_data()[:1]

    for i, sample in enumerate(samples):
        pts1 = np.array(sample["pts1"])
        pts2 = np.array(sample["pts2"])

        tck1, u1 = curve.fit_spline(pts1, s=0.2)
        tck2, u2 = curve.fit_spline(pts2, s=0.2)

        u_max = min(u1[-1], u2[-1])
        pts1 = curve.sample_curve(tck1, 0, u_max, 5)
        pts2 = curve.sample_curve(tck2, 0, u_max, 5)

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        img1 = viz.convert_to_color(img1)
        img2 = viz.convert_to_color(img2)

        print("\nSample", i)
        get_curve_3d(
            tck1,
            tck2,
            u_max,
            10,
            {
                "img1": img1,
                "img2": img2,
                "tck1": tck1,
                "tck2": tck2,
                "u1": u1,
                "u2": u2,
            },
            True,
        )


if __name__ == "__main__":
    main()
