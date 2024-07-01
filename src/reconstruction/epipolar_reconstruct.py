from typing import List, Tuple

import calibration
import cv2
import numpy as np
import representations.curve as curve
import scipy.optimize as opt
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
i = 0


def get_data():
    import json

    frames = [100]
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


def find_closest_u_on_spline_to_line(ln, tck, u_min, u_max):
    def distance_u_to_line(u):
        x0, y0 = splev(u, tck)

        # Parameters of the line ax + by + c = 0
        a, b, c = ln

        # Calculate the distance from the point to the line
        distance = np.abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)

        return distance

    def optimization_function(u):
        return distance_u_to_line(u) + np.abs(u - u_min)

    # Minimize the distance function
    result = opt.minimize(distance_u_to_line, u_min, bounds=[(u_min, u_max)])

    if result.success:
        return result.x[0]
    else:
        raise ValueError("Optimization failed")


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


def find_match(pt, tck, u_min, u_max, F, delta=50, eps=100):
    # Compute the epipolar line
    ln = (
        cv2.computeCorrespondEpilines(pt.reshape(-1, 1, 2), 1, F)
        .reshape(-1, 3)
        .squeeze()
    )

    # Find line extremities
    extremities = fn.find_line_extremities(ln, 1024, 1024)

    # Generate spline segments
    u_values = np.linspace(u_min, u_max, int(u_max / 100) + 1)
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

    if np.abs(u_intersection - u_min) > eps:
        closest_u = find_closest_u_on_spline_to_line(
            ln, tck, min(u_min + 20, u_max), u_max
        )
        if np.abs(closest_u - u_min) > eps or np.abs(closest_u - u_min) < 5:
            return None
        else:
            u_intersection = closest_u
        return None

    return u_intersection


def match_points(
    tck1: tuple, tck2: tuple, uA: list, uB: list, n: int, delta: float, F: np.ndarray
):
    uB_min = 0
    uB_max = min(uA[-1], uB[-1])
    sampling_spacing = int(uB_max / delta) + 1 if delta else n

    uAs = np.linspace(0, uB_max, sampling_spacing)
    u_pairs = [0]
    u_matches = [0]
    for uA in uAs[1:]:
        if uB_min >= uB_max:
            break
        pt = np.column_stack(splev(uA, tck1))
        u_match = find_match(pt, tck2, uB_min, uB_max, F)
        u_matches.append(u_match)
        u_pairs.append(uA)
        if u_match is not None:
            uB_min = u_match

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
    uA: list,
    uB: list,
    n: int = None,
    delta: float = None,
    others: dict = {},
    visualize: bool = False,
    s: float = None,
):
    global i
    try:
        u_max = min(uA[-1], uB[-1])
        # find matching u values
        uA, uA_matches = match_points(
            tckA, tckB, uA, uB, n=n, delta=delta, F=calibration.F_A_B
        )

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
            u2, u2_mathces = match_points(
                tckB, tckA, uB, uA, n=n, delta=delta, F=calibration.F_B_A
            )
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
                i,
            )

        i += 1
        reconstructed_pts = triangulation.reconstruct(
            ptsA, ptsB, calibration.P1, calibration.P2, calibration.F_A_B
        )

        reconstructed_pts = remove_duplicates(reconstructed_pts)

        tck3d, u3D = curve.fit_spline(reconstructed_pts, s=s, k=5)

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
            plt.show()
            plt.close()

        return tck3d, u3D
    except Exception as e:
        print("Error in get_curve_3d:", e)
        return None, None


reconstruct = get_curve_3d


def main():
    import vars

    dataset_path = vars.dataset_path

    samples = get_data()[:4]

    for i, sample in enumerate(samples):
        pts1 = np.array(sample["pts1"])
        pts2 = np.array(sample["pts2"])

        tck1, u1 = curve.fit_spline(pts1)
        tck2, u2 = curve.fit_spline(pts2)

        u_max = min(u1[-1], u2[-1])
        pts1 = curve.sample_spline(tck1, u1, delta=20)
        pts2 = curve.sample_spline(tck2, u2, delta=20)

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        img1 = viz.convert_to_color(img1)
        img2 = viz.convert_to_color(img2)

        print("\nSample", i)
        get_curve_3d(
            tck1,
            tck2,
            u1,
            u2,
            delta=20,
            others={
                "img1": img1,
                "img2": img2,
                "tck1": tck1,
                "tck2": tck2,
                "u1": u1,
                "u2": u2,
            },
            visualize=True,
        )
        # exit()


if __name__ == "__main__":
    main()
