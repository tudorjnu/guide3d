import cv2
import guide3d.representations.curve as curve
import guide3d.utils.utils as utils
import guide3d.utils.viz as viz
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import PchipInterpolator, splev
from shapely.geometry import LineString


def find_line_extremities(line, width, height):
    """Given a line in the form (a, b, c), find the two points that define the line"""
    a, b, c = line

    if b != 0:
        y_left = -c / b
        y_right = -(a * width + c) / b
    else:
        y_left = 0
        y_right = height

    y_left = max(0, min(height, y_left))
    y_right = max(0, min(height, y_right))

    pt1 = (0, int(y_left))
    pt2 = (width, int(y_right))
    return np.array([pt1, pt2]).astype(np.int32)


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
        return None


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


def find_intersections(segment1, segment2, verbose=False):
    polyline1 = LineString(segment1)
    polyline2 = LineString(segment2)

    intersection = polyline1.intersection(polyline2)

    if intersection.is_empty:
        if verbose:
            print("No intersection found")
        return None
    elif intersection.geom_type == "MultiPoint":
        pts = [np.array((point.x, point.y)) for point in intersection.geoms]
        if verbose:
            print("Number of intersections:", len(pts))
        return pts
    elif intersection.geom_type == "Point":
        pts = [np.array((intersection.x, intersection.y))]
        if verbose:
            print("Number of intersections:", 1)
        return pts
    else:
        return None


def find_match(pt, tck, u_min, u_max, F, delta=50, eps=100):
    # Compute the epipolar line
    ln = (
        cv2.computeCorrespondEpilines(pt.reshape(-1, 1, 2), 1, F)
        .reshape(-1, 3)
        .squeeze()
    )

    # Find line extremities
    extremities = find_line_extremities(ln, 1024, 1024)

    # Generate spline segments
    u_values = np.linspace(u_min, u_max, int(u_max / 100) + 1)
    segments = np.column_stack(splev(u_values, tck))

    # Find intersections
    intersections = find_intersections(extremities, segments, verbose=False)

    if not intersections:
        return None

    # Take the first intersection point
    intersection = intersections[0]

    # Function to compute the squared distance between a point and the spline at parameter u
    def distance_squared(u, point):
        x, y = splev(u, tck)
        return (x - point[0]) ** 2 + (y - point[1]) ** 2

    # Find the best u for the intersection point using minimize_scalar with bounded method
    res = opt.minimize_scalar(
        distance_squared, args=(intersection,), bounds=(u_min, u_max), method="bounded"
    )

    u_intersection = res.x

    if np.abs(u_intersection - u_min) > eps:
        closest_u = find_closest_u_on_spline_to_line(ln, tck, min(u_min, u_max), u_max)
        if closest_u is None:
            return None
        elif np.abs(closest_u - u_min) > eps or np.abs(closest_u - u_min) < 5:
            return None
        else:
            u_intersection = closest_u
        return None

    return u_intersection


def match_points(
    tck1: tuple,
    tck2: tuple,
    uA: list,
    uB: list,
    n: int = None,
    delta: float = None,
    F: np.ndarray = None,
):
    assert not (n is None and delta is None), "Either n or delta must be provided\n"
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


def main():
    import matplotlib.pyplot as plt
    import vars

    dataset_path = vars.dataset_path

    samples = utils.get_data()[:4]

    for i, sample in enumerate(samples):
        pts1 = np.array(sample["pts1"])
        pts2 = np.array(sample["pts2"])

        tck1, u1 = curve.fit_spline(pts1)
        tck2, u2 = curve.fit_spline(pts2)

        pts1 = curve.sample_spline(tck1, u1, delta=20)
        pts2 = curve.sample_spline(tck2, u2, delta=20)

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        img1 = viz.convert_to_color(img1)
        img2 = viz.convert_to_color(img2)


if __name__ == "__main__":
    main()
