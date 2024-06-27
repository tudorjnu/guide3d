import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString


def project_points(
    points: np.ndarray,
    camera_matrix: np.ndarray = None,
) -> np.ndarray:
    if points.ndim == 1:
        points = points[np.newaxis, :]

    # Making points homogeneous
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Projecting to pixel coordinates
    pixel_coords = camera_matrix.dot(points_homogeneous.T)

    # Converting homogeneous coordinates to Cartesian coordinates
    pixel_coords = pixel_coords / pixel_coords[-1, :]
    pixel_coords = np.round(pixel_coords[:-1, :].T).astype(np.int32)

    return pixel_coords.squeeze()


def triangulate(P1, P2, x1s, x2s):
    N = x1s.shape[0]
    Xs = np.zeros((N, 4))

    for i in range(N):
        x1 = x1s[i]
        x2 = x2s[i]

        A = np.zeros((4, 4))
        A[0:2, :] = x1[0] * P1[2, :] - P1[0, :]
        A[2:4, :] = x1[1] * P1[2, :] - P1[1, :]

        B = np.zeros((4, 4))
        B[0:2, :] = x2[0] * P2[2, :] - P2[0, :]
        B[2:4, :] = x2[1] * P2[2, :] - P2[1, :]

        # Stacking A and B to form a 4x4 matrix
        A = np.vstack((A, B))

        # Solve for X by minimizing ||AX|| subject to ||X|| = 1
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        Xs[i, :] = X / X[-1]  # Dehomogenize (make last element 1)

    return Xs


def polyline_to_mask(polyline, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = cv2.polylines(mask, [polyline], isClosed=False, color=255, thickness=1)
    return mask


def interpolate_even_spacing(chain, spacing):
    # Ensure the tip of the guidewire is the starting point
    new_chain = [chain[0]]

    # Calculate distances between consecutive points and cumulative lengths
    distances = np.sqrt(np.sum(np.diff(chain, axis=0) ** 2, axis=1))
    cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_lengths[-1]

    # Determine the number of points based on the total length and spacing
    num_points = int(np.floor(total_length / spacing))

    for i in range(
        1, num_points + 1
    ):  # Start from 1 to keep the tip as the first point
        target_length = i * spacing
        # Find the segment where the target_length falls
        for j in range(1, len(cumulative_lengths)):
            if cumulative_lengths[j] >= target_length:
                # Calculate the exact position on this segment
                segment_start_index = j - 1
                segment_end_index = j
                segment_start_length = cumulative_lengths[segment_start_index]
                segment_end_length = cumulative_lengths[segment_end_index]

                # Interpolate within this segment
                segment_ratio = (target_length - segment_start_length) / (
                    segment_end_length - segment_start_length
                )

                new_point = chain[segment_start_index] + segment_ratio * (
                    chain[segment_end_index] - chain[segment_start_index]
                )
                new_chain.append(new_point)
                break

    return np.array(new_chain)


def get_smooth_curve(chain: np.ndarray, n_points=40):
    from scipy.interpolate import CubicSpline

    x = chain[:, 0]
    y = chain[:, 1]

    # Create a range of 't' values
    t = np.arange(len(chain))

    # Fit splines
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    # Evaluate splines over a smooth range of 't'
    t_smooth = np.linspace(t.min(), t.max(), n_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)

    return np.column_stack([x_smooth, y_smooth])


def calculate_segment_length(points):
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    total_length = np.sum(distances)
    return total_length


def fit_spline_3d(points, n_points=40):
    from scipy.interpolate import CubicSpline

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    t = np.arange(len(points))
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)

    t_smooth = np.linspace(t.min(), t.max(), n_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)

    return np.column_stack([x_smooth, y_smooth, z_smooth])


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


def generate_colors(n):
    return np.random.randint(0, 255, (n, 3))


if __name__ == "__main__":
    import plot
    import vars
    from annot_parser import get_structured_dataset
    from calibration import P1, P2
    from reconstruct import get_points

    test_dataset_path = vars.test_dataset_path
    dataset = get_structured_dataset(test_dataset_path / "annotations.xml")
    sample = dataset[0]

    img1 = plt.imread((test_dataset_path / sample["image1"]))
    img2 = plt.imread((test_dataset_path / sample["image2"]))
    points1 = sample["points1"]
    points2 = sample["points2"]

    p1, p2, p3d = get_points(points1, points2, P1, P2, spacing=15)

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    plot.make_3d_plot(p3d, ax=ax)
    plt.show()
