from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize

import fn
import reconstruct
import viz


def refine_fundamental_matrix(F_initial, pts1, pts2):
    def sampson_distance(F, pts1, pts2):
        """Compute the Sampson distance between points and epipolar lines."""
        Fx1 = np.dot(F, pts1.T)  # Transform pts1 by F
        Fx2 = np.dot(F.T, pts2.T)  # Transform pts2 by F.T
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        numer = np.sum(pts2.T * np.dot(F, pts1.T), axis=0) ** 2
        return numer / denom

    def objective_func(flat_F, pts1_homo, pts2_homo):
        """Objective function to minimize."""
        F = flat_F.reshape(3, 3)
        return sampson_distance(F, pts1_homo, pts2_homo)

    # Convert points to homogeneous coordinates
    pts1_homo = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_homo = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    # Flatten the initial estimate of the fundamental matrix for the optimizer
    flat_F_initial = F_initial.flatten()

    # Perform the optimization
    result = least_squares(
        objective_func, flat_F_initial, args=(pts1_homo, pts2_homo), ftol=1e-10
    )

    # Reshape the optimized result back into a 3x3 matrix
    F_optimized = result.x.reshape(3, 3)

    # Enforce the singularity constraint (det(F) = 0) on F_optimized
    U, S, Vt = np.linalg.svd(F_optimized)
    S[2] = 0  # Set the smallest singular value to zero
    F_refined = np.dot(U, np.dot(np.diag(S), Vt))

    return F_refined


def show_calibraion_image(path):
    img = cv2.imread(path, 0)
    plt.imshow(img, cmap="gray")
    plt.show()


def show_points(pts1, pts2, img1, img2, points_labels):
    fig, ax = plt.subplots(1, 2, figsize=(3, 6))
    ax[0].imshow(img1, cmap="gray")
    ax[0].scatter(pts1[:, 0], pts1[:, 1], s=1, c="r")
    ax[0].set_title("Image 1")
    ax[0].axis("off")
    for i, (x, y) in enumerate(pts1):
        y -= 10
        ax[0].text(x, y, points_labels[i], fontsize=4, color="r")

    ax[1].imshow(img2, cmap="gray")
    ax[1].scatter(pts2[:, 0], pts2[:, 1], s=1, c="r")
    ax[1].set_title("Image 2")
    ax[1].axis("off")
    for i, (x, y) in enumerate(pts2):
        y -= 10
        ax[1].text(x, y, points_labels[i], fontsize=4, color="r")

    plt.show()


def find_closest_point(line, pts2, previous_point=None, alpha=0.5, min_dist_to_prev=5):
    a, b, c = line
    # Compute perpendicular distances to the epipolar line
    distances_to_line = np.abs(a * pts2[:, 0] + b * pts2[:, 1] + c) / np.sqrt(
        a**2 + b**2
    )

    # Normalize distances
    distances_to_line_normalized = distances_to_line / np.max(distances_to_line)

    if previous_point is not None:
        distances_to_prev_point = np.linalg.norm(pts2 - previous_point, axis=1)
        distances_to_prev_point_normalized = distances_to_prev_point / np.max(
            distances_to_prev_point
        )
    else:
        distances_to_prev_point_normalized = np.zeros_like(distances_to_line_normalized)

    # Combine distances with weighting
    combined_distances = (
        alpha * distances_to_line_normalized
        + (1 - alpha) * distances_to_prev_point_normalized
    )

    # Ensure the selected point is different from the previous by enforcing a minimum distance
    if previous_point is not None:
        for i, dist in enumerate(distances_to_prev_point):
            if dist < min_dist_to_prev:
                combined_distances[i] = np.inf  # Effectively exclude this point

    min_index = np.argmin(combined_distances)
    if combined_distances[min_index] == np.inf:
        return None  # No suitable point found

    return pts2[min_index]


def compute_epipolar_lines(points1, F):
    """Compute all epipolar lines for points in the first image."""
    # Ensure points are in the correct shape for OpenCV
    points1_reshaped = points1.reshape(-1, 1, 2)
    # Compute epipolar lines in the second image for points from the first image
    lines_in_img2 = cv2.computeCorrespondEpilines(points1_reshaped, 1, F).reshape(-1, 3)
    return lines_in_img2


def line_segment_intersection(line, P1, P2):
    """
    Find intersection of a line (ax + by + c = 0) with a segment (P1, P2).
    Returns the intersection point if it exists, None otherwise.
    """
    a, b, c = line
    x1, y1 = P1
    x2, y2 = P2

    # Compute the denominator to avoid division by zero
    denominator = a * (x2 - x1) + b * (y2 - y1)
    if denominator == 0:
        return None  # Line and segment are parallel or coincident

    # Compute t where the intersection occurs
    t = -(a * x1 + b * y1 + c) / denominator

    if 0 <= t <= 1:
        # Compute the actual intersection point
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y])
    else:
        return None


def find_intersection(polyline, line):
    for i in range(len(polyline) - 1):
        intersection = line_segment_intersection(line, polyline[i], polyline[i + 1])
        if intersection is not None:
            return intersection, polyline[i + 1 :]
    return None, polyline


def find_match(points1, points2, F):
    lines_in_img2 = compute_epipolar_lines(points1, F)
    pts1_returned = []
    pts2_returned = []

    for i, point1 in enumerate(points1):
        line = lines_in_img2[i]
        if i == 0:
            match = points2[0]
        else:
            match, points2 = find_intersection(points2, line)

        if match is not None and len(points2) != 0:
            pts1_returned.append(point1)
            pts2_returned.append(match)

    print(f"Matched {len(pts1_returned)} points")
    return np.array(pts1_returned), np.array(pts2_returned)


def order_matches(pts1, pts2):
    # order the matching points
    initial_point = pts1[0]
    ordered_pts = [initial_point]
    for i in range(1, len(pts1)):
        d_1 = np.linalg.norm(pts1[i] - initial_point)
        pt2 = pts2[0]
        d_2 = np.linalg.norm(pt2 - initial_point)
        if d_1 < d_2:
            ordered_pts.append(pts1[i])
        else:
            ordered_pts.append(pt2)
            pts2 = np.delete(pts2, 0, axis=0)
    return np.array(ordered_pts)


def get_points(
    resampled1: np.ndarray,
    resampled2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    spacing: int = 10,
    F: np.ndarray = None,
):
    resampled1 = reconstruct.order_points(resampled1)
    resampled2 = reconstruct.order_points(resampled2)

    resampled1 = fn.interpolate_even_spacing(resampled1, spacing=spacing)
    resampled2 = fn.interpolate_even_spacing(resampled2, spacing=spacing)
    min_length = min(len(resampled1), len(resampled2))
    resampled1 = resampled1[:min_length]
    resampled2 = resampled2[:min_length]

    pts1, pts2 = resampled1, resampled2
    # pts2, pts1 = find_match(resampled2, resampled1, F)

    # pts1, pts2 = refine_point_correspondences(pts1, pts2, F)

    points_3d_h = fn.triangulate(P1, P2, pts1, pts2)

    points_3d = points_3d_h[:, :3] / points_3d_h[:, 3, np.newaxis]

    assert (
        pts1.shape == pts2.shape and pts2.shape[0] == points_3d.shape[0]
    ), f"points1.shape = {pts1.shape}, points2.shape = {pts2.shape}, points_3d.shape = {points_3d.shape}"

    return resampled1, resampled2, points_3d


def drawlines_single(img, lines, pts):
    pts = pts.astype(int)
    r, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for r, pt in zip(lines, pts):
        color = (255, 255, 255)
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv2.circle(img, tuple(pt), 5, color, -1)
    return img


def draw_line(img, line, pt):
    pt = pt.astype(int)
    r, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = 1
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    img = cv2.circle(img, tuple(pt), 5, color, -1)
    return img


def get_F_matrix():
    pts1, pts2 = get_p()

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return F, (pts1, pts2)


def refine_points(points, image):
    points_dtype = points.dtype
    binary_image = image < 80

    refined_points = []
    for point in points:
        x, y = point

        # Define the local neighborhood to search for the bead's center
        size = 10
        x_min, x_max = max(0, x - size), min(image.shape[1], x + size)
        y_min, y_max = max(0, y - size), min(image.shape[0], y + size)

        # Extract the local neighborhood
        local_area = binary_image[y_min:y_max, x_min:x_max]

        # Calculate the centroid of the black pixels in the local area
        if local_area.any():
            ys, xs = np.nonzero(local_area)
            centroid_x = xs.mean() + x_min
            centroid_y = ys.mean() + y_min
            refined_point = (int(centroid_x), int(centroid_y))
        else:
            refined_point = point

        refined_points.append(refined_point)

    return np.array(refined_points, dtype=points_dtype)


def get_p():
    points_labels = [43, 27, 40, 60, 26, 63, 58, 15, 38, 44]

    pts1 = np.array(
        [
            [579, 526],
            [870, 520],
            [804, 182],
            [293, 165],
            [865, 855],
            [124, 503],
            [270, 889],
            [951, 492],
            [774, 932],
            [599, 180],
        ],
        dtype=np.int32,
    )

    pts2 = np.array(
        [
            [684, 527],
            [500, 502],
            [977, 194],
            [921, 184],
            [494, 820],
            [601, 544],
            [902, 912],
            [67, 489],
            [957, 856],
            [707, 190],
        ],
        dtype=np.int32,
    )

    img1_path = (
        Path.cwd()
        / "src/calibration/"
        / "2024-01-18"
        / "pre-experiment"
        / "Calibration1_16709.jpg"
    )
    img2_path = (
        Path.cwd()
        / "src/calibration/"
        / "2024-01-18"
        / "pre-experiment"
        / "Calibration1_16710.jpg"
    )

    img1 = cv2.imread(img1_path.as_posix(), 0)
    img2 = cv2.imread(img2_path.as_posix(), 0)

    fig, ax = plt.subplots(2, 1, figsize=(3, 6))
    ax[0].imshow(img1, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("16709")

    for i, (x, y) in enumerate(pts1):
        y -= 10
        ax[0].text(x, y, points_labels[i], fontsize=4, color="r")

    ax[1].imshow(img2, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("16710")
    for i, (x, y) in enumerate(pts2):
        y -= 10
        ax[1].text(x, y, points_labels[i], fontsize=4, color="r")

    plt.show()
    exit()

    # pts1 = refine_points(pts1, img1)
    # pts2 = refine_points(pts2, img2)

    # show_points(pts1, pts2, img1, img2, points_labels)
    return pts1, pts2


def test():
    points_labels = [43, 27, 40, 60, 26, 63, 58, 15]
    pts1, pts2 = get_p()

    img1_path = (
        Path.cwd()
        / "src/calibration/"
        / "2024-01-18"
        / "pre-experiment"
        / "Calibration1_16709.jpg"
    )
    img2_path = (
        Path.cwd()
        / "src/calibration/"
        / "2024-01-18"
        / "pre-experiment"
        / "Calibration1_16710.jpg"
    )

    img1 = cv2.imread(img1_path.as_posix(), 0)
    img2 = cv2.imread(img2_path.as_posix(), 0)

    show_points(pts1, pts2, img1, img2, points_labels)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS, 0.1, 0.99)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    img1_with_lines = drawlines_single(img1, lines1, pts1)
    img2_with_lines = drawlines_single(img2, lines2, pts2)

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(img1_with_lines)
    ax[0].set_title("Image 1")
    ax[0].axis("off")
    ax[1].imshow(img2_with_lines)
    ax[1].set_title("Image 2")
    ax[1].axis("off")
    plt.show()


def refine_point_correspondences(pts1, pts2, F):
    """
    Refine point correspondences to better satisfy the epipolar constraint.
    pts1, pts2: Original matching points in the two images.
    F: Fundamental matrix.
    Returns refined points pts1_refined, pts2_refined.
    """

    # Function to compute the distance of a point from a line
    def point_line_distance(point, line):
        x, y = point[0], point[1]
        a, b, c = line[0], line[1], line[2]
        return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    # Optimization function to minimize
    def optimize_points(x, pts1, pts2, F):
        num_points = pts1.shape[0]
        x1_refined = x[: 2 * num_points].reshape(-1, 2)
        x2_refined = x[2 * num_points :].reshape(-1, 2)

        total_distance = 0
        for i in range(num_points):
            l2 = np.dot(F, np.append(pts1[i], 1))
            l1 = np.dot(F.T, np.append(pts2[i], 1))
            total_distance += point_line_distance(x1_refined[i], l1) ** 2
            total_distance += point_line_distance(x2_refined[i], l2) ** 2
        return total_distance

    # Initial guess for the optimization (flatten the point arrays)
    x0 = np.hstack((pts1.flatten(), pts2.flatten()))

    # Perform optimization
    result = minimize(optimize_points, x0, args=(pts1, pts2, F), method="L-BFGS-B")

    # Extract refined points
    refined = result.x.reshape(2, -1).T
    pts1_refined = refined[: pts1.shape[0]]
    pts2_refined = refined[pts1.shape[0] :]

    return pts1_refined, pts2_refined


def plot_points_w_lines(F):
    points_labels = [43, 27, 40, 60, 26, 63, 58, 15, 38, 44]

    pts1 = np.array(
        [
            [579, 526],
            [870, 520],
            [804, 182],
            [293, 165],
            [865, 855],
            [124, 503],
            [270, 889],
            [951, 492],
            [774, 932],
            [599, 180],
        ],
        dtype=np.int32,
    )

    pts2 = np.array(
        [
            [684, 527],
            [500, 502],
            [977, 194],
            [921, 184],
            [494, 820],
            [601, 544],
            [902, 912],
            [67, 489],
            [957, 856],
            [707, 190],
        ],
        dtype=np.int32,
    )

    img1_path = (
        Path.cwd()
        / "src/calibration/"
        / "2024-01-18"
        / "pre-experiment"
        / "Calibration1_16709.jpg"
    )
    img2_path = (
        Path.cwd()
        / "src/calibration/"
        / "2024-01-18"
        / "pre-experiment"
        / "Calibration1_16710.jpg"
    )

    img1 = cv2.imread(img1_path.as_posix(), 0)
    img2 = cv2.imread(img2_path.as_posix(), 0)

    if d1 is not None and d2 is not None:
        print("Undistorting images")
        print(d1, d2)
        img1 = cv2.undistort(img1, K1, d1)
        img2 = cv2.undistort(img2, K2, d2)

    pts1 = refine_points(pts1, img1)
    pts2 = refine_points(pts2, img2)

    img1 = viz.convert_to_color(img1)
    img2 = viz.convert_to_color(img2)

    random_colors = np.random.randint(0, 255, (len(pts1), 3), dtype=np.uint8)

    lines_in_img1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(
        -1, 3
    )
    lines_in_img2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(
        -1, 3
    )

    img1_with_points = viz.draw_points(img1, pts1, list(random_colors))
    img2_with_points = viz.draw_points(img2, pts2, list(random_colors))

    img1_with_lines = viz.draw_lines(img1_with_points, lines_in_img1, random_colors)
    img2_with_lines = viz.draw_lines(img2_with_points, lines_in_img2, random_colors)

    # print(img1_with_lines.shape, img2_with_lines.shape)
    # print(img1_with_lines.dtype, img2_with_lines.dtype)

    fig, ax = plt.subplots(2, 1, figsize=(3, 6))
    ax[0].imshow(img1_with_lines)
    ax[0].axis("off")
    ax[0].set_title("16709")

    for i, (x, y) in enumerate(pts1):
        y -= 10
        ax[0].text(x, y, points_labels[i], fontsize=4, color="r")

    ax[1].imshow(img2_with_lines)
    ax[1].axis("off")
    ax[1].set_title("16710")
    for i, (x, y) in enumerate(pts2):
        y -= 10
        ax[1].text(x, y, points_labels[i], fontsize=4, color="r")

    plt.show()
    exit()
    #
    # pts1 = refine_points(pts1, img1)
    # pts2 = refine_points(pts2, img2)

    # show_points(pts1, pts2, img1, img2, points_labels)
    return pts1, pts2


def make_spline_3D(control_points):
    # Calculate cumulative distances as parameter t
    distances = np.sqrt(np.sum(np.diff(control_points, axis=0) ** 2, axis=1))
    total_distance = np.sum(distances)
    t = np.insert(np.cumsum(distances), 0, 0)

    # Separate control points into x, y, z components
    x = control_points[:, 0]
    y = control_points[:, 1]
    z = control_points[:, 2]

    # Create cubic splines for each dimension
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)

    # Define a function that evaluates the spline at a given t
    def spline_3d(t_eval):
        x_eval = cs_x(t_eval)
        y_eval = cs_y(t_eval)
        z_eval = cs_z(t_eval)
        return np.array([x_eval, y_eval, z_eval])

    return spline_3d


def test_splines():
    import annot_parser
    from calibration import P1, P2, F

    dataset_path = Path.home() / "data" / "segment-real"
    dataset = annot_parser.get_structured_dataset(dataset_path / "annotations.xml")

    for i in range(0, 100 * 3, 100):
        sample = dataset[i]

        points1 = sample["points1"]
        points2 = sample["points2"]

        img1 = plt.imread(dataset_path / sample["image1"])
        img2 = plt.imread(dataset_path / sample["image2"])

        points1_resampled, points2_resampled, points_3d = get_points(
            points1, points2, P1, P2, spacing=10, F=F
        )

        spline_3d = make_spline_3D(points_3d)
        t_values = np.linspace(0, 15, 30)

        points_3d = spline_3d(t_values).T
        print(points_3d.shape)

        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], "o-", label="spline")
        plt.show()
        plt.close()


if __name__ == "__main__":
    import annot_parser
    import plot
    from calibration import K1, K2, P1, P2, F, d1, d2

    test_splines()
    exit()

    dataset_path = Path.home() / "data" / "segment-real"
    dataset = annot_parser.get_structured_dataset(dataset_path / "annotations.xml")

    # F, _ = get_F_matrix()
    plot_points_w_lines(F)
    exit()

    for i in range(0, 100 * 3, 100):
        sample = dataset[i]

        points1 = sample["points1"]
        points2 = sample["points2"]

        img1 = plt.imread(dataset_path / sample["image1"])
        img2 = plt.imread(dataset_path / sample["image2"])

        points1_resampled, points2_resampled, points_3d = get_points(
            points1, points2, P1, P2, spacing=10, F=F
        )

        lines_in_img2 = cv2.computeCorrespondEpilines(
            points1_resampled.reshape(-1, 1, 2), 1, F
        ).reshape(-1, 3)
        lines_in_img1 = cv2.computeCorrespondEpilines(
            points2_resampled.reshape(-1, 1, 2), 2, F
        ).reshape(-1, 3)

        img2_with_lines = drawlines_single(img2, lines_in_img2, points2_resampled)
        img1_with_lines = drawlines_single(img1, lines_in_img1, points1_resampled)

        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

        reprojected1 = fn.project_points(points_3d, P1)
        reprojected2 = fn.project_points(points_3d, P2)

        print("min", points_3d.min(), "max", points_3d.max())
        print("shape", points_3d.shape)
        print("chain_length", fn.calculate_segment_length(points_3d))

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        plot.make_paired_plots(
            [img1] * 3,
            [points1, points1_resampled, reprojected1],
            ["Original", "Resampled", "Reprojected"],
            ax[0],
        )
        plot.make_paired_plots(
            [img2] * 3,
            [points2, points2_resampled, reprojected2],
            ["Original", "Resampled", "Reprojected"],
            ax[1],
        )
        plt.show()
        plt.close()
