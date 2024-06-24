import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import fn
import plot
import viz
from calibration import F


def order_matches(pts1, pts2):
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


def match_points(pts1, pts2, F):
    """Finds matching point using epipolar geometry"""

    ln_in_img2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(
        -1, 3
    )

    matches_in_img2 = []

    for i in range(len(ln_in_img2)):
        if pts2.size == 0:
            matches_in_img2.append(None)
            continue
        ln2_pts = fn.find_line_extremities(ln_in_img2[i], 1024, 1024)
        intersections = fn.find_intersections(ln2_pts, pts2, True)
        if intersections:
            matches_in_img2.append(np.array(intersections[0]))
            pts2 = cut_points_from(pts2, intersections[0])
    return matches_in_img2


def interpolate_between(polyline, pts, n_pts):
    polyline = LineString(polyline)
    interpolated_points = []
    pts = [Point(p) for p in pts]

    distances = [polyline.project(pt) for pt in pts]

    for i in range(len(distances) - 1):
        segment_start = distances[i]
        segment_end = distances[i + 1]
        segment_length = segment_end - segment_start

        # Generate n_pts evenly spaced distances along the segment
        for j in range(1, n_pts + 1):
            # Calculate the distance along the polyline for this interpolated point
            dist_along_polyline = segment_start + (segment_length * j / (n_pts + 1))

            # Find the interpolated point at this distance along the polyline
            interpolated_point = polyline.interpolate(dist_along_polyline)
            interpolated_points.append((interpolated_point.x, interpolated_point.y))

    return np.array(interpolated_points)


def cut_points_from(points, cut_point):
    polyline = LineString(points)
    pt = Point(cut_point)

    nearest_pt_on_polyline = nearest_points(polyline, pt)[0]

    distances_from_start = [polyline.project(Point(p)) for p in points]

    cut_point_distance = polyline.project(nearest_pt_on_polyline)

    cut_points = [
        [point[0], point[1]]
        for point, dist in zip(points, distances_from_start)
        if dist >= cut_point_distance
    ]

    return np.array(cut_points)


def get_data():
    import json

    frames = [200]
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


def set_ax(ax):
    mesh = plot.get_mesh()
    plot.plot_mesh(mesh, ax)

    # remove labels and tick numbers
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def is_tip_first(pts):
    pt0 = pts[0]
    pt_last = pts[-1]
    if pt0[0] > pt_last[0]:
        return True
    else:
        return False


def reconstruct(pts1, pts2):
    import calibration

    F = calibration.F

    pts1_matches = match_points(pts1, pts2, F)
    pts2_matches = match_points(pts2, pts1, F)

    pass


def preprocess_points(pts1, pts2):
    from scipy.interpolate import splev

    import representations.curve as curve

    if not is_tip_first(pts1):
        pts1 = np.flip(pts1, axis=0)
    if not is_tip_first(pts2):
        pts2 = np.flip(pts2, axis=0)

    curve1 = curve.parametrize_curve(pts1, s=10)
    curve2 = curve.parametrize_curve(pts2, s=10)

    u1, u2 = curve1[1], curve2[1]
    tck1, tck2 = curve1[0], curve2[0]
    u_min = 0
    u_max = min(u1[-1], u2[-1])

    u = np.linspace(u_min, u_max, 4)

    x1, y1 = splev(u, tck1)
    x2, y2 = splev(u, tck2)

    pts1 = np.column_stack((x1, y1))
    pts2 = np.column_stack((x2, y2))

    pts1 = pts1.astype(np.int32)
    pts2 = pts2.astype(np.int32)

    print(pts1)
    print(pts2)

    return pts1, pts2


def draw_line(img, line, color):
    r, c, _ = img.shape
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img


def draw_pts_and_lines(img1, img2, pts):
    lns = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    r, c, _ = img1.shape

    for i, (pt, ln) in enumerate(zip(pts, lns)):
        label = f"{i}"
        color = np.random.randint(0, 255, 3).tolist()
        img1 = cv2.circle(img1, tuple(pt), 5, color, -1)
        img1 = cv2.putText(
            img1,
            label,
            tuple(pt + np.array([10, 10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            2,
        )

        img2 = draw_line(img2, ln, color)
    return img1, img2


def plot_pts_and_epilines(pts1, pts2, img1, img2, F=F):
    fig, axs = plt.subplots(2, 2)

    img11, img12 = np.copy(img1), np.copy(img1)
    img21, img22 = np.copy(img2), np.copy(img2)

    img11, img21 = draw_pts_and_lines(img11, img21, pts1)
    img22, img12 = draw_pts_and_lines(img22, img12, pts2)

    axs[0, 0].imshow(img11)
    axs[0, 0].axis("off")
    axs[0, 1].imshow(img21)
    axs[0, 1].axis("off")
    axs[1, 0].imshow(img12)
    axs[1, 0].axis("off")
    axs[1, 1].imshow(img22)
    axs[1, 1].axis("off")
    plt.show()
    plt.close()


def main():
    import vars

    dataset_path = vars.dataset_path

    samples = get_data()[:-1]  # remove last sample to match the number of subplots

    for i, sample in enumerate(samples):
        pts1 = np.array(sample["pts1"])
        pts2 = np.array(sample["pts2"])

        pts1, pts2 = preprocess_points(pts1, pts2)

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        img1 = viz.convert_to_color(img1)
        img2 = viz.convert_to_color(img2)

        plot_pts_and_epilines(pts1, pts2, img1, img2)

        continue
        pts_reconstructed = reconstruct(pts1, pts2)
        exit()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
