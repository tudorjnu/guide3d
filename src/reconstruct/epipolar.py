import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import fn
import plot


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


def reconstruct(pts1, pts2):
    import calibration

    F = calibration.F

    pts1_matches = match_points(pts1, pts2, F)
    # pts2_matches = match_points(pts2, pts1, F)

    pass


def preprocess_points(pts1, pts2):
    from scipy.interpolate import splev

    import representations.curve as curve

    curve1 = curve.parametrize_curve(pts1, s=10)
    curve2 = curve.parametrize_curve(pts2, s=10)

    u1, u2 = curve1[1], curve2[1]
    tck1, tck2 = curve1[0], curve2[0]
    u_min, u_max = max(u1[0], u2[0]), min(u1[-1], u2[-1])

    u = np.linspace(u_min, u_max, 15)

    x1, y1 = splev(u, tck1)
    x2, y2 = splev(u, tck2)

    pts1 = np.column_stack((x1, y1))
    pts2 = np.column_stack((x2, y2))

    pts1 = pts1.astype(np.int32)
    pts2 = pts2.astype(np.int32)

    print(pts1)
    print(pts2)

    return pts1, pts2


def main():
    import plot
    import vars
    import viz

    dataset_path = vars.dataset_path

    samples = get_data()[:-1]  # remove last sample to match the number of subplots

    for i, sample in enumerate(samples):
        pts1 = np.array(sample["pts1"])
        pts2 = np.array(sample["pts2"])

        pts1, pts2 = preprocess_points(pts1, pts2)

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot.make_paired_plot(img1, img2, pts1, pts2, axs)
        plt.show()
        plt.close()
        continue

        pts_reconstructed = reconstruct(pts1, pts2)
        exit()

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        img1 = viz.convert_to_color(img1)
        img2 = viz.convert_to_color(img2)

        ax = axs[i]
        set_ax(ax)

        # exit()
        #
        # pts1_resampled = pts1_resampled.astype(np.int32)
        # pts2_resampled = pts2_resampled.astype(np.int32)
        #
        # ln2 = cv2.computeCorrespondEpilines(
        #     pts1_resampled.reshape(-1, 1, 2), 1, F
        # ).reshape(-1, 3)
        # ln1 = cv2.computeCorrespondEpilines(
        #     pts2_resampled.reshape(-1, 1, 2), 2, F
        # ).reshape(-1, 3)
        #
        # subset = np.linspace(0, len(pts1_resampled) - 1, 10, dtype=int)
        # colors = fn.generate_colors(len(subset))
        #
        # pts1_resampled_sample = pts1_resampled[subset]
        # pts2_resampled_sample = pts2_resampled[subset]
        # ln1 = ln1[subset]
        # ln2 = ln2[subset]
        # pts3d = pts3d[subset]
        #
        # # img1 = viz.draw_points(img1, pts1_resampled_sample, colors)
        # # img1 = viz.draw_lines(img1, ln1, colors)
        # # img1 = viz.draw_points(img1, pts1_resampled)
        # img1 = viz.draw_polyline(img1, pts1_resampled, color=(0, 0, 255))
        #
        # img2 = viz.draw_points(img2, pts2_resampled_sample, colors)
        # img2 = viz.draw_lines(img2, ln2, colors)
        #
        # for i in range(len(pts3d)):
        #     color = tuple(map(int, colors[i]))
        #     ln_pts = fn.find_line_extremities(ln1[i], *img1.shape[:2])
        #     intersections = find_intersections(ln_pts, pts1_resampled, True)
        #     if intersections is not None:
        #         # print(intersections)
        #         # intersection = intersections[0].astype(np.int32)
        #         if len(intersections) > 1:
        #             img1 = viz.draw_polyline(img1, ln_pts, color)
        #             for i, (intersection, color) in enumerate(
        #                 zip(intersections, fn.generate_colors(len(intersections)))
        #             ):
        #                 color = tuple(map(int, color))
        #                 img1 = cv2.circle(img1, tuple(intersection), 5, color, -1)
        #                 # put text over points in img1
        #                 cv2.putText(
        #                     img1,
        #                     f"({i}-{intersection[0]}, {intersection[1]})",
        #                     tuple(intersection),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color,
        #                     2,
        #                 )
        #             # img1 = cv2.circle(img1, tuple(intersection), 5, color, -1)
        #
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[1].imshow(img1)
        # ax[1].axis("off")
        # ax[0].imshow(img2)
        # ax[0].axis("off")
        # plt.show()
        # plt.close()
    # remove horizontal and vertical spacing between subplots
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
