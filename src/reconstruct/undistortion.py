from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

import calibration
import plot
import viz
from calibration import F


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

    # pts1 = refine_points(pts1, img1)
    # pts2 = refine_points(pts2, img2)

    # show_points(pts1, pts2, img1, img2, points_labels)
    return pts1, pts2


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

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS, 0.1, 0.99)
    F = calibration.F

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


def get_F_from_pts():
    pts1, pts2 = get_p()
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS, 0.1, 0.99)
    return F


def undistort_image(img, K, d):
    h, w = img.shape[:2]

    # Get the optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_img = cv2.undistort(img, K, d, None, newcameramtx)

    # Crop the image (if needed)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y : y + h, x : x + w]

    return undistorted_img


def undistort_points(pts, K, d):
    pts = pts.astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        K, d, (1024, 1024), 1, (1024, 1024)
    )
    undistorted_pts = cv2.undistortPoints(pts, K, d, P=newcameramtx)
    undistorted_pts = undistorted_pts.reshape(-1, 2)
    return undistorted_pts.astype(np.int32)


def main():
    import vars

    # test()
    # exit()

    dataset_path = vars.dataset_path

    samples = get_data()[:-1]  # remove last sample to match the number of subplots

    for i, sample in enumerate(samples):
        pts1 = np.array(sample["pts1"])
        pts2 = np.array(sample["pts2"])

        pts1, pts2 = preprocess_points(pts1, pts2)

        pts_undistorted1 = undistort_points(pts1, calibration.K2, calibration.d2)
        pts_undistorted2 = undistort_points(pts2, calibration.K1, calibration.d1)

        print("Pts 2", pts2)
        print("Pts 2 Undistorted", pts_undistorted2)

        img1 = plt.imread(dataset_path / sample["img1"])
        img2 = plt.imread(dataset_path / sample["img2"])

        img1_undist = undistort_image(img1, calibration.K2, calibration.d2)
        img2_undist = undistort_image(img2, calibration.K1, calibration.d1)

        fig, axs = plt.subplots(2, 2, figsize=(10, 5))

        axs[0, 0].imshow(img1, cmap="gray")
        axs[0, 0].axis("off")
        axs[0, 0].scatter(pts1[:, 0], pts1[:, 1], s=1)
        axs[0, 0].set_title("Image 1 - Original")

        axs[0, 1].imshow(img1_undist, cmap="gray")
        axs[0, 1].axis("off")
        axs[0, 1].scatter(pts_undistorted1[:, 0], pts_undistorted1[:, 1], s=1)
        axs[0, 1].set_title("Image 1 - Undistorted")

        axs[1, 0].imshow(img2, cmap="gray")
        axs[1, 0].axis("off")
        axs[1, 0].scatter(pts2[:, 0], pts2[:, 1], s=1)
        axs[1, 0].set_title("Image 2 - Original")

        axs[1, 1].imshow(img2_undist, cmap="gray")
        axs[1, 1].axis("off")
        axs[1, 1].scatter(pts_undistorted2[:, 0], pts_undistorted2[:, 1], s=1)
        axs[1, 1].set_title("Image 2 - Undistorted")

        plt.show()
        plt.close()
        continue
        exit()

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(img1)
        # ax[0].axis("off")
        # ax[1].imshow(img1_undistorted)
        # ax[1].axis("off")
        # plt.show()
        # plt.close()
        # exit()

        img1 = viz.convert_to_color(img1)
        img2 = viz.convert_to_color(img2)

        plot_pts_and_epilines(pts1, pts2, img1, img2)

        continue
        exit()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
