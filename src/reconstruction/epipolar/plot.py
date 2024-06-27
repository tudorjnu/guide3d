import calibration
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils.fn as fn
from scipy.interpolate import splev


def plot_curve_2D(ax: plt.Axes, tck: tuple, u: np.ndarray, **kwargs):
    pts = np.column_stack(splev(u, tck))
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        marker="o",
        markersize=1,
        linestyle="-",
        **kwargs,
    )


def plot_reprojection(img, tckA, uA_max, tck3d, u3d_max, PA):
    ptsA = np.column_stack(splev(np.linspace(0, uA_max, 100), tckA))

    pts3d = np.column_stack(splev(np.linspace(0, u3d_max, 100), tck3d))
    ptsA_reprojected = fn.project_points(pts3d, PA)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    ax.plot(ptsA[:, 0], ptsA[:, 1], markersize=1, label="GT")
    ax.plot(
        ptsA_reprojected[:, 0],
        ptsA_reprojected[:, 1],
        markersize=1,
        label="Reprojected",
    )
    ax.legend()
    ax.axis("off")
    plt.show()
    plt.close()


def plot_with_matches(pts1, pts2, matches1, matches2, img1, img2):
    def set_fig():
        fig, axs = plt.subplots(2, 2)
        for ax in axs.flatten():
            ax.axis("off")
        return fig, axs

    fig, axs = set_fig()

    pts1 = pts1.astype(int)
    pts2 = pts2.astype(int)
    matches1 = matches1.astype(int)
    matches2 = matches2.astype(int)

    img11, img12 = np.copy(img1), np.copy(img1)
    img21, img22 = np.copy(img2), np.copy(img2)

    img11, img21 = draw_pts_and_lines_and_matches(
        img11, img21, pts1, matches1, calibration.F_A_B
    )
    img22, img12 = draw_pts_and_lines_and_matches(
        img22, img12, pts2, matches2, calibration.F_B_A
    )

    # plot 3D curve
    axs[0, 0].imshow(img11, cmap="gray")
    axs[0, 1].imshow(img21, cmap="gray")
    axs[1, 0].imshow(img12, cmap="gray")
    axs[1, 1].imshow(img22, cmap="gray")

    fig.tight_layout()
    fig.savefig("figs/matches.png")
    plt.show()
    plt.close()


def draw_pts_and_lines(img1, img2, pts, F):
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


def draw_line(img, line, color):
    r, c, _ = img.shape
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img


def draw_pts_and_lines_and_matches(img1, img2, pts, matches, F):
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
        img2 = cv2.circle(img2, tuple(matches[i]), 5, color, -1)
    return img1, img2
