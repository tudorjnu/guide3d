import cv2
import guide3d.calibration as calibration
import matplotlib.pyplot as plt
import numpy as np


def draw_line(img, line, color):
    r, c, _ = img.shape
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img


def plot_with_matches(pts1, pts2, matches1, matches2, img1, img2):
    fig, axs = plt.subplots(2, 2)

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
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img21, cmap="gray")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(img12, cmap="gray")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(img22, cmap="gray")
    axs[1, 1].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
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


def draw_pts_and_lines_and_matches(img1, img2, pts, matches, F):
    lns = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    r, c, _ = img1.shape

    # img2 = cv2.polylines(img2, [matches], isClosed=False, color=(255, 255, 255))

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


def plot_pts_and_epilines(pts1, pts2, img1, img2):
    fig, axs = plt.subplots(2, 2)

    img11, img12 = np.copy(img1), np.copy(img1)
    img21, img22 = np.copy(img2), np.copy(img2)

    img11, img21 = draw_pts_and_lines(img11, img21, pts1, calibration.F_A_B)
    img22, img12 = draw_pts_and_lines(img22, img12, pts2, calibration.F_B_A)

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
