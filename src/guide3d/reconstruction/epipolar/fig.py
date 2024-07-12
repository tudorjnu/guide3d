import cv2
import guide3d.utils.viz as viz
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev


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


def draw_line(img, line, color):
    r, c, _ = img.shape
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img


def draw_pts_and_lines(ptsA, ptsB, imgA, imgB, F):
    lnsB = cv2.computeCorrespondEpilines(ptsA.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lnsA = cv2.computeCorrespondEpilines(ptsB.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    r, c, _ = imgA.shape

    colors = np.random.randint(0, 255, (len(ptsA), 3))

    for i, (ptA, lnB, ptB, lnA) in enumerate(zip(ptsA, lnsB, ptsB, lnsA)):
        color = colors[i].tolist()
        imgA = cv2.circle(imgA, tuple(ptA), 5, color, -1)
        imgB = cv2.circle(imgB, tuple(ptB), 5, color, -1)
        imgA = draw_line(imgA, lnA, color)
        imgB = draw_line(imgB, lnB, color)
    return imgA, imgB


def make_epilines_plot(ptsA, ptsB, imgA, imgB, F, tckA, uA, tckB, uB, i):
    def set_up_fig():
        fig, axs = plt.subplots(1, 2, figsize=(4.5, 2))

        for ax in axs.flatten():
            ax.axis("off")
        return fig, axs

    fig, axs = set_up_fig()

    ptsA = ptsA.astype(int)
    ptsB = ptsB.astype(int)

    curveA = np.column_stack(splev(uA, tckA)).astype(np.int32)
    curveB = np.column_stack(splev(uB, tckB)).astype(np.int32)

    imgA = viz.draw_polyline(imgA, curveA, color=(255, 255, 255))
    imgB = viz.draw_polyline(imgB, curveB, color=(255, 255, 255))

    imgA, imgB = draw_pts_and_lines(ptsA, ptsB, imgA, imgB, F)

    plt.imsave(f"figs/raw/epilines/{i}_A.png", imgA)
    plt.imsave(f"figs/raw/epilines/{i}_B.png", imgB)

    axs[0].imshow(imgA)
    axs[1].imshow(imgB)

    fig.tight_layout()
    fig.savefig("figs/epilines.png")
    plt.show()
    plt.close()


def make_reprojection_plot():
    def set_up_fig():
        fig, axs = plt.subplots(2, 2)
        for ax in axs.flatten():
            ax.axis("off")
        return fig, axs

    pass


def make_3D_plot():
    def set_up_fig():
        fig, axs = plt.subplots(2, 2)
        for ax in axs.flatten():
            ax.axis("off")
        return fig, axs

    pass


def main():
    data = get_data()
    exit()
    make_epilines_plot()
    make_reprojection_plot()
    make_3D_plot()


if __name__ == "__main__":
    main()
