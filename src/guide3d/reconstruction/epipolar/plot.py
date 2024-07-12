import cv2
import guide3d.calibration as calibration
import guide3d.reconstruction.epipolar.viz as ep_viz
import guide3d.utils.fn as fn
import guide3d.utils.viz as viz
import guide3d.vars as vars
import matplotlib.pyplot as plt
import numpy as np
from guide3d.utils.utils import get_data
from scipy.interpolate import splev

i = 0


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


def plot_with_matches_bckup(pts1, pts2, matches1, matches2, img1, img2):
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

    img11, img21 = ep_viz.draw_pts_and_lines_and_matches(
        img11, img21, pts1, matches1, calibration.F_A_B
    )
    img22, img12 = ep_viz.draw_pts_and_lines_and_matches(
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

        img2 = ep_viz.draw_line(img2, ln, color)
    return img1, img2


def plot_with_matches(
    imgA, imgB, ptsA, ptsA_matches, ax=None, show=True, n_pts=10, save=False
):
    if ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Ensure n_pts does not exceed the length of ptsA
    n_pts = min(n_pts, len(ptsA))

    imgA = np.copy(imgA)
    imgB = np.copy(imgB)

    ptsA = ptsA.astype(np.int32)
    ptsA_matches = ptsA_matches.astype(np.int32)

    indices = np.linspace(0, len(ptsA) - 1, n_pts, dtype=int)
    ptsA = ptsA[indices]
    ptsA_matches = ptsA_matches[indices]

    lnsA_in_B = cv2.computeCorrespondEpilines(
        ptsA.reshape(-1, 1, 2), 1, calibration.F_A_B
    ).reshape(-1, 3)
    lnsB_in_A = cv2.computeCorrespondEpilines(
        ptsA_matches.reshape(-1, 1, 2), 1, calibration.F_B_A
    ).reshape(-1, 3)

    for ptA, lnA, ptB, lnB in zip(ptsA, lnsB_in_A, ptsA_matches, lnsA_in_B):
        color = np.random.randint(0, 255, 3).tolist()
        imgA = cv2.circle(imgA, tuple(ptA), 8, color, -1)
        imgB = cv2.circle(imgB, tuple(ptB), 8, color, -1)
        imgA = viz.draw_line(imgA, lnA, color, thickness=3)
        imgB = viz.draw_line(imgB, lnB, color, thickness=3)

    if save:
        plt.imsave(f"figs/raw/epilines/{i}_A.png", imgA)
        plt.imsave(f"figs/raw/epilines/{i}_B.png", imgB)

    axs[0].axis("off")
    axs[1].axis("off")

    axs[0].imshow(imgA, cmap="gray")
    axs[1].imshow(imgB, cmap="gray")

    if show:
        plt.show()
        plt.close()
        return axs
    else:
        plt.close()
        return axs


def make_3D_plot(pts, ax: plt.Axes = None, show=True, save=False):
    from utils import plot

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        marker="o",
        markersize=1,
        linestyle="-",
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(5, 180)
    plot.plot_mesh(plot.get_mesh(), ax)
    if save:
        fig.savefig(f"figs/raw/3d/{i}.png", pad_inches=0, bbox_inches="tight")

    if show:
        plt.show()
        plt.close()
        return ax
    else:
        return ax


def main():
    global i
    import utils.viz as viz
    from representations import curve

    import reconstruction.epipolar.fn as ep_fn
    from reconstruction import triangulation

    DELTA = 30

    # get some sample data
    dataset_path = vars.dataset_path
    samples = get_data()

    for sample in samples:
        ptsA = sample["pts1"]
        ptsB = sample["pts2"]

        tckA, uA = curve.fit_spline(ptsA)
        tckB, uB = curve.fit_spline(ptsB)

        ptsA = curve.sample_spline(tckA, uA, delta=DELTA)
        ptsB = curve.sample_spline(tckB, uB, delta=DELTA)

        # process the images
        imgA = plt.imread(dataset_path / sample["img1"])
        imgB = plt.imread(dataset_path / sample["img2"])

        imgA = viz.convert_to_color(imgA)
        imgB = viz.convert_to_color(imgB)

        imgA = viz.draw_curve(imgA, tckA, uA, color=(255, 255, 255))
        imgB = viz.draw_curve(imgB, tckB, uB, color=(255, 255, 255))

        uA, uA_matches = ep_fn.match_points(
            tckA, tckB, uA, uB, delta=DELTA, F=calibration.F
        )
        if uA_matches is None or len(uA_matches) < 2:
            continue

        # interpolate_matches
        uA_matches = ep_fn.interpolate_matches(uA, uA_matches)

        # sample points at u values
        ptsA = np.column_stack(splev(uA, tckA))
        ptsA_matches = np.column_stack(splev(uA_matches, tckB))

        plot_with_matches(imgA, imgB, ptsA, ptsA_matches, show=False)

        reconstructed_pts = triangulation.reconstruct(
            ptsA, ptsA_matches, calibration.P1, calibration.P2, calibration.F
        )

        reconstructed_pts = ep_fn.remove_duplicates(reconstructed_pts)
        if reconstructed_pts.shape[0] < 4:
            continue

        # fit a spline to the reconstructed points
        tck3d, u3D = curve.fit_spline(reconstructed_pts, s=None, k=5)

        # sample the spline
        reconstructed_pts = np.column_stack(splev(u3D, tck3d))
        make_3D_plot(reconstructed_pts, show=False, save=True)
        # exit()
        i += 1


if __name__ == "__main__":
    main()
