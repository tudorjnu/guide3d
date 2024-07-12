from typing import List, Tuple

import guide3d.reconstruction.epipolar.fig as ep_fig
import guide3d.reconstruction.epipolar.plot as ep_plot
import guide3d.reconstruction.epipolar.viz as ep_viz
import guide3d.representations.curve as curve
import guide3d.utils.plot as plot
import guide3d.utils.viz as viz
import numpy as np
from guide3d import calibration
from guide3d.reconstruction import triangulation
from guide3d.reconstruction.epipolar import fn as ep_fn
from guide3d.utils.utils import get_data
from matplotlib import pyplot as plt
from scipy.interpolate import splev

ep_viz = ep_viz
i = 0


def get_curve_3d(
    tckA: Tuple[np.ndarray, List[np.ndarray], int],
    tckB: Tuple[np.ndarray, List[np.ndarray], int],
    uA: list,
    uB: list,
    n: int = None,
    delta: float = None,
    others: dict = {},
    visualize: bool = False,
    s: float = None,
):
    global i
    try:
        # find matching u values
        uA, uA_matches = ep_fn.match_points(
            tckA, tckB, uA, uB, n=n, delta=delta, F=calibration.F_A_B
        )

        # interpolate_matches
        uA_matches = ep_fn.interpolate_matches(uA, uA_matches)

        # sample points at u values
        ptsA = np.column_stack(splev(uA, tckA))
        ptsA_matches = np.column_stack(splev(uA_matches, tckB))

        ptsA = ptsA
        ptsB = ptsA_matches

        if visualize:
            img1 = others["img1"]
            img2 = others["img2"]
            u2, u2_mathces = ep_fn.match_points(
                tckB, tckA, uB, uA, n=n, delta=delta, F=calibration.F_B_A
            )
            u2_mathces = ep_fn.interpolate_matches(u2, u2_mathces)
            pts2 = np.column_stack(splev(u2, tckB))
            pts2_matches = np.column_stack(splev(u2_mathces, tckA))
            ep_plot.plot_with_matches(
                ptsA, pts2, ptsA_matches, pts2_matches, img1, img2
            )
            ep_fig.make_epilines_plot(
                ptsA,
                ptsA_matches,
                img1,
                img2,
                calibration.F_A_B,
                tckA,
                others["u1"],
                tckB,
                others["u2"],
                i,
            )

        i += 1
        reconstructed_pts = triangulation.reconstruct(
            ptsA, ptsB, calibration.P1, calibration.P2, calibration.F_A_B
        )

        reconstructed_pts = ep_fn.remove_duplicates(reconstructed_pts)

        tck3d, u3D = curve.fit_spline(reconstructed_pts, s=s, k=5)

        if visualize:
            reconstructed_pts = np.column_stack(splev(u3D, tck3d))
            # ep_plot.plot_reprojection(img1, tckA, u_max, tck3d, u3D[-1], calibration.P1)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                reconstructed_pts[:, 0],
                reconstructed_pts[:, 1],
                reconstructed_pts[:, 2],
                marker="o",
                markersize=1,
                linestyle="-",
            )
            ax.view_init(5, 180)
            plot.plot_mesh(plot.get_mesh(), ax)
            plt.show()
            plt.close()

        return tck3d, u3D
    except Exception as e:
        print("Error in get_curve_3d:", e)
        return None, None


reconstruct = get_curve_3d


def main():
    import guide3d.vars as vars

    dataset_path = vars.dataset_path

    samples = get_data()[:4]

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

        print("\nSample", i)
        get_curve_3d(
            tck1,
            tck2,
            u1,
            u2,
            delta=20,
            others={
                "img1": img1,
                "img2": img2,
                "tck1": tck1,
                "tck2": tck2,
                "u1": u1,
                "u2": u2,
            },
            visualize=True,
        )


if __name__ == "__main__":
    main()
