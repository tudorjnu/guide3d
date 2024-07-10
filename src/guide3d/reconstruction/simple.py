import fn
import numpy as np


def get_valid_points(points: np.ndarray):
    if np.any(points[3, :] == 0) or np.any(np.isnan(points[3, :])):
        mask = np.logical_and(points[3, :] != 0, ~np.isnan(points[3, :]))
        print(f"Foudn {np.sum(~mask)} invalid points")
        return mask


def get_points(
    points1: np.ndarray,
    points2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    spacing: int = 10,
):
    points1 = fn.interpolate_even_spacing(points1, spacing=spacing)
    points2 = fn.interpolate_even_spacing(points2, spacing=spacing)

    # Ensure that the two polylines have the same length
    min_length = min(len(points1), len(points2))
    points1 = points1[:min_length]
    points2 = points2[:min_length]

    points_3d_h = fn.triangulate(P1, P2, points1, points2)

    points_3d = points_3d_h[:, :3] / points_3d_h[:, 3, np.newaxis]

    assert (
        points1.shape == points2.shape and points2.shape[0] == points_3d.shape[0]
    ), f"points1.shape = {points1.shape}, points2.shape = {points2.shape}, points_3d.shape = {points_3d.shape}"

    return points1, points2, points_3d


if __name__ == "__main__":
    from pathlib import Path

    import annotation
    import calibration
    import matplotlib.pyplot as plt
    from visualization import plot

    dataset_path = Path.home() / "data" / "segment-real"
    dataset = annotation.get_structured_dataset(dataset_path / "annotations.xml")
    P1, P2 = calibration.P1, calibration.P2

    for i in range(0, 100 * 8, 100):
        sample = dataset[i]

        points1 = sample["points1"]
        points2 = sample["points2"]
        img1 = plt.imread(dataset_path / sample["image1"])
        img2 = plt.imread(dataset_path / sample["image2"])

        points1_resampled, points2_resampled, points_3d = get_points(
            points1, points2, P1, P2, spacing=30
        )
        reprojected1 = fn.project_points(points_3d, P1)
        reprojected2 = fn.project_points(points_3d, P2)

        print("min", points_3d.min(), "max", points_3d.max())
        print("shape", points_3d.shape)
        print(fn.compute_chain_distance(points_3d))

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # plot.plot_image_w_points(img1, points1, ax=ax1)
        # plot.plot_image_w_points(img2, points2, ax=ax2)

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
