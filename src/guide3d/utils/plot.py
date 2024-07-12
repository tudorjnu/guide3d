from pathlib import Path

import guide3d.utils.viz as viz
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def get_mesh(transformed=False):
    mesh = trimesh.load_mesh(Path(__file__).parent.parent / "assets/mesh.stl")
    affine_matrix = np.load(
        Path(__file__).parent.parent / "assets/transformation_matrix.npy"
    )
    if transformed:
        mesh.apply_transform(affine_matrix)
    return mesh


def plot_mesh(mesh, ax):
    vertices = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="grey",
        alpha=0.1,
    )


def plot_image_w_points(img, points, ax=None, **kwargs):
    ax.imshow(img, cmap="gray")
    ax.scatter(points[:, 0], points[:, 1], s=0.5, **kwargs)
    ax.axis("off")
    return ax


def make_paired_plot(img1, img2, points1, points2, ax=None, s=1):
    ax1, ax2 = ax
    ax1.imshow(img1, cmap="gray")
    ax1.scatter(points1[:, 0], points1[:, 1], s=s)
    ax1.set_title("Image 1")
    ax1.axis("off")
    ax2.imshow(img2, cmap="gray")
    ax2.scatter(points2[:, 0], points2[:, 1], s=s)
    ax2.set_title("Image 2")
    ax2.axis("off")
    return ax1, ax2


def make_paired_plots(images, points, titles, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, len(images), figsize=(10, 5))
    for i, (img, p, title) in enumerate(zip(images, points, titles)):
        if img.ndim == 2:
            img = viz.convert_to_color(img)
        if p.dtype != np.int32:
            p = p.astype(np.int32)
        img = viz.draw_points(img, p)
        ax[i].imshow(img)
        ax[i].set_title(title)
        ax[i].axis("off")
    return ax


def make_error_plot(mean_errors, std_errors, ax=None, **kwargs):
    ax.errorbar(
        np.arange(mean_errors.shape[0]),
        mean_errors,
        yerr=std_errors,
        fmt="o",
        capsize=1,
        capthick=1,
        elinewidth=1,
        markersize=2,
        **kwargs,
    )
    ax.set_ylabel("Error (pixels)")
    ax.set_xlabel("Point Index")
    ax.grid(
        True, which="both", axis="both", linestyle="--", linewidth=0.5, color="gray"
    )
    return ax


def make_error_plot_3D(mean_errors, std_errors, ax=None, **kwargs):
    ax.errorbar(
        np.arange(mean_errors.shape[0]),
        mean_errors,
        yerr=std_errors,
        fmt="o",
        capsize=1,
        capthick=1,
        elinewidth=1,
        markersize=2,
        **kwargs,
    )
    ax.set_ylabel("Error (mm)")
    ax.set_xlabel("Segment Index")
    ax.grid(
        True, which="both", axis="both", linestyle="--", linewidth=0.5, color="gray"
    )
    return ax


def make_3d_plot(points, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel("X", linespacing=0, labelpad=-10)
    ax.set_ylabel("Y", linespacing=0, labelpad=-10)
    ax.set_zlabel("Z", linespacing=0, labelpad=-10)

    # make the plot have equal aspect ratio
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # remove the ax tick labels but keep the grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # set view to be parallel with the x axis and a bit elevated
    # ax.view_init(5, 90)

    return ax


def make_3d_plot_with_mesh(points, ax=None, color="blue"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    mesh = get_mesh()
    plot_mesh(mesh, ax)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=color)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1)

    ax.set_xlabel("X", linespacing=0, labelpad=-10)
    ax.set_ylabel("Y", linespacing=0, labelpad=-10)
    ax.set_zlabel("Z", linespacing=0, labelpad=-10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(5, 180)

    return ax


def error_plot_reprojection(points1, points2, P1, P2, F, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # Compute epipolar lines
    lines1 = np.array([F.dot(points2[i]) for i in range(points2.shape[0])])
    lines2 = np.array([F.T.dot(points1[i]) for i in range(points1.shape[0])])
    # Compute reprojection errors
    errors1 = np.abs(np.diag(points1.dot(P1.T.dot(lines1.T)))) / np.linalg.norm(
        lines1[:, :2], axis=1
    )
    errors2 = np.abs(np.diag(points2.dot(P2.T.dot(lines2.T)))) / np.linalg.norm(
        lines2[:, :2], axis=1
    )
    ax.plot(errors1, label="Image 1")
    ax.plot(errors2, label="Image 2")
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Reprojection Error (px)")
    ax.legend()
    ax.grid(
        True, which="both", axis="both", linestyle="--", linewidth=0.5, color="gray"
    )
    return ax


if __name__ == "__main__":
    import guide3d.annotation as annotation
    import guide3d.calibration as calibration
    import guide3d.reconstruct as reconstruct
    import guide3d.utils as utils
    import paths

    dataset_path = paths.test_dataset_path
    dataset = annotation.get_structured_dataset(dataset_path / "annotations.xml")
    P1, P2 = calibration.P1, calibration.P2

    sample = dataset[0]
    points1 = sample["points1"]
    points2 = sample["points2"]
    img1_path = dataset_path / sample["image1"]
    img2_path = dataset_path / sample["image2"]

    p1, p2, p3d = reconstruct.get_points(points1, points2, P1, P2)
    p1_reprojected = utils.functions.project_points(p3d, P1)
    p2_reprojected = utils.functions.project_points(p3d, P2)

    fig = plt.figure()
    row1 = fig.add_subplot(2, 3, 1)
    make_paired_plot(
        plt.imread(img1_path),
        plt.imread(img2_path),
        p1,
        p2,
        ax=(row1, fig.add_subplot(2, 3, 4)),
    )

    row2 = fig.add_subplot(2, 3, 2)
