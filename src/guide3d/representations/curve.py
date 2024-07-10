import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep


def fit_spline(pts: np.ndarray, s: float = None, k: int = 3, eps: float = 1e-10):
    dims = pts.shape[1]
    if dims == 2:
        x = pts[:, 0]
        y = pts[:, 1]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + eps)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        tck, u = splprep([x, y], s=s, k=k, u=cumulative_distances)
        return tck, u
    elif dims == 3:
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2 + eps)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        tck, u = splprep([x, y, z], s=s, k=k, u=cumulative_distances)
        return tck, u
    else:
        raise ValueError("Input points must be 2D or 3D")


def sample_spline(tck: tuple, u: list, n: int = None, delta: float = None):
    assert delta or n, "Either delta or n must be provided"
    assert not (delta and n), "Only one of delta or n must be provided"

    def is2d(tck):
        return len(tck[1]) == 2

    u_max = u[-1]
    num_samples = int(u_max / delta) + 1 if delta else n
    u = np.linspace(0, u_max, num_samples)
    if is2d(tck):
        x, y = splev(u, tck)
        return np.column_stack([x, y]).astype(np.int32)
    else:
        x, y, z = splev(u, tck)
        return np.column_stack([x, y, z])


def main():
    import utils.viz as viz
    import vars
    from utils.utils import get_data

    dataset_path = vars.dataset_path
    data = get_data()

    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5, 10))
    axs = axs.flatten()
    plt.grid(True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
    plt.axis(False)
    for i, sample in enumerate(data[:6]):
        ax = axs[i]
        ax.axis("off")
        img = plt.imread(dataset_path / sample["img1"])
        pts = sample["pts1"]
        x = pts[:, 0]
        y = pts[:, 1]

        img = viz.convert_to_color(img)
        curve = fit_spline(pts)
        tck, u = curve
        control_points = tck[1]

        print("Knots", tck[0])
        print("Control Points", tck[1])
        print("Knots Len", len(tck[0]))
        print("Control Points Len", len(tck[1][0]))
        print("Number of control points:", len(tck[1][0]))

        x_fine, y_fine = splev(u, tck)
        ax.plot(x, y, "bo", label="Original points", **vars.plot_defaults)
        ax.plot(x_fine, y_fine, "g", label="Curve", **vars.plot_defaults)
        ax.plot(
            control_points[0],
            control_points[1],
            "yo",
            label="Control Points",
            **vars.plot_defaults,
        )
        ax.imshow(img)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        ncol=3,
        borderaxespad=0.1,
        handletextpad=0.1,
    )
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
