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


def make_figure():
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
    axs = axs.flatten()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
    plt.axis(False)
    return axs


def main():
    import vars
    import viz

    def get_data():
        import json

        frames = [100]
        dummy_data = []
        annotations = json.load(open("data/annotations/raw.json"))
        for video in annotations:
            for frame in video["frames"]:
                if frame["frame_number"] in frames:
                    dummy_data.append(
                        {
                            "img1": frame["camera1"]["image"],
                            "img2": frame["camera2"]["image"],
                            "pts1": np.array(frame["camera1"]["points"]),
                            "pts2": np.array(frame["camera2"]["points"]),
                        }
                    )
        return dummy_data

    dataset_path = vars.dataset_path
    data = get_data()

    axs = make_figure()
    for i, sample in enumerate(data[:6]):
        ax = axs[i]
        ax.axis("off")
        img = plt.imread(dataset_path / sample["img1"])
        pts = sample["pts1"]

        img = viz.convert_to_color(img)

        x = pts[:, 0]
        y = pts[:, 1]
        curve = parametrize_curve(pts, s=10)
        tck, u = curve

        print("Knots", tck[0])
        print("Control Points", tck[1])
        print("Knots Len", len(tck[0]))
        print("Control Points Len", len(tck[1][0]))

        control_points = tck[1]
        print("Number of control points:", len(tck[1][0]))

        x_fine, y_fine = splev(u, tck)

        ax.plot(x, y, "o-", label="Original", markersize=1, linewidth=0.5, alpha=0.7)
        ax.plot(x_fine, y_fine, "-", label="Curve", linewidth=1, alpha=0.7)
        ax.scatter(
            control_points[0], control_points[1], s=1, c="red", label="Control Points"
        )
        ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
