import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev

import plot
import utils
import vars

data = utils.parse(vars.dataset_path / "annotations.xml")
data = utils.flatten(data)
print(len(data))

i = 2
img = plt.imread(vars.dataset_path / data[i]["img"])
pts = data[i]["reconstruction"]
print("Points:", pts.shape)
pts_random = np.random.rand(30, 2)


tck, u = utils.parametrize_curve(pts, s=0.1, k=3)
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]


# Evaluate the spline at more points for a smooth curve
u_fine = np.linspace(0, u[-1], 1000)
x_fine, y_fine, z_fine = splev(u_fine, tck)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

mesh = plot.get_mesh()
plot.plot_mesh(mesh, ax)

plt.plot(x, y, z, "o-", label="Original Polyline", linewidth=1, markersize=2)
for s, ls in zip([0.5, 1.0], ["--", "-.", ":"]):
    tck, u = utils.parametrize_curve(pts, s=s, k=3)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    knots = np.array(splev(tck[0], tck)).T
    print("Knots:", knots.shape)
    plt.scatter(
        knots[:, 0],
        knots[:, 1],
        knots[:, 2],
        label=f"Knots s={s}",
    )
    plt.plot(
        x_fine,
        y_fine,
        z_fine,
        "-",
        label=f"Smooth Curve s={s}",
        linewidth=1,
        linestyle=ls,
    )
    break
plt.title("Polyline and Smooth Curve Representation")
plt.legend()
plt.grid(True)
plt.show()
