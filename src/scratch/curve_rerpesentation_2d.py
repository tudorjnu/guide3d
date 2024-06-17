import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep

import utils
import vars
import viz

data = utils.parse(vars.dataset_path / "annotations.xml")
data = utils.flatten(data)
print(len(data))

i = 2
img = plt.imread(vars.dataset_path / data[i]["img"])
pts = data[i]["pts"]
pts_random = np.random.rand(30, 2)

img = viz.convert_to_color(img)

# Separate the points into x and y coordinates
x = pts[:, 0]
y = pts[:, 1]

# Use splprep to create the spline representation of the curve
tck, u = splprep([x, y], s=0, k=3)

# Evaluate the spline at more points for a smooth curve
u_fine = np.linspace(0, 1, 1000)
x_fine, y_fine = splev(u_fine, tck)

# Plot the original polyline and the smooth curve
plt.figure(figsize=(8, 8))
plt.plot(x, y, "o-", label="Original Polyline")  # Polyline points
plt.plot(x_fine, y_fine, "-", label="Smooth Curve")  # Smooth curve
plt.imshow(img)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polyline and Smooth Curve Representation")
plt.legend()
plt.grid(True)
plt.show()
