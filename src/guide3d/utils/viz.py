import cv2
import guide3d.vars as vars
import numpy as np
import spectra as sp
from guide3d.representations import curve


def _check_points(points):
    assert points.ndim == 2 and points.shape[1] == 2, "Points should be in 2D format"
    assert points.dtype == np.int32, "Points should be in int32 format"


def _check_image(image):
    assert image.ndim == 3 and (
        image.shape[0] == 3 or image.shape[-1] == 3
    ), "Image is in grayscale format, use convert_to_color"
    assert image.dtype == np.uint8, "Image should be in uint8 format"


def str_to_tuple_color(color):
    color = sp.html(color).to("rgb").values
    color = tuple(map(lambda x: int(x * 255), color))
    return color


def convert_to_color(image):
    converted_image = image.copy()
    converted_image = cv2.cvtColor(converted_image, cv2.COLOR_GRAY2RGB)
    if converted_image.max() <= 1:
        converted_image = converted_image * 255
    converted_image = converted_image.astype(np.uint8)

    return converted_image


def draw_polyline(image, points, color=None, **kwargs):
    _check_image(image)
    _check_points(points)
    if not isinstance(color, tuple):
        color = str_to_tuple_color(color)
    image = np.copy(image)
    points = points.reshape((-1, 1, 2))
    if color is None:
        color = sp.html(vars.colors["polyline-colors"]).to("rgb").values
        color = tuple(map(lambda x: int(x * 255), color))
    cv2.polylines(image, [np.int32(points)], isClosed=False, color=color, **kwargs)
    return image


def draw_line(img, line, color, **kwargs):
    r, c, _ = img.shape
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, **kwargs)
    return img


def draw_lines(img, lines, colors=None):
    _check_image(img)
    img = np.copy(img)

    if colors is None:
        color = sp.html(vars.colors["matplotlib-colors"][0]).to("rgb").values
        color = tuple(map(lambda x: int(x * 255), color))
        colors = [color] * len(lines)

    r, c, _ = img.shape  # Updated to unpack the number of channels as well

    for r, color in zip(lines, colors):
        color = tuple(map(int, color))
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img


def draw_points(image, points, colors=None):
    _check_image(image)
    _check_points(points)
    image = np.copy(image)

    if colors is None:
        color = sp.html(vars.colors["matplotlib-colors"][0]).to("rgb").values
        color = tuple(map(lambda x: int(x * 255), color))
        colors = [color] * points.shape[0]

    for point, color in zip(points, colors):
        point = tuple(map(int, point))
        color = tuple(map(int, color))
        cv2.circle(image, point, 5, color, -1)
    return image


def draw_curve(img, tck, u, **kwargs):
    pts = curve.sample_spline(tck, u, delta=10)
    img = draw_polyline(img, pts, **kwargs)
    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = np.zeros((512, 512, 3), np.uint8)
    points = np.array([[10, 10], [100, 100], [200, 200], [300, 300]], np.int32)
    image = draw_polyline(image, points)
    plt.imshow(image)
    plt.show()
