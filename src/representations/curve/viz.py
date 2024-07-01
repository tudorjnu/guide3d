import representations.curve as curve
import utils.viz as viz


def draw_curve(img, tck, u, **kwargs):
    pts = curve.sample_spline(tck, u, delta=10)
    img = viz.draw_polyline(img, pts, **kwargs)
    return img
