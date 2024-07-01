import representations.curve as curve
import utils.viz as viz


def draw_curve(img, tck, u_max, **kwargs):
    pts = curve.sample_spline_n(tck, 0, u_max, 50)
    img = viz.draw_polyline(img, pts, **kwargs)
    return img
