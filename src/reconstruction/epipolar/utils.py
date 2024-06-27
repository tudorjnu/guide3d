import numpy as np
from scipy.interpolate import splev
from scipy.optimize import minimize_scalar
from shapely.geometry import LineString


def find_closest_point_on_spline_to_line(
    ln, tck, u_min, u_max, delta=2, alpha=10.0, eps=1e-10
):
    # Composite distance function to minimize
    def composite_distance(u, ln, tck, u_min, alpha):
        # Distance from the point on the spline to the line
        x, y = splev(u, tck)
        a, b, c = ln
        distance_to_line = np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

        # Distance from u to u_min
        distance_to_u_min = np.abs(u - u_min)

        # Composite distance with weight alpha
        return distance_to_line + distance_to_u_min

    # Define the constrained bounds
    constrained_u_min = u_min + delta
    constrained_u_max = u_max

    try:
        # Use minimize_scalar to find the point that minimizes the composite distance
        result = minimize_scalar(
            composite_distance,
            args=(ln, tck, u_min, alpha),
            bounds=(constrained_u_min, constrained_u_max),
            method="bounded",
            options={"xatol": eps},
        )

        return result.x
    except Exception as e:
        # Handle any exception that might occur during the optimization process
        print(f"Optimization failed: {e}")
        return None


def find_intersections(segment1, segment2, verbose=False):
    polyline1 = LineString(segment1)
    polyline2 = LineString(segment2)

    intersection = polyline1.intersection(polyline2)

    if intersection.is_empty:
        if verbose:
            print("No intersection found")
        return None
    elif intersection.geom_type == "MultiPoint":
        pts = [np.array((point.x, point.y)) for point in intersection.geoms]
        if verbose:
            print("Number of intersections:", len(pts))
        return pts
    elif intersection.geom_type == "Point":
        pts = [np.array((intersection.x, intersection.y))]
        if verbose:
            print("Number of intersections:", 1)
        return pts
    else:
        return None
