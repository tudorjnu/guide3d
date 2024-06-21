import numpy as np
from scipy.optimize import least_squares


def project_points(P, points_3d):
    """
    Projects 3D points back to 2D using a projection matrix.
    """
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_2d_h = np.dot(P, points_3d_h.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2, np.newaxis]
    return points_2d


def reconstruction_error(params, points1, points2, P1, P2):
    """
    Computes the total reconstruction error for the current set of 3D points.
    """
    points_3d = params.reshape(-1, 3)
    proj_points1 = project_points(P1, points_3d)
    proj_points2 = project_points(P2, points_3d)
    error1 = np.linalg.norm(proj_points1 - points1, axis=1)
    error2 = np.linalg.norm(proj_points2 - points2, axis=1)
    return np.hstack((error1, error2))


def minimize_reconstruction_error(points1, points2, P1, P2, initial_points_3d):
    """
    Minimizes the reconstruction error to refine 3D points.
    """
    result = least_squares(
        reconstruction_error, initial_points_3d.ravel(), args=(points1, points2, P1, P2)
    )
    refined_points_3d = result.x.reshape(-1, 3)
    return refined_points_3d
