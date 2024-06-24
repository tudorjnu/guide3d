import numpy as np


def refine_fundamental_matrix(F):
    def get_data():
        import json

        pts1 = []
        pts2 = []

        annotations = json.load(open("data/annotations/raw.json"))
        for video in annotations:
            for frame in video["frames"]:
                pts1.append(frame["camera1"]["points"][0])
                pts2.append(frame["camera2"]["points"][0])
        return np.array(pts1), np.array(pts2)

    points1, points2 = get_data()

    # Convert points to homogeneous coordinates
    points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])
    points2_h = np.hstack([points2, np.ones((points2.shape[0], 1))])

    # Optimize F using the reprojection error
    def reprojection_error(params, points1_h, points2_h):
        F = params.reshape(3, 3)
        errors = []
        for p1, p2 in zip(points1_h, points2_h):
            error = p2 @ F @ p1.T
            errors.append(error)
        return np.array(errors)

    from scipy.optimize import least_squares

    res = least_squares(reprojection_error, F.flatten(), args=(points1_h, points2_h))
    F_refined = res.x.reshape(3, 3)
    return F_refined
