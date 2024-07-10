import numpy as np

import calibration


def get_data():
    import json

    pts1_list = []
    pts2_list = []

    annotations = json.load(open("data/annotations/raw.json"))
    for video in annotations:
        for frame in video["frames"]:
            pts1_list.append(frame["camera1"]["points"][0])
            pts2_list.append(frame["camera2"]["points"][0])
    return np.array(pts1_list), np.array(pts2_list)


def main():
    points1, points2 = get_data()

    # Step 2: Estimate the Fundamental Matrix using RANSAC
    F = calibration.F

    # Step 3: Refine the Fundamental Matrix using non-linear optimization (optional)
    # OpenCV's findFundamentalMat with RANSAC already provides a robust estimation
    # Further refinement can be done if needed

    print("Estimated Fundamental Matrix:", F)

    # Optional: refine the fundamental matrix using least squares
    def refine_fundamental_matrix(F, points1, points2):
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

        res = least_squares(
            reprojection_error, F.flatten(), args=(points1_h, points2_h)
        )
        F_refined = res.x.reshape(3, 3)
        return F_refined

    F_refined = refine_fundamental_matrix(F, points1, points2)

    print("Refined Fundamental Matrix:", F_refined)
    pass


if __name__ == "__main__":
    main()
