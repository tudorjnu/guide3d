import cv2
import guide3d.calibration as calibration
import numpy as np
from scipy.optimize import minimize


def gauss_newton_triangulation(
    points1, points2, P1, P2, max_iterations=100, tolerance=1e-8
):
    def project_point(P, point_3d):
        """Project a 3D point using a projection matrix P."""
        point_3d_h = np.append(point_3d, 1)  # Convert to homogeneous coordinates
        projected = P @ point_3d_h
        return projected[:2] / projected[2]

    def compute_jacobian(P, point_3d):
        """Compute the Jacobian of the projection function."""
        X, Y, Z = point_3d
        jacobian = np.zeros((2, 3))

        projected = P @ np.array([X, Y, Z, 1])
        u, v, w = projected

        jacobian[0, 0] = P[0, 0] / w - (u * P[2, 0]) / (w * w)
        jacobian[0, 1] = P[0, 1] / w - (u * P[2, 1]) / (w * w)
        jacobian[0, 2] = P[0, 2] / w - (u * P[2, 2]) / (w * w)

        jacobian[1, 0] = P[1, 0] / w - (v * P[2, 0]) / (w * w)
        jacobian[1, 1] = P[1, 1] / w - (v * P[2, 1]) / (w * w)
        jacobian[1, 2] = P[1, 2] / w - (v * P[2, 2]) / (w * w)

        return jacobian

    num_points = points1.shape[0]
    points_3d = np.zeros((num_points, 3))

    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]

        # Initial guess using midpoint method
        A = np.array(
            [
                x1 * P1[2, :] - P1[0, :],
                y1 * P1[2, :] - P1[1, :],
                x2 * P2[2, :] - P2[0, :],
                y2 * P2[2, :] - P2[1, :],
            ]
        )
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        point_3d = X[:3] / X[3]

        for iteration in range(max_iterations):
            J = np.vstack(
                [compute_jacobian(P1, point_3d), compute_jacobian(P2, point_3d)]
            )

            f = np.hstack(
                [
                    project_point(P1, point_3d) - np.array([x1, y1]),
                    project_point(P2, point_3d) - np.array([x2, y2]),
                ]
            )

            delta = np.linalg.lstsq(J, -f, rcond=None)[0]
            point_3d += delta

            if np.linalg.norm(delta) < tolerance:
                break

        points_3d[i] = point_3d

    return points_3d


def triangulate_openCV(pts1, pts2, P1, P2):
    pts1 = cv2.undistortPoints(
        pts1, calibration.K1, calibration.d1, None, calibration.K1
    )
    pts2 = cv2.undistortPoints(
        pts2, calibration.K2, calibration.d2, None, calibration.K2
    )
    pts3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d /= pts3d[3]
    return pts3d[:3].T


def optimal_solution(ptsA, ptsB, P1, P2):
    """
    Reconstruct 3D points using the method of optimal triangulation.
    """
    reconstructed_pts = []
    for i in range(ptsA.shape[0]):
        x = ptsA[i]
        x_prime = ptsB[i]

        # Epipolar line in the second image corresponding to x in the first image
        l_prime = np.dot(calibration.F_A_B, np.append(x, 1))
        # Epipolar line in the first image corresponding to x_prime in the second image
        l = np.dot(calibration.F_B_A, np.append(x_prime, 1))

        # Normalize the lines
        l_prime /= np.sqrt(l_prime[0] ** 2 + l_prime[1] ** 2)
        l /= np.sqrt(l[0] ** 2 + l[1] ** 2)

        # Compute the distances from points to epipolar lines
        d_x_l = np.abs(np.dot(l, np.append(x, 1))) / np.sqrt(l[0] ** 2 + l[1] ** 2)
        d_x_prime_l_prime = np.abs(np.dot(l_prime, np.append(x_prime, 1))) / np.sqrt(
            l_prime[0] ** 2 + l_prime[1] ** 2
        )

        # Function to minimize
        def distance_function(t):
            l_t = l + t * l_prime
            l_prime_t = l_prime + t * l
            d = d_x_l**2 + d_x_prime_l_prime**2
            return d

        # Find the t that minimizes the distance function
        t_values = np.linspace(-10, 10, 1000)
        distances = [distance_function(t) for t in t_values]
        t_optimal = t_values[np.argmin(distances)]

        # Compute the optimal points on epipolar lines
        l_optimal = l + t_optimal * l_prime
        l_prime_optimal = l_prime + t_optimal * l

        x_optimal = np.array(
            [x[0] - l_optimal[0] * l_optimal[2], x[1] - l_optimal[1] * l_optimal[2]]
        )
        x_prime_optimal = np.array(
            [
                x_prime[0] - l_prime_optimal[0] * l_prime_optimal[2],
                x_prime[1] - l_prime_optimal[1] * l_prime_optimal[2],
            ]
        )

        # Triangulate to find the 3D point
        point_3d = triangulate_point(x_optimal, x_prime_optimal, P1, P2)
        reconstructed_pts.append(point_3d)

    # transform from homogenous coordinates
    reconstructed_pts = np.array(reconstructed_pts)
    reconstructed_pts = reconstructed_pts[:, :3] / reconstructed_pts[:, 3, None]

    return reconstructed_pts


def reconstruct(ptsA, ptsB, PA, PB, F):
    """
    Reconstruct the 3D points from noisy 2D correspondences.

    Parameters:
    ptsA (ndarray): Noisy points in the first image (shape: Nx2).
    ptsB (ndarray): Noisy points in the second image (shape: Nx2).
    PA (ndarray): Projection matrix for the first image (shape: 3x4).
    PB (ndarray): Projection matrix for the second image (shape: 3x4).
    F (ndarray): Fundamental matrix between the two images (shape: 3x3).

    Returns:
    X (ndarray): Reconstructed 3D points (shape: Nx3).
    """

    def geometric_error(x_hat, x, F):
        x1, x2 = x_hat[:2], x_hat[2:]
        x1_h = np.append(x1, 1)  # Convert to homogeneous coordinates
        x2_h = np.append(x2, 1)
        error = np.sum((x1 - x[:2]) ** 2) + np.sum((x2 - x[2:]) ** 2)
        epipolar_constraint = np.dot(x2_h.T, np.dot(F, x1_h))
        return error + epipolar_constraint**2

    def triangulate_point(x1, x2, P1, P2):
        A = np.array(
            [
                (x1[0] * P1[2] - P1[0]),
                (x1[1] * P1[2] - P1[1]),
                (x2[0] * P2[2] - P2[0]),
                (x2[1] * P2[2] - P2[1]),
            ]
        )
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    X = []
    for i in range(ptsA.shape[0]):
        x = np.hstack((ptsA[i], ptsB[i]))
        initial_guess = x.copy()
        result = minimize(
            geometric_error,
            initial_guess,
            args=(x, F),
            method="BFGS",
        )
        x_hat = result.x
        x1_hat, x2_hat = x_hat[:2], x_hat[2:]
        X_hat = triangulate_point(x1_hat, x2_hat, PA, PB)
        X.append(X_hat)

    return np.array(X)


triangulate = reconstruct
