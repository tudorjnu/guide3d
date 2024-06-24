from pathlib import Path

import numpy as np
from scipy.linalg import null_space, pinv

from calibration.refinement import refine_fundamental_matrix


def parse_calibration_file(filepath: Path):
    data = {}
    section = None

    if not filepath.exists():
        print("Filepath doesn't exist", filepath)
        return

    with open(filepath, "r") as file:
        for line in file:
            stripped_line = line.strip()

            if line == "\n":
                continue

            # Checking if the line is a section header
            if stripped_line in {
                "image size",
                "camera matrix",
                "rotation",
                "translation",
                "undistortion",
            }:
                section = stripped_line
                data[section] = []
            elif section:
                # Special handling for 'image size' because it's a single line of values
                if section == "image size":
                    data[section] = [int(x) for x in stripped_line.split(",")]
                else:
                    # For other sections, append each line of values as a new list
                    data[section].append([float(x) for x in stripped_line.split(",")])

    # For 'undistortion', we expect a single list of values, not a list of lists
    if "undistortion" in data:
        # Flatten the list of lists into a single list
        data["undistortion"] = [
            item for sublist in data["undistortion"] for item in sublist
        ]

    return data


def get_projection_matrix(camera_matrix, rotation, translation):
    return np.matmul(camera_matrix, np.hstack((rotation, translation)))


def get_cross_product_matrix(vector):
    assert vector.shape == (3,), "Translation vector must be a 3-element vector."
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


def get_essential_matrix(rotation, translation):
    """Compute the Essential Matrix from rotation and translation."""
    t_cross = get_cross_product_matrix(translation)
    return t_cross @ rotation


def get_fundamental_matrix(K1, K2, t1, t2, R1, R2):
    R = R2 @ R1.T
    t = t2 - (R @ t1)

    T_x = skew_symmetric_matrix(t)

    F = np.linalg.inv(K2).T @ T_x @ R @ np.linalg.inv(K1)
    return F


def skew_symmetric(v):
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def compute_fundamental_matrix(P_A, P_B):
    # Compute the camera center o_A of P_A
    o_A = null_space(P_A)
    o_A = o_A / o_A[-1]  # Ensure homogeneous coordinates

    # Project the camera center o_A into the second image
    o_A_tilde = P_B @ o_A

    # Construct the skew-symmetric matrix [P_B * o_A]_x
    skew_matrix = skew_symmetric_matrix(o_A_tilde[:3])

    # Compute the pseudo-inverse of P_A
    P_A_pinv = pinv(P_A)

    # Compute the fundamental matrix F_A
    F_A = skew_matrix @ P_B @ P_A_pinv

    return F_A


def skew_symmetric_matrix(vector):
    vector = np.array(vector).flatten()
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


calibration_path = Path(__file__).parent / "2024-03-04" / "norm"

P1_data = parse_calibration_file(calibration_path / "16710.txt")
R1 = np.array(P1_data["rotation"], np.float64)
t1 = np.array(P1_data["translation"], np.float64)
K1 = np.array(P1_data["camera matrix"], np.float64)
P1 = get_projection_matrix(K1, R1, t1)
d1 = np.array(P1_data["undistortion"]) if "undistortion" in P1_data else None

P2_data = parse_calibration_file(calibration_path / "16709.txt")
R2 = np.array(P2_data["rotation"], np.float64)
t2 = np.array(P2_data["translation"], np.float64)
K2 = np.array(P2_data["camera matrix"], np.float64)
P2 = get_projection_matrix(K2, R2, t2)
d2 = np.array(P2_data["undistortion"]) if "undistortion" in P2_data else None

E = get_essential_matrix(R1, t1.flatten())
F = get_fundamental_matrix(K1, K2, t1, t2, R1, R2)
F = refine_fundamental_matrix(F)
F_A_B = compute_fundamental_matrix(P1, P2)
F_B_A = compute_fundamental_matrix(P2, P1)
# print("F1\n", F1)
# print("F2\n", F2)


if __name__ == "__main__":
    import cv2

    camera_paths = Path(__file__).parent / "2024-03-04" / "matrices"
    h, w = 1024, 1024
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K1, d1, (w, h), 1, (w, h))
    newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(K2, d2, (w, h), 1, (w, h))

    newcameramtx = np.round(newcameramtx, 3)

    print(K1)
    print(newcameramtx)
