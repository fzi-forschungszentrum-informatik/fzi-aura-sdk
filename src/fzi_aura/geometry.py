from __future__ import annotations

import numpy as np


def transform_matrix(translation_xyz, rotation_xyzw) -> np.ndarray:
    quat = np.asarray(rotation_xyzw, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError("rotation_xyzw must contain exactly four values")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation_xyzw must be a finite, non-zero quaternion")
    x, y, z, w = quat / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return matrix


def invert_transform(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -(inv[:3, :3] @ matrix[:3, 3])
    return inv


def rotation_matrix_to_quaternion_xyzw(rotation) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation does not produce a finite quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def transform_points(points, matrix) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homog = np.hstack([points[:, :3], np.ones((len(points), 1), dtype=np.float64)])
    return (np.asarray(matrix, dtype=np.float64).reshape(4, 4) @ homog.T).T[:, :3]


def box_corners_lwh(center, size_lwh) -> np.ndarray:
    l, w, h = np.asarray(size_lwh, dtype=np.float64)  # noqa: E741
    x, y, z = l / 2.0, w / 2.0, h / 2.0
    corners = np.array(
        [
            [x, y, z],
            [x, -y, z],
            [-x, -y, z],
            [-x, y, z],
            [x, y, -z],
            [x, -y, -z],
            [-x, -y, -z],
            [-x, y, -z],
        ],
        dtype=np.float64,
    )
    return corners + np.asarray(center, dtype=np.float64)[None, :]


def project_points(points_cam, P) -> tuple[np.ndarray, np.ndarray]:
    points_cam = np.asarray(points_cam, dtype=np.float64)
    homog = np.hstack([points_cam, np.ones((len(points_cam), 1), dtype=np.float64)])
    proj = (np.asarray(P, dtype=np.float64).reshape(3, 4) @ homog.T).T
    depth = proj[:, 2]
    uv = proj[:, :2] / np.clip(depth[:, None], 1e-9, None)
    return uv, depth
