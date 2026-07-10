from typing import Dict, List, Tuple

import numpy as np

# COCO-17 joint indexes
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def _as_xyc(keypoints: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.shape[0] != 51:
        raise ValueError(f"Expected 51 values for COCO17 XYC keypoints, got {arr.shape[0]}")
    arr = arr.reshape(17, 3)
    xy = arr[:, :2]
    conf = np.clip(arr[:, 2], 0.0, 1.0)
    return xy, conf


def _safe_center(points: np.ndarray, conf: np.ndarray, min_conf: float) -> np.ndarray:
    mask = conf >= min_conf
    if np.any(mask):
        return points[mask].mean(axis=0)
    return points.mean(axis=0)


def _safe_scale(xy: np.ndarray, shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
    torso = np.linalg.norm(shoulder_center - hip_center)
    if torso > 1e-5:
        return float(torso)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy)
    if diag > 1e-5:
        return float(diag)
    return 1.0


def _frame_relational(norm_xy: np.ndarray) -> np.ndarray:
    l_wrist = norm_xy[LEFT_WRIST]
    r_wrist = norm_xy[RIGHT_WRIST]
    l_hip = norm_xy[LEFT_HIP]
    r_hip = norm_xy[RIGHT_HIP]
    torso_center = 0.5 * (0.5 * (norm_xy[LEFT_SHOULDER] + norm_xy[RIGHT_SHOULDER]) + 0.5 * (l_hip + r_hip))

    feats = np.asarray(
        [
            np.linalg.norm(l_wrist - l_hip),
            np.linalg.norm(r_wrist - r_hip),
            np.linalg.norm(l_wrist - torso_center),
            np.linalg.norm(r_wrist - torso_center),
            np.linalg.norm(l_wrist - r_wrist),
        ],
        dtype=np.float32,
    )
    return feats


def build_person_feature_sequence(
    frame_map: Dict[str, Dict[str, List[float]]],
    min_confidence: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    frame_ids = sorted([int(k) for k in frame_map.keys()])
    if not frame_ids:
        return np.empty((0,), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    norm_xy_seq = []
    conf_seq = []
    rel_seq = []
    used_frame_ids = []

    for fid in frame_ids:
        rec = frame_map[str(fid)]
        if "keypoints" not in rec:
            continue
        xy, conf = _as_xyc(rec["keypoints"])

        hip_center = _safe_center(xy[[LEFT_HIP, RIGHT_HIP]], conf[[LEFT_HIP, RIGHT_HIP]], min_confidence)
        shoulder_center = _safe_center(
            xy[[LEFT_SHOULDER, RIGHT_SHOULDER]],
            conf[[LEFT_SHOULDER, RIGHT_SHOULDER]],
            min_confidence,
        )
        scale = _safe_scale(xy, shoulder_center, hip_center)

        norm_xy = (xy - hip_center[None, :]) / scale
        rel = _frame_relational(norm_xy)

        norm_xy_seq.append(norm_xy)
        conf_seq.append(conf)
        rel_seq.append(rel)
        used_frame_ids.append(fid)

    if not norm_xy_seq:
        return np.empty((0,), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    used_frame_ids = np.asarray(used_frame_ids, dtype=np.int32)

    norm_xy_arr = np.asarray(norm_xy_seq, dtype=np.float32)  # [T, 17, 2]
    conf_arr = np.asarray(conf_seq, dtype=np.float32)  # [T, 17]
    rel_arr = np.asarray(rel_seq, dtype=np.float32)  # [T, 5]

    vel = np.zeros_like(norm_xy_arr, dtype=np.float32)
    vel[1:] = norm_xy_arr[1:] - norm_xy_arr[:-1]

    acc = np.zeros_like(norm_xy_arr, dtype=np.float32)
    acc[1:] = vel[1:] - vel[:-1]

    feat = np.concatenate(
        [
            norm_xy_arr.reshape(norm_xy_arr.shape[0], -1),
            vel.reshape(vel.shape[0], -1),
            acc.reshape(acc.shape[0], -1),
            rel_arr,
            conf_arr,
        ],
        axis=1,
    )

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return used_frame_ids, feat
