import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import build_windows_for_track, load_feature_map, pair_pose_and_gt
from .metrics import best_threshold_by_hprs, compute_binary_metrics, compute_score_curves
from .model import BiLSTMAutoencoder
from .utils import save_json, select_device


def _score_single_video(
    model: torch.nn.Module,
    device: torch.device,
    pose_path: str,
    min_confidence: float,
    window_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    fmap = load_feature_map(pose_path, min_confidence)
    if not fmap:
        return np.zeros((0,), dtype=np.float32)

    person_curves: List[np.ndarray] = []

    for _pid, (frame_ids, feat) in fmap.items():
        windows, frame_windows = build_windows_for_track(feat, frame_ids, window_size, stride)
        if windows.shape[0] == 0:
            continue

        all_scores: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, windows.shape[0], batch_size):
                w = torch.from_numpy(windows[i : i + batch_size]).to(device)
                _, per_err = model(w)
                all_scores.append(per_err.detach().cpu().numpy())

        win_scores = np.concatenate(all_scores, axis=0)

        frame_score_map: Dict[int, List[float]] = {}
        for s, fwin in zip(win_scores, frame_windows):
            for f in fwin:
                frame_score_map.setdefault(int(f), []).append(float(s))

        max_frame = max(frame_score_map.keys()) if frame_score_map else -1
        curve = np.zeros((max_frame + 1,), dtype=np.float32)
        for f, vals in frame_score_map.items():
            # Conservative per-person aggregation over overlapping windows.
            curve[f] = float(np.max(vals))

        person_curves.append(curve)

    if not person_curves:
        return np.zeros((0,), dtype=np.float32)

    max_len = max(len(x) for x in person_curves)
    merged = np.zeros((max_len,), dtype=np.float32)
    for curve in person_curves:
        if len(curve) < max_len:
            curve = np.pad(curve, (0, max_len - len(curve)), mode="constant", constant_values=0.0)
        merged = np.maximum(merged, curve)
    return merged


def _collect_split_scores(
    model: torch.nn.Module,
    device: torch.device,
    pairs: List[Tuple[str, str]],
    min_confidence: float,
    window_size: int,
    stride: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_true: List[np.ndarray] = []
    all_score: List[np.ndarray] = []

    for pose_path, gt_path in tqdm(pairs, desc=f"scoring {len(pairs)} videos"):
        y_true = np.load(gt_path).astype(np.int32)
        y_score = _score_single_video(
            model=model,
            device=device,
            pose_path=pose_path,
            min_confidence=min_confidence,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
        )

        if y_score.shape[0] < y_true.shape[0]:
            y_score = np.pad(y_score, (0, y_true.shape[0] - y_score.shape[0]), mode="constant", constant_values=0.0)
        elif y_score.shape[0] > y_true.shape[0]:
            y_score = y_score[: y_true.shape[0]]

        all_true.append(y_true)
        all_score.append(y_score)

    if not all_true:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    return np.concatenate(all_true, axis=0), np.concatenate(all_score, axis=0)


def evaluate_main(cfg: Dict[str, Any], checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    c_cfg = ckpt["config"]

    # Runtime config comes from checkpoint for model consistency.
    window_size = int(ckpt["window_size"])
    stride = int(ckpt["stride"])
    min_conf = float(ckpt["min_confidence"])
    feature_dim = int(ckpt["feature_dim"])
    threshold_unsup = float(ckpt.get("train_threshold", 0.0))

    device = select_device()
    model = BiLSTMAutoencoder(
        input_dim=feature_dim,
        hidden_size=int(c_cfg["train"]["hidden_size"]),
        latent_size=int(c_cfg["train"]["latent_size"]),
        num_layers=int(c_cfg["train"]["num_layers"]),
        dropout=float(c_cfg["train"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch_size = int(ckpt.get("batch_size", 128))

    dcfg = cfg["data"]
    root = dcfg["dataset_root"]

    staged_pairs = pair_pose_and_gt(
        os.path.join(root, dcfg["staged_pose_dir"]),
        os.path.join(root, dcfg["staged_gt_dir"]),
    )
    real_pairs = pair_pose_and_gt(
        os.path.join(root, dcfg["real_pose_dir"]),
        os.path.join(root, dcfg["real_gt_dir"]),
    )

    y_true_staged, y_score_staged = _collect_split_scores(
        model=model,
        device=device,
        pairs=staged_pairs,
        min_confidence=min_conf,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
    )
    y_true_real, y_score_real = _collect_split_scores(
        model=model,
        device=device,
        pairs=real_pairs,
        min_confidence=min_conf,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
    )

    # Threshold 1: unsupervised from train distribution.
    staged_unsup = compute_binary_metrics(y_true_staged, y_score_staged, threshold_unsup) if y_true_staged.size else {}
    real_unsup = compute_binary_metrics(y_true_real, y_score_real, threshold_unsup) if y_true_real.size else {}

    # Threshold 2: best HPRS on staged split (for tuned operating point).
    if y_true_staged.size:
        tuned_t = best_threshold_by_hprs(y_true_staged, y_score_staged)
    else:
        tuned_t = threshold_unsup

    staged_tuned = compute_binary_metrics(y_true_staged, y_score_staged, tuned_t) if y_true_staged.size else {}
    real_tuned = compute_binary_metrics(y_true_real, y_score_real, tuned_t) if y_true_real.size else {}

    staged_curves = compute_score_curves(y_true_staged, y_score_staged) if y_true_staged.size else {}
    real_curves = compute_score_curves(y_true_real, y_score_real) if y_true_real.size else {}

    report = {
        "checkpoint": checkpoint_path,
        "window_size": window_size,
        "stride": stride,
        "min_confidence": min_conf,
        "num_staged_videos": len(staged_pairs),
        "num_realworld_videos": len(real_pairs),
        "threshold_unsupervised": threshold_unsup,
        "threshold_tuned_on_staged": float(tuned_t),
        "staged_unsupervised": staged_unsup,
        "realworld_unsupervised": real_unsup,
        "staged_tuned": staged_tuned,
        "realworld_tuned": real_tuned,
        "staged_curves": staged_curves,
        "realworld_curves": real_curves,
        "primary_metric": cfg["eval"]["primary_metric"],
    }

    save_json(os.path.join(output_dir, "eval_report.json"), report)
    return report
