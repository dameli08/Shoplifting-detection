import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import build_person_feature_sequence


@dataclass(frozen=True)
class TrackRef:
    pose_path: str
    person_id: str
    num_windows: int


def list_json_files(folder: str) -> List[str]:
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".json")
    ])


def pair_pose_and_gt(pose_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    pose = {os.path.splitext(x)[0]: os.path.join(pose_dir, x) for x in os.listdir(pose_dir) if x.endswith('.json')}
    gt = {os.path.splitext(x)[0]: os.path.join(gt_dir, x) for x in os.listdir(gt_dir) if x.endswith('.npy')}
    keys = sorted(set(pose.keys()).intersection(gt.keys()))
    return [(pose[k], gt[k]) for k in keys]


@lru_cache(maxsize=128)
def _cached_feature_map(pose_path: str, min_confidence: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    with open(pose_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for person_id, frame_map in payload.items():
        frame_ids, feat = build_person_feature_sequence(frame_map, min_confidence=min_confidence)
        if feat.shape[0] > 0:
            out[str(person_id)] = (frame_ids, feat)
    return out


def load_feature_map(pose_path: str, min_confidence: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return _cached_feature_map(pose_path, float(min_confidence))


def scan_tracks(
    pose_files: Sequence[str],
    window_size: int,
    stride: int,
    min_track_len: int,
    min_confidence: float,
) -> Tuple[List[TrackRef], int, int]:
    tracks: List[TrackRef] = []
    total_windows = 0
    feature_dim = -1

    for p in pose_files:
        fmap = load_feature_map(p, min_confidence)
        for pid, (_, feat) in fmap.items():
            t = int(feat.shape[0])
            if feature_dim < 0:
                feature_dim = int(feat.shape[1])
            if t < max(window_size, min_track_len):
                continue
            n_w = 1 + (t - window_size) // stride
            tracks.append(TrackRef(pose_path=p, person_id=pid, num_windows=n_w))
            total_windows += n_w

    if feature_dim < 0:
        feature_dim = 0
    return tracks, total_windows, feature_dim


class SampledPoseWindowDataset(Dataset):
    def __init__(
        self,
        tracks: Sequence[TrackRef],
        window_size: int,
        stride: int,
        min_confidence: float,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        if not tracks:
            raise ValueError("No valid tracks were found for training.")
        self.tracks = list(tracks)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.min_confidence = float(min_confidence)
        self.samples_per_epoch = int(samples_per_epoch)
        self.rng = np.random.default_rng(seed)

        weights = np.asarray([t.num_windows for t in self.tracks], dtype=np.float64)
        weights = weights / weights.sum()
        self.track_indices = self.rng.choice(len(self.tracks), size=self.samples_per_epoch, replace=True, p=weights)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        ref = self.tracks[int(self.track_indices[idx])]
        fmap = load_feature_map(ref.pose_path, self.min_confidence)
        _, feat = fmap[ref.person_id]

        max_start = feat.shape[0] - self.window_size
        if max_start < 0:
            raise RuntimeError("Encountered track shorter than window size during sampling.")

        # Convert window index to start by stride to match scan_tracks window count.
        win_idx = self.rng.integers(0, ref.num_windows)
        start = int(win_idx * self.stride)
        window = feat[start : start + self.window_size]
        return torch.from_numpy(window.astype(np.float32))


def build_windows_for_track(
    feat: np.ndarray,
    frame_ids: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    t = feat.shape[0]
    if t == 0:
        return np.empty((0, window_size, feat.shape[1]), dtype=np.float32), []

    starts: List[int] = []
    if t <= window_size:
        starts = [0]
    else:
        starts = list(range(0, t - window_size + 1, stride))
        if starts[-1] != t - window_size:
            starts.append(t - window_size)

    windows = []
    frame_windows: List[np.ndarray] = []
    for s in starts:
        e = s + window_size
        chunk_feat = feat[s:e]
        chunk_frames = frame_ids[s:e]

        if chunk_feat.shape[0] < window_size:
            pad = window_size - chunk_feat.shape[0]
            chunk_feat = np.pad(chunk_feat, ((0, pad), (0, 0)), mode="edge")
            chunk_frames = np.pad(chunk_frames, (0, pad), mode="edge")

        windows.append(chunk_feat)
        frame_windows.append(chunk_frames)

    return np.asarray(windows, dtype=np.float32), frame_windows
