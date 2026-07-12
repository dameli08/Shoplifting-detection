import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SampledPoseWindowDataset, list_json_files, scan_tracks
from .model import BiLSTMAutoencoder
from .utils import count_parameters, save_json, select_device, set_seed


def _maybe_autobatch(
    model: torch.nn.Module,
    device: torch.device,
    window_size: int,
    feature_dim: int,
    mixed_precision: bool,
) -> int:
    if device.type != "cuda":
        return 64

    candidates = [1024, 768, 512, 384, 320, 256, 192, 160, 128, 96, 64, 48, 32]
    model = model.to(device)
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    best = 32
    for b in candidates:
        try:
            x = torch.randn(b, window_size, feature_dim, device=device)
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=mixed_precision):
                recon, _ = model(x)
                loss = ((recon - x) ** 2).mean()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            best = b
            del x, recon, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
    torch.cuda.empty_cache()
    return best


def _resolve_train_paths(cfg: Dict[str, Any]) -> str:
    data_cfg = cfg["data"]
    return os.path.join(data_cfg["dataset_root"], data_cfg["train_pose_dir"])


def train_main(cfg: Dict[str, Any], output_dir: str) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    device = select_device()
    train_pose_dir = _resolve_train_paths(cfg)
    pose_files = list_json_files(train_pose_dir)

    fcfg = cfg["features"]
    tcfg = cfg["train"]

    window_size = int(fcfg["window_size"])
    stride = int(fcfg["stride"])
    min_track_len = int(fcfg["min_track_len"])
    min_conf = float(fcfg["min_confidence"])

    print(f"[train] scanning tracks from {len(pose_files)} train JSON files")
    tracks, total_windows, feature_dim = scan_tracks(
        pose_files=pose_files,
        window_size=window_size,
        stride=stride,
        min_track_len=min_track_len,
        min_confidence=min_conf,
    )
    if not tracks:
        raise RuntimeError("No valid training tracks found after scan.")

    max_train_windows = int(tcfg["max_train_windows"])
    samples_per_epoch = min(max_train_windows, total_windows)

    ds = SampledPoseWindowDataset(
        tracks=tracks,
        window_size=window_size,
        stride=stride,
        min_confidence=min_conf,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    model = BiLSTMAutoencoder(
        input_dim=feature_dim,
        hidden_size=int(tcfg["hidden_size"]),
        latent_size=int(tcfg["latent_size"]),
        num_layers=int(tcfg["num_layers"]),
        dropout=float(tcfg["dropout"]),
    )

    mixed_precision = bool(tcfg["mixed_precision"]) and device.type == "cuda"
    batch_cfg = tcfg["batch_size"]
    if isinstance(batch_cfg, str) and batch_cfg.lower() == "auto":
        batch_size = _maybe_autobatch(model, device, window_size, feature_dim, mixed_precision)
    else:
        batch_size = int(batch_cfg)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    total_epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    early_stopping_patience = int(tcfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
    early_stopping_min_epochs = int(tcfg.get("early_stopping_min_epochs", 1))
    use_early_stopping = early_stopping_patience > 0
    epochs_without_improve = 0
    stopped_early = False
    stop_epoch = total_epochs

    train_losses = []
    best_loss = float("inf")
    best_path = os.path.join(output_dir, "best_model.pt")
    periodic_ckpt_every = int(tcfg.get("checkpoint_every_n_steps", 0))
    periodic_dir = os.path.join(output_dir, "checkpoints")
    if periodic_ckpt_every > 0:
        os.makedirs(periodic_dir, exist_ok=True)
    global_step = 0

    def _checkpoint_payload(loss_value: float) -> Dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "config": cfg,
            "feature_dim": feature_dim,
            "window_size": window_size,
            "stride": stride,
            "min_confidence": min_conf,
            "train_loss": float(loss_value),
            "batch_size": batch_size,
            "device": str(device),
            "global_step": int(global_step),
        }

    print(f"[train] device={device}, feature_dim={feature_dim}, batch_size={batch_size}, tracks={len(tracks)}, windows/epoch={samples_per_epoch}")
    print(f"[train] trainable_params={count_parameters(model):,}")

    for epoch in range(1, total_epochs + 1):
        model.train()
        running = 0.0
        n = 0

        pbar = tqdm(dl, desc=f"epoch {epoch}/{total_epochs}")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
                recon, _ = model(x)
                loss = ((recon - x) ** 2).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))
            global_step += 1
            pbar.set_postfix(loss=f"{running / max(n,1):.6f}")

            if periodic_ckpt_every > 0 and global_step % periodic_ckpt_every == 0:
                periodic_path = os.path.join(
                    periodic_dir,
                    f"epoch_{epoch:03d}_step_{global_step:07d}.pt",
                )
                torch.save(_checkpoint_payload(running / max(n, 1)), periodic_path)

        scheduler.step()
        epoch_loss = running / max(n, 1)
        train_losses.append(epoch_loss)

        if not np.isfinite(epoch_loss):
            print(f"[train] non-finite loss at epoch {epoch}, stopping early")
            stopped_early = True
            stop_epoch = epoch
            break

        if epoch_loss < (best_loss - early_stopping_min_delta):
            best_loss = epoch_loss
            epochs_without_improve = 0
            torch.save(_checkpoint_payload(epoch_loss), best_path)
        else:
            epochs_without_improve += 1

        if use_early_stopping and epoch >= early_stopping_min_epochs and epochs_without_improve >= early_stopping_patience:
            print(
                f"[train] early stopping at epoch {epoch} "
                f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
            )
            stopped_early = True
            stop_epoch = epoch
            break

    # Fit threshold from train reconstruction errors on a sampled subset.
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    model.eval()

    sample_eval_count = min(50000, len(ds))
    err_values = []

    with torch.no_grad():
        for i in tqdm(range(0, sample_eval_count, batch_size), desc="threshold calibration"):
            j = min(i + batch_size, sample_eval_count)
            x = torch.stack([ds[k] for k in range(i, j)], dim=0).to(device)
            _, per_win_error = model(x)
            err_values.append(per_win_error.detach().cpu().numpy())

    err_values = np.concatenate(err_values, axis=0) if err_values else np.asarray([0.0], dtype=np.float32)
    threshold = float(np.quantile(err_values, float(cfg["inference"]["train_threshold_quantile"])))

    ckpt = torch.load(best_path, map_location="cpu")
    ckpt["train_threshold"] = threshold
    ckpt["train_error_stats"] = {
        "mean": float(np.mean(err_values)),
        "std": float(np.std(err_values)),
        "q95": float(np.quantile(err_values, 0.95)),
        "q99": float(np.quantile(err_values, 0.99)),
        "q995": float(np.quantile(err_values, 0.995)),
    }
    torch.save(ckpt, best_path)

    summary = {
        "created_at": int(time.time()),
        "best_train_loss": float(best_loss),
        "epochs": total_epochs,
        "stopped_early": bool(stopped_early),
        "stop_epoch": int(stop_epoch),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_min_epochs": int(early_stopping_min_epochs),
        "checkpoint_every_n_steps": int(periodic_ckpt_every),
        "final_global_step": int(global_step),
        "batch_size": int(batch_size),
        "feature_dim": int(feature_dim),
        "tracks": int(len(tracks)),
        "total_scanned_windows": int(total_windows),
        "windows_per_epoch": int(samples_per_epoch),
        "train_threshold": threshold,
        "train_losses": [float(x) for x in train_losses],
    }

    save_json(os.path.join(output_dir, "train_summary.json"), summary)
    return best_path, summary
