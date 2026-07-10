import argparse
import os
import time
from typing import Any, Dict

from .evaluate import evaluate_main
from .train import train_main
from .utils import load_yaml


def _load_cfg(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    required = ["data", "features", "train", "inference", "eval"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing config section: {k}")
    return cfg


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    out_dir = args.output_dir or os.path.join("runs", time.strftime("exp_%Y%m%d_%H%M%S"))
    ckpt_path, summary = train_main(cfg, out_dir)
    print("[done] train complete")
    print(f"[done] checkpoint: {ckpt_path}")
    print(f"[done] threshold: {summary['train_threshold']:.6f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    out_dir = args.output_dir or os.path.join("runs", time.strftime("eval_%Y%m%d_%H%M%S"))
    report = evaluate_main(cfg, args.checkpoint, out_dir)
    print("[done] evaluation complete")
    print(f"[done] report: {os.path.join(out_dir, 'eval_report.json')}")
    for split_name in ("staged_unsupervised", "staged_tuned", "realworld_unsupervised", "realworld_tuned"):
        split = report.get(split_name)
        if not split:
            continue
        print(
            f"[metric] {split_name}: "
            f"HPRS={split['hprs']:.4f} "
            f"F1={split['f1']:.4f} "
            f"Precision={split['precision']:.4f} "
            f"Recall={split['recall']:.4f} "
            f"Specificity={split['specificity']:.4f} "
            f"PR-AUC={split['pr_auc']:.4f} "
            f"ROC-AUC={split['roc_auc']:.4f}"
        )
    print(f"[metric] PR/ROC curve data saved in: {os.path.join(out_dir, 'eval_report.json')}")


def cmd_full(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    base_dir = args.output_dir or os.path.join("runs", time.strftime("full_%Y%m%d_%H%M%S"))
    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    eval_dir = os.path.join(base_dir, "eval")

    ckpt_path, _summary = train_main(cfg, train_dir)
    report = evaluate_main(cfg, ckpt_path, eval_dir)

    print("[done] full pipeline complete")
    print(f"[done] checkpoint: {ckpt_path}")
    print(f"[done] eval report: {os.path.join(eval_dir, 'eval_report.json')}")
    for split_name in ("staged_unsupervised", "staged_tuned", "realworld_unsupervised", "realworld_tuned"):
        split = report.get(split_name)
        if not split:
            continue
        print(
            f"[metric] {split_name}: "
            f"HPRS={split['hprs']:.4f} "
            f"F1={split['f1']:.4f} "
            f"Precision={split['precision']:.4f} "
            f"Recall={split['recall']:.4f} "
            f"Specificity={split['specificity']:.4f} "
            f"PR-AUC={split['pr_auc']:.4f} "
            f"ROC-AUC={split['roc_auc']:.4f}"
        )
    print(f"[metric] PR/ROC curve data saved in: {os.path.join(eval_dir, 'eval_report.json')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RetailS pose-based BiLSTM anomaly pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one-class BiLSTM autoencoder")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument("--output-dir", default=None)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained checkpoint")
    p_eval.add_argument("--config", default="configs/default.yaml")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--output-dir", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_full = sub.add_parser("full", help="Run train + evaluate end-to-end")
    p_full.add_argument("--config", default="configs/default.yaml")
    p_full.add_argument("--output-dir", default=None)
    p_full.set_defaults(func=cmd_full)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
