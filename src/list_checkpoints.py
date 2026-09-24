import argparse
import csv
from pathlib import Path

from checkpoints import load_metadata
from evaluation import metrics_to_row

FIELDNAMES = [
    "checkpoint", "model", "param", "val_perplexity", "test_perplexity",
    "test_top1_acc", "test_top5_acc", "test_top1_acc_words", "test_top5_acc_words",
    "train_time_s", "converged", "timestamp",
]


def load_checkpoint_rows(checkpoints_dir):
    rows = []
    for path in sorted(Path(checkpoints_dir).glob("*.json")):
        # path.with_suffix("") here only strips the trailing ".json" the glob
        # itself guaranteed -- safe even though the stem may contain other
        # dots (e.g. a dropout value like "rnn_h4_l1_d0.1").
        data = load_metadata(path.with_suffix(""))
        cfg = data.get("config", {})
        metrics = data.get("metrics", {})
        val_metrics = metrics.get("val")
        test_metrics = metrics.get("test")

        row = {
            "checkpoint": path.stem,
            "model": cfg.get("model_type", "?"),
            "param": cfg.get("n_states", cfg.get("hidden_size", cfg.get("order", "?"))),
            "train_time_s": cfg.get("train_time_s", ""),
            "timestamp": data.get("timestamp", ""),
        }
        if cfg.get("model_type") in ("HMM", "VB-HMM"):
            row["converged"] = cfg.get("converged", "")

        if val_metrics:
            row["val_perplexity"] = round(val_metrics["perplexity"], 2)
        if test_metrics:
            row.update({f"test_{k}": v for k, v in metrics_to_row(test_metrics).items()})
        else:
            row["test_perplexity"] = "pending"

        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["param"] if isinstance(r["param"], int) else 0))
    return rows


def print_table(rows):
    if not rows:
        print("no checkpoints found")
        return

    present = [f for f in FIELDNAMES if any(f in r for r in rows)]
    widths = {f: max(len(f), max((len(str(r.get(f, ""))) for r in rows), default=0)) for f in present}

    print("  ".join(f.ljust(widths[f]) for f in present))
    print("  ".join("-" * widths[f] for f in present))
    for r in rows:
        print("  ".join(str(r.get(f, "")).ljust(widths[f]) for f in present))


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out", default=None, help="optional path to also save the table as CSV")
    args = parser.parse_args()

    rows = load_checkpoint_rows(args.checkpoints_dir)
    print_table(rows)

    if args.out:
        save_csv(rows, args.out)
        print(f"\nsaved manifest to {args.out}")


if __name__ == "__main__":
    main()
