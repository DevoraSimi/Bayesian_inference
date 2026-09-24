import argparse
import math
from pathlib import Path

from checkpoints import load_metadata

MODEL_KEYS = {
    "HMM": ("n_states", "EM iteration", "HMM (Baum-Welch)"),
    "VB-HMM": ("n_states", "VB-EM iteration", "VB-HMM (Variational Bayes)"),
}


def load_runs(checkpoints_dir):
    """Returns (runs, skipped_types): runs holds (config, history, val_history)
    tuples for the iteratively-trained model types this tool can plot a loss
    curve for; skipped_types names every OTHER model_type found in
    checkpoints_dir (e.g. Ngram, fit in one closed-form pass with no
    iteration to plot; or RNN-MCDropout, which has no checkpoint/history of
    its own since it re-scores an already-trained RNN rather than training
    anything new)."""
    runs = {"HMM": [], "VB-HMM": [], "RNN": []}
    skipped_types = set()
    for path in sorted(Path(checkpoints_dir).glob("*.json")):
        data = load_metadata(path.with_suffix(""))
        cfg = data.get("config", {})
        history = data.get("history", [])
        val_history = data.get("val_history", [])
        model_type = cfg.get("model_type")
        if model_type in runs and history:
            runs[model_type].append((cfg, history, val_history))
        else:
            skipped_types.add(model_type)
    return runs, skipped_types


def plot_hmm_train(ax, runs, model_type):
    param_key, xlabel, title = MODEL_KEYS[model_type]
    if runs:
        for cfg, history, _ in sorted(runs, key=lambda r: r[0].get(param_key, 0)):
            n_tokens = cfg.get("total_train_tokens")
            ys = [-ll / n_tokens for ll in history] if n_tokens else [-ll for ll in history]
            ax.plot(range(1, len(history) + 1), ys, marker="o", label=f"{param_key}={cfg.get(param_key)}")
        ax.set_ylabel("train NLL / token (nats)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"no {model_type} history found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}: training loss")


def plot_hmm_val(ax, runs, model_type):
    param_key, xlabel, title = MODEL_KEYS[model_type]
    runs_with_val = [(cfg, h, vh) for cfg, h, vh in runs if vh]
    if runs_with_val:
        for cfg, history, val_history in sorted(runs_with_val, key=lambda r: r[0].get(param_key, 0)):
            ax.plot(range(1, len(val_history) + 1), val_history, marker="o", label=f"{param_key}={cfg.get(param_key)}")
        ax.set_ylabel("val NLL / token (nats)")
        ax.legend(fontsize=8)
    else:
        msg = f"no {model_type} val history found" + ("\n(retrain to capture it)" if runs else "")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}: validation loss")


def plot_rnn_train(ax, runs):
    if runs:
        for cfg, history, _ in sorted(runs, key=lambda r: (r[0].get("hidden_size", 0), r[0].get("num_layers", 0))):
            xs = [h["epoch"] for h in history]
            ys = [h["train_loss"] for h in history]
            label = f"hidden={cfg.get('hidden_size')}, layers={cfg.get('num_layers')}"
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_ylabel("train loss (cross-entropy, nats)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no RNN history found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("epoch")
    ax.set_title("RNN: training loss")


def plot_rnn_val(ax, runs):
    if runs:
        for cfg, history, _ in sorted(runs, key=lambda r: (r[0].get("hidden_size", 0), r[0].get("num_layers", 0))):
            xs = [h["epoch"] for h in history]
            ys = [math.log(h["val_perplexity"]) for h in history]  # loss = log(perplexity), same nats scale as train_loss
            label = f"hidden={cfg.get('hidden_size')}, layers={cfg.get('num_layers')}"
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_ylabel("val loss (cross-entropy, nats)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no RNN history found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("epoch")
    ax.set_title("RNN: validation loss (watch for overfitting)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out", default="results/loss_curves.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, skipped_types = load_runs(args.checkpoints_dir)
    if not any(runs.values()):
        print(f"no checkpoints with saved history found in {args.checkpoints_dir}/")
        return

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    plot_hmm_train(axes[0, 0], runs["HMM"], "HMM")
    plot_hmm_val(axes[0, 1], runs["HMM"], "HMM")
    plot_hmm_train(axes[1, 0], runs["VB-HMM"], "VB-HMM")
    plot_hmm_val(axes[1, 1], runs["VB-HMM"], "VB-HMM")
    plot_rnn_train(axes[2, 0], runs["RNN"])
    plot_rnn_val(axes[2, 1], runs["RNN"])

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"saved plot to {out_path}")

    SKIP_REASONS = {
        "Ngram": "it's fit in one closed-form counting pass, not iteratively, so there's no per-iteration "
                 "loss to plot. See results/ngram_perplexity.png (from experiment.py) instead.",
        "RNN-MCDropout": "it re-scores an already-trained RNN's existing weights rather than training "
                          "anything new, so it has a metrics-only checkpoint but no training history to plot. "
                          "See results/mc_dropout_convergence.png (from experiment.py) instead.",
    }
    for t in sorted(name for name in skipped_types if name):
        reason = SKIP_REASONS.get(t, "it has no per-iteration training history to plot.")
        print(f"\nnote: {t} checkpoint(s) found but not plotted here -- {reason}")


if __name__ == "__main__":
    main()
