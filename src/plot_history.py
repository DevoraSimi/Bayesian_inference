import argparse
from pathlib import Path

from checkpoints import load_metadata

# model_type -> (config key that distinguishes runs, x-axis label, panel title)
MODEL_KEYS = {
    "HMM": ("n_states", "EM iteration", "HMM (Baum-Welch)"),
    "VB-HMM": ("n_states", "VB-EM iteration", "VB-HMM (Variational Bayes)"),
    "RNN": (("hidden_size", "num_layers", "dropout"), "epoch", "RNN (LSTM)"),
}


def load_runs(checkpoints_dir):
    """Returns (runs, skipped_types): runs maps model_type -> list of
    (config, train_nll, val_nll) for the iteratively-trained model types, where
    train_nll[k] and val_nll[k] are both average NLL/token (nats), both scored
    with the same function in eval mode on the parameters as they stood at the
    end of iteration/epoch k+1 -- so train and val are directly comparable, and
    so are all three model types. skipped_types names every OTHER model_type
    found in checkpoints_dir (e.g. Ngram, fit in one closed-form pass with no
    iteration to plot; or RNN-MCDropout, which has no checkpoint/history of
    its own since it re-scores an already-trained RNN rather than training
    anything new), plus old checkpoints saved before train NLL was recorded."""
    runs = {model_type: [] for model_type in MODEL_KEYS}
    skipped_types = set()
    for path in sorted(Path(checkpoints_dir).glob("*.json")):
        data = load_metadata(path.with_suffix(""))
        cfg = data.get("config", {})
        model_type = cfg.get("model_type")
        if model_type == "RNN":
            history = data.get("history", [])
            train_nll = [h["train_nll"] for h in history if "train_nll" in h]
            val_nll = [h["val_nll"] for h in history if "val_nll" in h]
        elif model_type in runs:
            train_nll = data.get("train_nll_history", [])
            val_nll = data.get("val_history", [])
        else:
            skipped_types.add(model_type)
            continue
        if train_nll and val_nll:
            runs[model_type].append((cfg, train_nll, val_nll))
        else:
            skipped_types.add(f"{model_type} (saved before train NLL was recorded -- retrain)")
    return runs, skipped_types


def _run_label(cfg, param_key):
    keys = param_key if isinstance(param_key, tuple) else (param_key,)
    return ", ".join(f"{k}={cfg.get(k)}" for k in keys)


def plot_train_val(ax, runs, model_type):
    """One colour per run: dashed = train, solid = val, same x = same parameters."""
    param_key, xlabel, title = MODEL_KEYS[model_type]
    keys = param_key if isinstance(param_key, tuple) else (param_key,)
    if runs:
        for cfg, train_nll, val_nll in sorted(runs, key=lambda r: tuple(r[0].get(k, 0) for k in keys)):
            label = _run_label(cfg, param_key)
            line, = ax.plot(range(1, len(train_nll) + 1), train_nll, linestyle="--", label=f"{label} train")
            ax.plot(range(1, len(val_nll) + 1), val_nll, color=line.get_color(), marker="o", markersize=3,
                    label=f"{label} val")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, f"no {model_type} train/val history found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NLL / token (nats)")
    ax.set_title(f"{title}\ndashed = train, solid = val")


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
        print(f"no checkpoints with saved train/val history found in {args.checkpoints_dir}/")
        return

    # shared y-axis: all panels use the same metric, so levels compare across models too
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    for ax, model_type in zip(axes, MODEL_KEYS):
        plot_train_val(ax, runs[model_type], model_type)

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
        reason = SKIP_REASONS.get(t, "it has no per-iteration train/val history to plot.")
        print(f"\nnote: {t} checkpoint(s) found but not plotted here -- {reason}")


if __name__ == "__main__":
    main()
