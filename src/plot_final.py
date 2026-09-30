"""Final report figures, all on the VALIDATION split, read from
all_checkpoints_in_report/*.json (every configuration; checkpoints/ holds only
the final model of each family).

  A  loss curves      -- train (dashed) vs val (solid) NLL/token per iteration/epoch
                         for the iteratively trained models (HMM, VB-HMM, LSTM).
  B  val perplexity   -- final val perplexity vs each family's main hyperparameter:
                         (a) n-gram, (b) HMM & VB-HMM, (c) LSTM.
  C  summary          -- best val perplexity of each model family, side by side.

A and B are not duplicates: perplexity = exp(NLL/token), so the last val point of
a curve in A is log() of one point in B -- but A shows the training dynamics
(convergence, over-fitting, early stopping) while B compares the finished models
across hyperparameters, and B also covers the n-gram, which has no curve.

Usage (from the repo root):
    python src/plot_final.py [--checkpoints-dir all_checkpoints_in_report] [--out-dir results]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, ScalarFormatter  # noqa: E402

# categorical slots (fixed order) and an ordinal blue ramp for ordered sizes
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "lines.linewidth": 1.8,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- loading

def load_all(checkpoints_dir):
    """Returns every checkpoint's metadata as a list of dicts (with 'name')."""
    runs = []
    for path in sorted(Path(checkpoints_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        data["name"] = path.stem
        runs.append(data)
    return runs


def of_type(runs, model_type):
    return [r for r in runs if r.get("config", {}).get("model_type") == model_type]


def shared_alpha_ngrams(runs):
    """n-gram runs with one alpha shared by every level (per-level runs store a list)."""
    return [r for r in of_type(runs, "Ngram") if not isinstance(r["config"]["alpha"], list)]


def val_ppl(run):
    return run["metrics"]["val"]["perplexity"]


def curves(run):
    """(train_nll, val_nll) per iteration/epoch, both in nats/token."""
    if run["config"]["model_type"] == "RNN":
        h = run.get("history", [])
        return [e["train_nll"] for e in h], [e["val_nll"] for e in h]
    return run.get("train_nll_history", []), run.get("val_history", [])


def lstm_grid(runs):
    """The LSTM grid runs (rnn_h*_l*), excluding the Optuna re-train."""
    return [r for r in of_type(runs, "RNN") if r["config"].get("selected_by") != "optuna"]


# ---------------------------------------------------------------- axis helpers

def plain_log_x(ax, ticks, base=10):
    """Log-scaled x with the actual tick values printed (4, 8, 16 ... / 10, 30, 100),
    never 10^2-style labels, and no minor tick clutter."""
    ax.set_xscale("log", base=base)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    lo, hi = min(ticks), max(ticks)
    ax.set_xlim(lo / 1.25, hi * 1.8)  # extra right margin for the end-value labels


def plain_y(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)


def end_labels(ax, points, min_gap_pt=16):
    """Direct value labels right of each line's last point (ink, not series colour),
    nudged apart vertically so close endpoints don't overprint.
    points: list of (x, y, text); call after tight_layout so the axes size is final."""
    to_px = ax.transData.transform
    pt_per_px = 72 / ax.figure.dpi
    placed = []
    for x, y, text in sorted(points, key=lambda p: p[1]):
        y_pt = to_px((x, y))[1] * pt_per_px
        ty_pt = max(y_pt, placed[-1] + min_gap_pt) if placed else y_pt
        placed.append(ty_pt)
        ax.annotate(text, (x, y), xytext=(8, ty_pt - y_pt), textcoords="offset points",
                    va="center", fontsize=13, color=INK2, annotation_clip=False)


def save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"saved {out_dir / stem}.png/.pdf")


# ---------------------------------------------------------------- plot A

def _curve_panel(ax, items, title, xlabel, best_epoch_marker=False):
    """items: list of (label, colour, run). Dashed = train, solid = val."""
    for label, color, run in items:
        tr, va = curves(run)
        x_tr, x_va = range(1, len(tr) + 1), range(1, len(va) + 1)
        ax.plot(x_tr, tr, color=color, linestyle="--", linewidth=1.4)
        ax.plot(x_va, va, color=color, linestyle="-", label=label)
        if best_epoch_marker and run["config"].get("best_epoch"):
            b = run["config"]["best_epoch"]
            ax.plot([b], [va[b - 1]], "o", color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plain_y(ax)


def plot_a(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    style_handles = [Line2D([], [], color=INK2, linestyle="--", linewidth=1.4, label="train"),
                     Line2D([], [], color=INK2, linestyle="-", label="validation")]

    # row 1: HMM / VB-HMM, colour = number of hidden states (ordinal ramp), shared y
    for ax, mtype, title in [(axes[0, 0], "HMM", "(a) HMM (Baum-Welch EM)"),
                             (axes[0, 1], "VB-HMM", "(b) VB-HMM (variational Bayes EM)")]:
        rs = sorted(of_type(runs, mtype), key=lambda r: r["config"]["n_states"])
        items = [(f"K={r['config']['n_states']}", RAMP[i % len(RAMP)], r) for i, r in enumerate(rs)]
        _curve_panel(ax, items, title, "EM iteration")
    axes[0, 1].sharey(axes[0, 0])

    # row 2: LSTM split by depth, colour = hidden size, shared y
    grid = lstm_grid(runs)
    sizes = sorted({r["config"]["hidden_size"] for r in grid})
    size_color = {h: RAMP[[0, 2, 4][i]] if len(sizes) == 3 else RAMP[i % len(RAMP)]
                  for i, h in enumerate(sizes)}
    for ax, layers, title in [(axes[1, 0], 1, "(c) LSTM, 1 layer"),
                              (axes[1, 1], 2, "(d) LSTM, 2 layers")]:
        rs = sorted([r for r in grid if r["config"]["num_layers"] == layers],
                    key=lambda r: r["config"]["hidden_size"])
        items = [(f"hidden={r['config']['hidden_size']}", size_color[r["config"]["hidden_size"]], r)
                 for r in rs]
        _curve_panel(ax, items, title, "epoch", best_epoch_marker=True)
    axes[1, 1].sharey(axes[1, 0])

    # rows share y, so label only the left panel of each row
    for ax in axes[:, 0]:
        ax.set_ylabel("NLL per token (nats)")
    # one legend per row, outside the plot area to the right (both panels share entries)
    best_handle = Line2D([], [], color=INK2, marker="o", linestyle="none", markersize=8,
                         markeredgecolor="white", label="best epoch\n(restored)")
    for row, extra in [(0, []), (1, [best_handle])]:
        h, _ = axes[row, 1].get_legend_handles_labels()
        axes[row, 1].legend(handles=h + style_handles + extra, loc="upper left",
                            bbox_to_anchor=(1.02, 1), handlelength=2.5)

    fig.suptitle("Training vs validation loss (lower is better)",
                 fontsize=18, color=INK)
    fig.tight_layout()
    save(fig, out_dir, "A_loss_curves")


# ---------------------------------------------------------------- plot B

def _line(ax, xs, ys, color, marker, label, linestyle="-"):
    """Plots one series and returns its endpoint for end_labels()."""
    ax.plot(xs, ys, color=color, marker=marker, markersize=7, linestyle=linestyle,
            markeredgecolor="white", markeredgewidth=1.2, label=label)
    return xs[-1], ys[-1], f"{ys[-1]:.1f}"


def plot_b(runs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))

    # (a) n-gram: x = smoothing alpha, one line per order
    ends = {ax: [] for ax in axes}
    ax = axes[0]
    ngrams = shared_alpha_ngrams(runs)
    alphas = sorted({r["config"]["alpha"] for r in ngrams})
    for order, color, marker in [(2, BLUE, "o"), (3, ORANGE, "s"), (4, AQUA, "^")]:
        rs = sorted([r for r in ngrams if r["config"]["order"] == order], key=lambda r: r["config"]["alpha"])
        if rs:
            ends[ax].append(_line(ax, [r["config"]["alpha"] for r in rs], [val_ppl(r) for r in rs],
                                  color, marker, f"{order}-gram"))
    plain_log_x(ax, alphas)
    ax.set_xlabel("smoothing α (log scale)")
    ax.set_title("(a) n-gram", loc="left")

    # (b) HMM vs VB-HMM: x = number of hidden states
    ax = axes[1]
    states = set()
    for mtype, color, marker in [("HMM", BLUE, "o"), ("VB-HMM", ORANGE, "s")]:
        rs = sorted(of_type(runs, mtype), key=lambda r: r["config"]["n_states"])
        xs = [r["config"]["n_states"] for r in rs]
        states.update(xs)
        ends[ax].append(_line(ax, xs, [val_ppl(r) for r in rs], color, marker, mtype))
    plain_log_x(ax, sorted(states), base=2)
    ax.set_xlabel("hidden states K (log scale)")
    ax.set_title("(b) HMM (EM) vs VB-HMM", loc="left")

    # (c) LSTM: x = hidden size, colour = depth
    ax = axes[2]
    grid = lstm_grid(runs)
    sizes = sorted({r["config"]["hidden_size"] for r in grid})
    for layers, color, marker in [(1, BLUE, "o"), (2, ORANGE, "s")]:
        rs = sorted([r for r in grid if r["config"]["num_layers"] == layers],
                    key=lambda r: r["config"]["hidden_size"])
        if rs:
            ends[ax].append(_line(ax, [r["config"]["hidden_size"] for r in rs], [val_ppl(r) for r in rs],
                                  color, marker, f"{layers} layer{'s' * (layers > 1)}"))
    plain_log_x(ax, sizes, base=2)
    ax.set_xlabel("LSTM hidden size (log scale)")
    ax.set_title("(c) LSTM", loc="left")

    axes[0].set_ylabel("validation perplexity")
    for ax in axes:
        plain_y(ax)
        # below the panel, so it never covers data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=3)
    fig.suptitle("Validation perplexity by model and hyperparameter (lower is better; "
                 "y-axes differ per panel)",
                 fontsize=16, color=INK)
    fig.tight_layout()
    for ax in axes:
        end_labels(ax, ends[ax])
    save(fig, out_dir, "B_val_perplexity")


# ---------------------------------------------------------------- plot C

def _best(rs):
    return min(rs, key=val_ppl) if rs else None


def plot_c(runs, out_dir):
    """Best val perplexity per model family on one common, zero-based axis."""
    ngrams = shared_alpha_ngrams(runs)
    rows = []
    for order in sorted({r["config"]["order"] for r in ngrams}):
        b = _best([r for r in ngrams if r["config"]["order"] == order])
        rows.append((f"{order}-gram (α={b['config']['alpha']:g})", val_ppl(b)))
    for mtype in ("HMM", "VB-HMM"):
        b = _best(of_type(runs, mtype))
        if b:
            rows.append((f"{mtype} (K={b['config']['n_states']})", val_ppl(b)))
    b = _best(lstm_grid(runs))
    if b:
        c = b["config"]
        rows.append((f"LSTM (h={c['hidden_size']}, {c['num_layers']}L)", val_ppl(b)))

    rows.sort(key=lambda r: r[1], reverse=True)  # best at the top
    labels, vals = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(rows) + 1.6))
    bars = ax.barh(labels, vals, color=BLUE, height=0.6, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=15, color=INK)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("validation perplexity (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Best configuration of each model family", loc="left", fontsize=18)
    fig.tight_layout()
    save(fig, out_dir, "C_best_per_family")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="all_checkpoints_in_report")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    runs = load_all(args.checkpoints_dir)
    out_dir = Path(args.out_dir)
    plot_a(runs, out_dir)
    plot_b(runs, out_dir)
    plot_c(runs, out_dir)


if __name__ == "__main__":
    main()
