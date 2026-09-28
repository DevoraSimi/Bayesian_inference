import argparse
import copy
import csv
import itertools
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from checkpoints import save_metrics_checkpoint, save_pickle_checkpoint, save_rnn_checkpoint
from data_utils import load_split_ids, punctuation_ids
from evaluation import evaluate_hmm, evaluate_ngram, evaluate_rnn, evaluate_rnn_mc_dropout, metrics_to_row, rnn_nll
from hmm import train_hmm
from ngram import train_ngram
from rnn import LSTMLanguageModel, batchify, sentences_to_stream, train_epoch
from vb_hmm import train_vb_hmm

RESULT_FIELDS = [
    "model", "param", "num_layers", "dropout", "alpha", "beta0", "mc_samples", "train_time_s",
    "converged", "n_iter_used", "val_perplexity", "test_perplexity",
    "test_top1_acc", "test_top5_acc", "test_top1_acc_words", "test_top5_acc_words",
]


def _row(model_name, param, train_time, val_metrics, test_metrics, converged="", n_iter_used="",
         num_layers="", dropout="", alpha="", beta0="", mc_samples=""):
    val_row = metrics_to_row(val_metrics)
    test_row = metrics_to_row(test_metrics)
    return {
        "model": model_name,
        "param": param,
        "num_layers": num_layers,
        "dropout": dropout,
        "alpha": alpha,
        "beta0": beta0,
        "mc_samples": mc_samples,
        "train_time_s": round(train_time, 1),
        "converged": converged,
        "n_iter_used": n_iter_used,
        "val_perplexity": val_row["perplexity"],
        "test_perplexity": test_row["perplexity"],
        "test_top1_acc": test_row["top1_acc"],
        "test_top5_acc": test_row["top5_acc"],
        "test_top1_acc_words": test_row["top1_acc_words"],
        "test_top5_acc_words": test_row["top5_acc_words"],
    }


def run_hmm_sweep(train_ids, val_ids, test_ids, vocab_size, states_list, n_iter, seed, punct_ids, checkpoints_dir):
    results = []
    total_train_tokens = sum(len(s) for s in train_ids)
    for n_states in states_list:
        print(f"[HMM] n_states={n_states}")
        start = time.time()
        model = train_hmm(train_ids, n_states, vocab_size, n_iter, seed, val_sequences=val_ids)
        train_time = time.time() - start

        val_metrics = evaluate_hmm(model, val_ids, punct_ids=punct_ids)
        test_metrics = evaluate_hmm(model, test_ids, punct_ids=punct_ids)
        results.append(_row("HMM", n_states, train_time, val_metrics, test_metrics,
                             converged=model.monitor_.converged, n_iter_used=model.monitor_.iter))
        print(results[-1])

        config = {
            "model_type": "HMM", "n_states": n_states, "n_iter": n_iter, "seed": seed,
            "vocab_size": vocab_size, "train_time_s": round(train_time, 1),
            "converged": model.monitor_.converged, "n_iter_used": model.monitor_.iter,
            "total_train_tokens": total_train_tokens,
        }
        save_pickle_checkpoint(
            model, config, {"val": val_metrics, "test": test_metrics},
            Path(checkpoints_dir) / f"hmm_{n_states}",
            history=model.monitor_.history, val_history=model.monitor_.val_history,
            train_nll_history=model.monitor_.train_nll_history,
        )
    return results


def _run_vbhmm(train_ids, val_ids, test_ids, vocab_size, n_states, n_iter, seed, punct_ids, checkpoints_dir,
               alpha0, beta0, pi0, model_name, ckpt_name, row_beta0=""):
    start = time.time()
    model = train_vb_hmm(train_ids, n_states, vocab_size, n_iter, seed, alpha0, beta0, pi0, val_sequences=val_ids)
    train_time = time.time() - start

    val_metrics = evaluate_hmm(model, val_ids, punct_ids=punct_ids)
    test_metrics = evaluate_hmm(model, test_ids, punct_ids=punct_ids)
    row = _row(model_name, n_states, train_time, val_metrics, test_metrics,
               converged=model.monitor_.converged, n_iter_used=model.monitor_.iter, beta0=row_beta0)
    print(row)

    config = {
        "model_type": model_name, "n_states": n_states, "n_iter": n_iter, "seed": seed,
        "vocab_size": vocab_size, "alpha0": alpha0, "beta0": beta0, "pi0": pi0,
        "train_time_s": round(train_time, 1),
        "converged": model.monitor_.converged, "n_iter_used": model.monitor_.iter,
        "total_train_tokens": sum(len(s) for s in train_ids),
    }
    save_pickle_checkpoint(
        model, config, {"val": val_metrics, "test": test_metrics},
        Path(checkpoints_dir) / ckpt_name,
        history=model.monitor_.history, val_history=model.monitor_.val_history,
        train_nll_history=model.monitor_.train_nll_history,
    )
    return row


def run_vbhmm_sweep(train_ids, val_ids, test_ids, vocab_size, states_list, n_iter, seed, punct_ids, checkpoints_dir,
                     alpha0=1.0, beta0=0.1, pi0=1.0):
    results = []
    for n_states in states_list:
        print(f"[VB-HMM] n_states={n_states}")
        results.append(_run_vbhmm(train_ids, val_ids, test_ids, vocab_size, n_states, n_iter, seed, punct_ids,
                                  checkpoints_dir, alpha0, beta0, pi0, "VB-HMM", f"vbhmm_{n_states}"))
    return results


def run_vbhmm_beta0_sweep(train_ids, val_ids, test_ids, vocab_size, n_states, beta0_list, n_iter, seed, punct_ids,
                          checkpoints_dir, alpha0=1.0, pi0=1.0):
    """Varies only the emission prior beta0 at a single n_states, to show the
    prior's effect without a full alpha0 x beta0 x pi0 grid. beta0 matters
    most: it sets how sparse each state's word distribution is."""
    results = []
    for beta0 in beta0_list:
        print(f"[VB-HMM-beta0] n_states={n_states} beta0={beta0}")
        results.append(_run_vbhmm(train_ids, val_ids, test_ids, vocab_size, n_states, n_iter, seed, punct_ids,
                                  checkpoints_dir, alpha0, beta0, pi0, "VB-HMM-beta0",
                                  f"vbhmm_{n_states}_b{beta0}", row_beta0=beta0))
    return results


def run_ngram_sweep(train_ids, val_ids, test_ids, vocab_size, orders_list, alpha_list, punct_ids, checkpoints_dir):
    """Sweeps the grid order x alpha. alpha is the Dirichlet smoothing
    concentration -- how strongly the prior pulls each context's predictions
    toward the next-shorter context's -- as much a parameter worth
    experimenting with as order."""
    results = []
    for order, alpha in itertools.product(orders_list, alpha_list):
        print(f"[Ngram] order={order} alpha={alpha}")
        start = time.time()
        model = train_ngram(train_ids, order, vocab_size, alpha)
        train_time = time.time() - start

        val_metrics = evaluate_ngram(model, val_ids, punct_ids=punct_ids)
        test_metrics = evaluate_ngram(model, test_ids, punct_ids=punct_ids)
        results.append(_row("Ngram", order, train_time, val_metrics, test_metrics, alpha=alpha))
        print(results[-1])

        config = {
            "model_type": "Ngram", "order": order, "alpha": alpha, "vocab_size": vocab_size,
            "train_time_s": round(train_time, 1), "n_contexts": len(model.context_counts),
        }
        save_pickle_checkpoint(
            model, config, {"val": val_metrics, "test": test_metrics},
            Path(checkpoints_dir) / f"ngram_{order}_a{alpha}",
        )
    return results


def run_rnn_sweep(train_data, val_data, test_data, vocab_size, hidden_list, layers_list, dropout_list,
                   embed_size, epochs, bptt, lr, seed, device, punct_ids, checkpoints_dir,
                   mc_dropout_samples_list=(), patience=5):
    """Sweeps the full grid hidden_size x num_layers x dropout. Each config
    trains for up to `epochs` epochs with early stopping: if val perplexity
    hasn't improved for `patience` epochs, training stops and the BEST
    epoch's weights (not the last) are restored before final evaluation and
    saving -- the same protection tune_rnn.py's final retrain uses, applied
    here across the whole grid, since a bigger hidden_size/num_layers config
    is if anything more prone to overfitting than what a hyperparameter
    search might have been tuned at.

    For each trained model, ALSO evaluates it with MC Dropout (dropout kept
    active) once per n_samples value in mc_dropout_samples_list, reported as
    separate "RNN-MCDropout" rows distinguished by their mc_samples column
    -- same underlying weights, no separate checkpoint saved, just repeated
    re-scoring of the same model at different sample counts, to see how
    many stochastic passes it actually takes before perplexity/accuracy
    stop changing (i.e. where the averaging has converged)."""
    results = []
    for hidden_size, num_layers, dropout in itertools.product(hidden_list, layers_list, dropout_list):
        print(f"[RNN] hidden_size={hidden_size} num_layers={num_layers} dropout={dropout}")
        torch.manual_seed(seed)
        model = LSTMLanguageModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_val_ppl = float("inf")
        best_state = None
        best_epoch = 0
        epochs_since_improvement = 0
        start = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_data, optimizer, criterion, bptt)
            epoch_val_metrics = evaluate_rnn(model, val_data, criterion, bptt, punct_ids=punct_ids)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_nll": rnn_nll(model, train_data, criterion, bptt),
                "val_nll": math.log(epoch_val_metrics["perplexity"]),
                "val_perplexity": epoch_val_metrics["perplexity"],
                "val_top5_acc": epoch_val_metrics["topk_acc"][5],
                "val_top5_acc_words": epoch_val_metrics["topk_acc_words"][5],
            })

            if epoch_val_metrics["perplexity"] < best_val_ppl:
                best_val_ppl = epoch_val_metrics["perplexity"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"  no val improvement for {patience} epochs, stopping early at epoch {epoch} "
                          f"(best was epoch {best_epoch}, val_ppl={best_val_ppl:.2f})")
                    break
        train_time = time.time() - start

        model.load_state_dict(best_state)
        val_metrics = evaluate_rnn(model, val_data, criterion, bptt, punct_ids=punct_ids)
        test_metrics = evaluate_rnn(model, test_data, criterion, bptt, punct_ids=punct_ids)
        results.append(_row("RNN", hidden_size, train_time, val_metrics, test_metrics,
                             num_layers=num_layers, dropout=dropout))
        print(results[-1])

        config = {
            "model_type": "RNN", "vocab_size": vocab_size, "embed_size": embed_size,
            "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
            "bptt": bptt, "epochs": epochs, "best_epoch": best_epoch,
            "early_stopped": len(history) < epochs, "patience": patience,
            "lr": lr, "seed": seed, "train_time_s": round(train_time, 1),
        }
        save_rnn_checkpoint(
            model, optimizer, best_epoch, config, {"val": val_metrics, "test": test_metrics},
            Path(checkpoints_dir) / f"rnn_h{hidden_size}_l{num_layers}_d{dropout}", history=history,
        )

        for n_samples in mc_dropout_samples_list:
            print(f"[RNN-MCDropout] hidden_size={hidden_size} num_layers={num_layers} dropout={dropout} "
                  f"n_samples={n_samples}")
            mc_start = time.time()
            mc_val_metrics = evaluate_rnn_mc_dropout(model, val_data, bptt, punct_ids=punct_ids,
                                                       n_samples=n_samples)
            mc_test_metrics = evaluate_rnn_mc_dropout(model, test_data, bptt, punct_ids=punct_ids,
                                                        n_samples=n_samples)
            mc_time = time.time() - mc_start
            results.append(_row("RNN-MCDropout", hidden_size, mc_time, mc_val_metrics, mc_test_metrics,
                                 num_layers=num_layers, dropout=dropout, mc_samples=n_samples))
            print(results[-1])

            mc_config = {
                "model_type": "RNN-MCDropout", "hidden_size": hidden_size, "num_layers": num_layers,
                "dropout": dropout, "n_samples": n_samples, "train_time_s": round(mc_time, 1),
                "source_checkpoint": f"rnn_h{hidden_size}_l{num_layers}_d{dropout}",
            }
            save_metrics_checkpoint(
                mc_config, {"val": mc_val_metrics, "test": mc_test_metrics},
                Path(checkpoints_dir) / f"rnn_h{hidden_size}_l{num_layers}_d{dropout}_mc{n_samples}",
            )
    return results


CSV_KEY_FIELDS = ["model", "param", "num_layers", "dropout", "alpha", "beta0", "mc_samples"]


def _csv_row_key(row):
    return tuple(str(row.get(k, "")) for k in CSV_KEY_FIELDS)


def save_csv(results, path):
    """Merges `results` into any existing CSV at `path` (keyed by model +
    param + num_layers + dropout + alpha + mc_samples), instead of blindly
    overwriting it -- so a partial run (e.g. via --skip-rnn) only updates
    the rows it actually produced, without discarding rows a PREVIOUS run
    computed for model families this run skipped."""
    path = Path(path)
    merged = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                merged[_csv_row_key(row)] = row
    for r in results:
        merged[_csv_row_key(r)] = {k: r.get(k, "") for k in RESULT_FIELDS}

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(merged.values())


def _plot_vs_param(ax, rows, xlabel, title):
    rows = sorted(rows, key=lambda r: r["param"])
    ax.plot([r["param"] for r in rows], [r["test_perplexity"] for r in rows], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test perplexity")
    ax.set_title(title)


def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)

    hmm_rows = [r for r in results if r["model"] == "HMM"]
    if hmm_rows:
        plt.figure()
        _plot_vs_param(plt.gca(), hmm_rows, "n_states", "HMM: perplexity vs. states")
        plt.savefig(out_dir / "hmm_perplexity.png")
        plt.close()

    vbhmm_rows = [r for r in results if r["model"] == "VB-HMM"]
    if vbhmm_rows:
        plt.figure()
        _plot_vs_param(plt.gca(), vbhmm_rows, "n_states", "VB-HMM: perplexity vs. states")
        plt.savefig(out_dir / "vbhmm_perplexity.png")
        plt.close()

    beta0_rows = sorted([r for r in results if r["model"] == "VB-HMM-beta0"], key=lambda r: r["beta0"])
    if beta0_rows:
        plt.figure()
        plt.plot([r["beta0"] for r in beta0_rows], [r["test_perplexity"] for r in beta0_rows], marker="o")
        plt.xscale("log")
        plt.xlabel("emission prior beta0")
        plt.ylabel("test perplexity")
        plt.title(f"VB-HMM: effect of the prior (n_states={beta0_rows[0]['param']})")
        plt.savefig(out_dir / "vbhmm_beta0.png")
        plt.close()

    ngram_rows = [r for r in results if r["model"] == "Ngram"]
    if ngram_rows:
        plt.figure()
        for alpha in sorted(set(r["alpha"] for r in ngram_rows)):
            subset = sorted([r for r in ngram_rows if r["alpha"] == alpha], key=lambda r: r["param"])
            plt.plot([r["param"] for r in subset], [r["test_perplexity"] for r in subset],
                     marker="o", label=f"alpha={alpha}")
        plt.xlabel("order")
        plt.ylabel("test perplexity")
        plt.title("N-gram: perplexity vs. order")
        plt.legend()
        plt.savefig(out_dir / "ngram_perplexity.png")
        plt.close()

    rnn_rows = [r for r in results if r["model"] == "RNN"]
    if rnn_rows:
        plt.figure()
        groups = sorted(set((r["num_layers"], r["dropout"]) for r in rnn_rows))
        for layers, dropout in groups:
            subset = sorted(
                [r for r in rnn_rows if r["num_layers"] == layers and r["dropout"] == dropout],
                key=lambda r: r["param"],
            )
            xs = [r["param"] for r in subset]
            ys = [r["test_perplexity"] for r in subset]
            plt.plot(xs, ys, marker="o", label=f"layers={layers}, dropout={dropout}")
        plt.xlabel("hidden_size")
        plt.ylabel("test perplexity")
        plt.title("RNN: perplexity vs. hidden size")
        plt.legend()
        plt.savefig(out_dir / "rnn_perplexity.png")
        plt.close()

    mc_rows = [r for r in results if r["model"] == "RNN-MCDropout"]
    if mc_rows:
        plt.figure()
        configs = sorted(set((r["param"], r["num_layers"], r["dropout"]) for r in mc_rows))
        for hidden_size, layers, dropout in configs:
            subset = sorted(
                [r for r in mc_rows if r["param"] == hidden_size and r["num_layers"] == layers and r["dropout"] == dropout],
                key=lambda r: r["mc_samples"],
            )
            xs = [r["mc_samples"] for r in subset]
            ys = [r["test_perplexity"] for r in subset]
            plt.plot(xs, ys, marker="o", label=f"hidden={hidden_size}, layers={layers}, dropout={dropout}")
        plt.xlabel("MC-Dropout n_samples")
        plt.ylabel("test perplexity")
        plt.title("RNN-MCDropout: does more sampling change the result?")
        plt.legend()
        plt.savefig(out_dir / "mc_dropout_convergence.png")
        plt.close()

    # headline comparison: per model type, the config with the lowest VAL
    # perplexity (selecting on test would bias the reported test numbers
    # optimistically), then report that config's test metrics
    by_model = {}
    for r in results:
        if r["model"] not in by_model or r["val_perplexity"] < by_model[r["model"]]["val_perplexity"]:
            by_model[r["model"]] = r
    if by_model:
        order = [m for m in ("Ngram", "HMM", "VB-HMM", "RNN") if m in by_model]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        ppl = [by_model[m]["test_perplexity"] for m in order]
        axes[0].bar(order, ppl, color="C0")
        for i, m in enumerate(order):
            axes[0].text(i, ppl[i], f"param={by_model[m]['param']}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("test perplexity (config chosen on val)")
        axes[0].set_title("Best config per model")

        acc = [by_model[m]["test_top5_acc_words"] * 100 for m in order]
        axes[1].bar(order, acc, color="C1")
        axes[1].set_ylabel("test top-5 acc, words only (%)")
        axes[1].set_title("Best config's accuracy per model")

        plt.tight_layout()
        plt.savefig(out_dir / "summary_comparison.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--hmm-states", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--hmm-iter", type=int, default=30)
    parser.add_argument("--vbhmm-states", type=int, nargs="+", default=None,
                         help="defaults to --hmm-states, for a direct MAP-EM vs VB-EM comparison at matching n_states")
    parser.add_argument("--vbhmm-iter", type=int, default=None, help="defaults to --hmm-iter")
    parser.add_argument("--vbhmm-alpha0", type=float, default=1.0)
    parser.add_argument("--vbhmm-beta0", type=float, default=0.1)
    parser.add_argument("--vbhmm-pi0", type=float, default=1.0)
    parser.add_argument("--vbhmm-beta0-sweep", type=float, nargs="+", default=[],
                         help="if given, also train VB-HMM once per beta0 value here, at a single n_states, to show "
                              "the prior's effect (off by default). e.g. --vbhmm-beta0-sweep 0.01 0.1 1")
    parser.add_argument("--vbhmm-beta0-states", type=int, default=None,
                         help="n_states for --vbhmm-beta0-sweep; defaults to the VB-HMM n_states with the lowest "
                              "val perplexity in this run")
    parser.add_argument("--ngram-orders", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ngram-alpha", type=float, nargs="+", default=[10.0, 30.0, 100.0])
    parser.add_argument("--rnn-hidden", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--rnn-layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--rnn-dropout", type=float, nargs="+", default=[0.2])
    parser.add_argument("--rnn-embed-size", type=int, default=128)
    parser.add_argument("--rnn-epochs", type=int, default=8, help="max epochs; early stopping may stop sooner")
    parser.add_argument("--rnn-patience", type=int, default=5,
                         help="stop an RNN config's training early if val perplexity hasn't improved for this "
                              "many epochs; the saved checkpoint uses the best epoch's weights, not the last")
    parser.add_argument("--mc-dropout-samples", type=int, nargs="+", default=[],
                         help="if given, also evaluate each RNN config with MC Dropout once per n_samples value "
                              "here, reported as separate RNN-MCDropout rows plus a metrics-only checkpoint each "
                              "(re-scores the RNN's existing weights, so no new .pt is saved, just a .json) "
                              "-- multiplies RNN evaluation cost accordingly, so empty (off) by default. "
                              "e.g. --mc-dropout-samples 5 20 50 to see how many samples it takes to converge")
    parser.add_argument("--bptt", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ngram", action="store_true", help="skip the n-gram sweep entirely")
    parser.add_argument("--skip-hmm", action="store_true", help="skip the HMM sweep entirely")
    parser.add_argument("--skip-vbhmm", action="store_true", help="skip the VB-HMM sweep entirely")
    parser.add_argument("--skip-rnn", action="store_true", help="skip the RNN sweep entirely (also skips MC-Dropout)")
    args = parser.parse_args()

    vbhmm_states = args.vbhmm_states or args.hmm_states
    vbhmm_iter = args.vbhmm_iter or args.hmm_iter

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    word2id, id2word, train_ids, val_ids, test_ids = load_split_ids(args.data_dir)
    vocab_size = len(word2id)
    punct_ids = punctuation_ids(word2id)

    ngram_results = []
    if not args.skip_ngram:
        ngram_results = run_ngram_sweep(
            train_ids, val_ids, test_ids, vocab_size, args.ngram_orders, args.ngram_alpha,
            punct_ids, args.checkpoints_dir,
        )

    hmm_results = []
    if not args.skip_hmm:
        hmm_results = run_hmm_sweep(
            train_ids, val_ids, test_ids, vocab_size, args.hmm_states, args.hmm_iter, args.seed,
            punct_ids, args.checkpoints_dir,
        )

    vbhmm_results = []
    if not args.skip_vbhmm:
        vbhmm_results = run_vbhmm_sweep(
            train_ids, val_ids, test_ids, vocab_size, vbhmm_states, vbhmm_iter, args.seed,
            punct_ids, args.checkpoints_dir, args.vbhmm_alpha0, args.vbhmm_beta0, args.vbhmm_pi0,
        )

    if args.vbhmm_beta0_sweep:
        beta0_states = args.vbhmm_beta0_states
        if beta0_states is None:
            if not vbhmm_results:
                parser.error("--vbhmm-beta0-sweep needs --vbhmm-beta0-states when the VB-HMM sweep is skipped")
            beta0_states = min(vbhmm_results, key=lambda r: r["val_perplexity"])["param"]
        vbhmm_results += run_vbhmm_beta0_sweep(
            train_ids, val_ids, test_ids, vocab_size, beta0_states, args.vbhmm_beta0_sweep, vbhmm_iter, args.seed,
            punct_ids, args.checkpoints_dir, args.vbhmm_alpha0, args.vbhmm_pi0,
        )

    rnn_results = []
    if not args.skip_rnn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data = batchify(sentences_to_stream(train_ids), args.batch_size, device)
        val_data = batchify(sentences_to_stream(val_ids), args.batch_size, device)
        test_data = batchify(sentences_to_stream(test_ids), args.batch_size, device)

        rnn_results = run_rnn_sweep(
            train_data, val_data, test_data, vocab_size, args.rnn_hidden, args.rnn_layers, args.rnn_dropout,
            args.rnn_embed_size, args.rnn_epochs, args.bptt, args.lr, args.seed, device, punct_ids,
            args.checkpoints_dir, args.mc_dropout_samples, args.rnn_patience,
        )

    all_results = ngram_results + hmm_results + vbhmm_results + rnn_results
    if not all_results:
        print("nothing ran (everything was --skip-*'d) -- nothing to save/plot")
        return
    save_csv(all_results, results_dir / "comparison.csv")
    plot_results(all_results, results_dir)
    if args.skip_ngram or args.skip_hmm or args.skip_vbhmm or args.skip_rnn:
        print("\nnote: comparison.csv was MERGED, not overwritten -- rows from previous runs for the "
              "model(s) you skipped this time are still there. The PLOTS, though, are only rebuilt from "
              "what actually ran this run: per-model plots for a skipped family are left as whatever they "
              "were before (not deleted, just not refreshed), and summary_comparison.png's bars will only "
              "cover what ran this time, not the full merged CSV. Use list_checkpoints.py for a complete, "
              "always-fresh tabular view across every checkpoint regardless of which run produced it.")

    print("\n=== summary ===")
    for r in all_results:
        print(r)
    print(f"\nsaved results to {results_dir}/comparison.csv and plots")
    print(f"saved per-config checkpoints to {args.checkpoints_dir}/")


if __name__ == "__main__":
    main()
