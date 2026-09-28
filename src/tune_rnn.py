import argparse
import copy
import csv
import math
import time
from pathlib import Path

import optuna
import torch
import torch.nn as nn

from checkpoints import save_rnn_checkpoint
from data_utils import EOS, load_split_ids, punctuation_ids
from evaluation import evaluate_rnn, print_eval, rnn_nll
from rnn import LSTMLanguageModel, paragraph_batches, train_epoch


def make_objective(train_ids, val_batches, vocab_size, eos_id, hidden_size, num_layers, punct_ids, batch_size,
                    search_epochs, patience, device, seed):
    """hidden_size/num_layers are FIXED, not searched: if Optuna searched them
    jointly with embed_size/dropout/lr, the winning embed_size/dropout/lr
    would only be validated for whatever specific hidden_size/num_layers
    combo they happened to be paired with in that trial -- not a safe thing
    to then reuse while varying hidden_size/num_layers in a separate sweep.
    Fixing them here to one representative architecture keeps the searched
    embed_size/dropout/lr meaningfully reusable across that later sweep.

    Each trial uses the same early stopping as the final retrain and is scored
    by its BEST epoch's val perplexity, not its last: otherwise a long
    search_epochs scores every trial well past its overfitting point, which
    rewards configs that overfit slowly rather than ones that fit best."""
    def objective(trial):
        embed_size = trial.suggest_categorical("embed_size", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        torch.manual_seed(seed)
        model = LSTMLanguageModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_ppl = float("inf")
        best_epoch = 0
        for epoch in range(1, search_epochs + 1):
            train_batches = paragraph_batches(train_ids, batch_size, eos_id, device, shuffle=True, seed=seed + epoch)
            train_epoch(model, train_batches, optimizer, criterion)
            val_ppl = evaluate_rnn(model, val_batches, criterion, punct_ids=punct_ids)["perplexity"]
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_epoch = epoch
            elif epoch - best_epoch >= patience:
                break

        trial.set_user_attr("best_epoch", best_epoch)
        return best_val_ppl

    return objective


TRIAL_FIELDS = ["trial", "val_perplexity", "best_epoch", "embed_size", "dropout", "lr"]


def make_trial_logger(path, n_trials):
    """Returns an Optuna callback that appends each trial's result to `path`
    as soon as that trial finishes, instead of only writing everything once
    the whole study.optimize() call returns -- so a run that gets killed or
    crashes partway through doesn't lose every trial completed so far."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=TRIAL_FIELDS).writeheader()

    def callback(study, trial):
        row = {"trial": trial.number, "val_perplexity": trial.value,
               "best_epoch": trial.user_attrs.get("best_epoch")}
        row.update(trial.params)
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=TRIAL_FIELDS).writerow(row)
        print(f"  [trial {trial.number + 1}/{n_trials}] val_perplexity={trial.value:.2f} "
              f"(best epoch {row['best_epoch']}) params={trial.params}")

    return callback


def plot_search(study, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [t.value for t in study.trials]
    best_so_far = []
    best = float("inf")
    for v in values:
        best = min(best, v)
        best_so_far.append(best)

    plt.figure()
    plt.plot(range(1, len(values) + 1), values, "o", alpha=0.4, label="trial")
    plt.plot(range(1, len(best_so_far) + 1), best_so_far, "-", label="best so far")
    plt.xlabel("trial")
    plt.ylabel("val perplexity")
    plt.title("Optuna search over RNN embed_size / dropout / lr")
    plt.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--hidden-size", type=int, default=128,
                         help="fixed, not searched -- a representative architecture size to tune embed_size/"
                              "dropout/lr for, since those need to stay valid when you later vary hidden_size "
                              "in experiment.py's sweep")
    parser.add_argument("--num-layers", type=int, default=1, help="fixed, not searched (see --hidden-size)")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--search-epochs", type=int, default=5,
                         help="max epochs per trial during the search (early stopping with --patience may "
                              "stop sooner) -- kept short since it runs n_trials times")
    parser.add_argument("--final-epochs", type=int, default=20,
                         help="max epochs to retrain the best found config for, with full history saved")
    parser.add_argument("--patience", type=int, default=5,
                         help="stop each search trial and the final retrain early if val perplexity hasn't "
                              "improved for this many epochs; trials are scored by, and the saved checkpoint "
                              "uses, the best epoch, not the last")
    parser.add_argument("--batch-size", type=int, default=20, help="mini-batch size, in paragraphs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word2id, id2word, train_ids, val_ids, test_ids = load_split_ids(args.data_dir)
    vocab_size = len(word2id)
    punct_ids = punctuation_ids(word2id)
    eos_id = word2id[EOS]

    # training batches are reshuffled every epoch, below
    val_batches = paragraph_batches(val_ids, args.batch_size, eos_id, device)
    test_batches = paragraph_batches(test_ids, args.batch_size, eos_id, device)

    results_dir = Path(args.results_dir)
    trials_csv = results_dir / "optuna_trials.csv"
    trial_logger = make_trial_logger(trials_csv, args.n_trials)

    print(f"searching {args.n_trials} trials, up to {args.search_epochs} epochs each "
          f"(early stopping: patience={args.patience}), "
          f"at fixed hidden_size={args.hidden_size} num_layers={args.num_layers} "
          f"(no checkpoints saved per trial)")
    print(f"each trial's result is appended to {trials_csv} as it finishes")
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_objective(train_ids, val_batches, vocab_size, eos_id, args.hidden_size, args.num_layers,
                                punct_ids, args.batch_size, args.search_epochs, args.patience, device, args.seed)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[trial_logger])

    print(f"\nbest trial: val_perplexity={study.best_value:.2f}")
    print(f"best params: {study.best_params}")

    search_plot = results_dir / "optuna_search.png"
    plot_search(study, search_plot)
    print(f"saved search plot to {search_plot}")

    best = study.best_params
    print(f"\nretraining best config for up to {args.final_epochs} epochs "
          f"(early stopping: patience={args.patience}, full history + test eval)...")
    torch.manual_seed(args.seed)
    model = LSTMLanguageModel(
        vocab_size, best["embed_size"], args.hidden_size, args.num_layers, best["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_ppl = float("inf")
    best_state = None
    best_optimizer_state = None
    best_epoch = 0
    epochs_since_improvement = 0
    start = time.time()
    for epoch in range(1, args.final_epochs + 1):
        train_batches = paragraph_batches(train_ids, args.batch_size, eos_id, device, shuffle=True,
                                          seed=args.seed + epoch)
        train_loss = train_epoch(model, train_batches, optimizer, criterion)
        val_metrics = evaluate_rnn(model, val_batches, criterion, punct_ids=punct_ids)
        print(
            f"epoch {epoch}: train_loss={train_loss:.3f} "
            f"val_ppl={val_metrics['perplexity']:.2f} "
            f"val_top5(all)={val_metrics['topk_acc'][5]:.3%} "
            f"val_top5(words)={val_metrics['topk_acc_words'][5]:.3%}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_nll": rnn_nll(model, train_batches, criterion),
            "val_nll": math.log(val_metrics["perplexity"]),
            "val_perplexity": val_metrics["perplexity"],
            "val_top5_acc": val_metrics["topk_acc"][5],
            "val_top5_acc_words": val_metrics["topk_acc_words"][5],
        })

        if val_metrics["perplexity"] < best_val_ppl:
            best_val_ppl = val_metrics["perplexity"]
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"no val improvement for {args.patience} epochs, stopping early at epoch {epoch} "
                      f"(best was epoch {best_epoch}, val_ppl={best_val_ppl:.2f})")
                break
    train_time = time.time() - start

    print(f"restoring epoch {best_epoch}'s weights (best val_ppl={best_val_ppl:.2f}) before final evaluation")
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)

    val_metrics = evaluate_rnn(model, val_batches, criterion, punct_ids=punct_ids)
    test_metrics = evaluate_rnn(model, test_batches, criterion, punct_ids=punct_ids)
    print_eval("test", test_metrics)

    config = {
        "model_type": "RNN",
        "vocab_size": vocab_size,
        "embed_size": best["embed_size"],
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": best["dropout"],
        "batch_size": args.batch_size,
        "epochs": args.final_epochs,
        "best_epoch": best_epoch,
        "early_stopped": len(history) < args.final_epochs,
        "patience": args.patience,
        "lr": best["lr"],
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "train_time_s": round(train_time, 1),
        "selected_by": "optuna",
        "n_trials": args.n_trials,
        "search_epochs": args.search_epochs,
    }
    out_path = Path(args.checkpoints_dir) / "rnn_optuna_best"
    # history includes post-best epochs; load_rnn_training_state trims them on resume
    save_rnn_checkpoint(
        model, optimizer, best_epoch, config, {"val": val_metrics, "test": test_metrics},
        out_path, history=history,
    )
    print(f"saved best-config checkpoint: {out_path}.pt + {out_path}.json")


if __name__ == "__main__":
    main()
