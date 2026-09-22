import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from data_utils import EOS, encode_sentences, load_sentences, load_vocab
from hmm_model import evaluate_hmm, train_hmm
from rnn_model import LSTMLanguageModel, batchify, evaluate, sentences_to_stream, train_epoch


def run_hmm_sweep(train_ids, val_ids, test_ids, vocab_size, states_list, n_iter, seed):
    results = []
    for n_states in states_list:
        print(f"[HMM] n_states={n_states}")
        start = time.time()
        model = train_hmm(train_ids, n_states, vocab_size, n_iter, seed)
        train_time = time.time() - start

        val_metrics = evaluate_hmm(model, val_ids)
        test_metrics = evaluate_hmm(model, test_ids)
        results.append({
            "model": "HMM",
            "param": n_states,
            "train_time_s": round(train_time, 1),
            "converged": model.monitor_.converged,
            "n_iter_used": model.monitor_.iter,
            "val_perplexity": round(val_metrics["perplexity"], 2),
            "test_perplexity": round(test_metrics["perplexity"], 2),
            "test_top1_acc": round(test_metrics["topk_acc"][1], 4),
            "test_top5_acc": round(test_metrics["topk_acc"][5], 4),
        })
        print(results[-1])
    return results


def run_rnn_sweep(train_data, val_data, test_data, vocab_size, hidden_list, epochs, bptt, lr, seed, device):
    results = []
    for hidden_size in hidden_list:
        print(f"[RNN] hidden_size={hidden_size}")
        torch.manual_seed(seed)
        model = LSTMLanguageModel(vocab_size, hidden_size=hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(epochs):
            train_epoch(model, train_data, optimizer, criterion, bptt)
        train_time = time.time() - start

        val_metrics = evaluate(model, val_data, criterion, bptt)
        test_metrics = evaluate(model, test_data, criterion, bptt)
        results.append({
            "model": "RNN",
            "param": hidden_size,
            "train_time_s": round(train_time, 1),
            "val_perplexity": round(val_metrics["perplexity"], 2),
            "test_perplexity": round(test_metrics["perplexity"], 2),
            "test_top1_acc": round(test_metrics["topk_acc"][1], 4),
            "test_top5_acc": round(test_metrics["topk_acc"][5], 4),
        })
        print(results[-1])
    return results


def save_csv(results, path):
    fieldnames = ["model", "param", "train_time_s", "converged", "n_iter_used",
                  "val_perplexity", "test_perplexity", "test_top1_acc", "test_top5_acc"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model_name in ("HMM", "RNN"):
        rows = [r for r in results if r["model"] == model_name]
        if not rows:
            continue
        xs = [r["param"] for r in rows]
        ys = [r["test_perplexity"] for r in rows]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("n_states" if model_name == "HMM" else "hidden_size")
        plt.ylabel("test perplexity")
        plt.title(f"{model_name}: perplexity vs. size")
        plt.savefig(Path(out_dir) / f"{model_name.lower()}_perplexity.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--hmm-states", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--hmm-iter", type=int, default=30)
    parser.add_argument("--rnn-hidden", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--rnn-epochs", type=int, default=8)
    parser.add_argument("--bptt", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    word2id, id2word = load_vocab(data_dir / "vocab.json")
    vocab_size = len(word2id)
    eos_id = word2id[EOS]

    train_sents = load_sentences(data_dir / "train.txt")
    val_sents = load_sentences(data_dir / "val.txt")
    test_sents = load_sentences(data_dir / "test.txt")

    train_ids = encode_sentences(train_sents, word2id)
    val_ids = encode_sentences(val_sents, word2id)
    test_ids = encode_sentences(test_sents, word2id)

    hmm_results = run_hmm_sweep(
        train_ids, val_ids, test_ids, vocab_size, args.hmm_states, args.hmm_iter, args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = batchify(sentences_to_stream(train_ids, eos_id), args.batch_size, device)
    val_data = batchify(sentences_to_stream(val_ids, eos_id), args.batch_size, device)
    test_data = batchify(sentences_to_stream(test_ids, eos_id), args.batch_size, device)

    rnn_results = run_rnn_sweep(
        train_data, val_data, test_data, vocab_size, args.rnn_hidden,
        args.rnn_epochs, args.bptt, args.lr, args.seed, device,
    )

    all_results = hmm_results + rnn_results
    save_csv(all_results, results_dir / "comparison.csv")
    plot_results(all_results, results_dir)

    print("\n=== summary ===")
    for r in all_results:
        print(r)
    print(f"\nsaved results to {results_dir}/comparison.csv and plots")


if __name__ == "__main__":
    main()
