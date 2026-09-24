import argparse
import time
from pathlib import Path

from checkpoints import save_pickle_checkpoint
from data_utils import load_split_ids, punctuation_ids
from evaluation import evaluate_ngram, print_eval
from ngram import train_ngram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--order", type=int, default=3, help="2=bigram, 3=trigram, ...")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet smoothing concentration")
    args = parser.parse_args()

    word2id, id2word, train_ids, val_ids, test_ids = load_split_ids(args.data_dir)
    vocab_size = len(word2id)
    punct_ids = punctuation_ids(word2id)

    print(f"training {args.order}-gram: vocab={vocab_size}, alpha={args.alpha}")
    start = time.time()
    model = train_ngram(train_ids, args.order, vocab_size, args.alpha)
    train_time = time.time() - start
    print(f"train time: {train_time:.1f}s, distinct contexts: {len(model.context_counts)}")

    val_metrics = evaluate_ngram(model, val_ids, punct_ids=punct_ids)
    test_metrics = evaluate_ngram(model, test_ids, punct_ids=punct_ids)
    print_eval("val", val_metrics)
    print_eval("test", test_metrics)

    config = {
        "model_type": "Ngram",
        "order": args.order,
        "alpha": args.alpha,
        "vocab_size": vocab_size,
        "data_dir": str(args.data_dir),
        "train_time_s": round(train_time, 1),
        "n_contexts": len(model.context_counts),
    }
    metrics = {"val": val_metrics, "test": test_metrics}

    out_path = Path(args.checkpoints_dir) / f"ngram_{args.order}"
    save_pickle_checkpoint(model, config, metrics, out_path)
    print(f"saved checkpoint: {out_path}.pkl + {out_path}.json")


if __name__ == "__main__":
    main()
