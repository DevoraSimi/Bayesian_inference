import argparse
import time
from pathlib import Path

from checkpoints import save_pickle_checkpoint
from data_utils import EOS, load_split_ids, punctuation_ids
from evaluation import evaluate_ngram, print_eval
from ngram import alpha_tag, train_ngram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--order", type=int, default=3, help="2=bigram, 3=trigram, ...")
    parser.add_argument("--alpha", type=float, nargs="+", default=[100.0],
                        help="total Dirichlet prior mass pulling each context toward the next-shorter one: "
                             "one value shared by all levels, or one per level, unigram level first "
                             "(e.g. --order 3 --alpha 100 100 300)")
    args = parser.parse_args()
    if len(args.alpha) not in (1, args.order):
        parser.error(f"--alpha takes 1 value or {args.order} (one per level) for --order {args.order}")
    alpha = args.alpha[0] if len(args.alpha) == 1 else tuple(args.alpha)

    word2id, id2word, train_ids, val_ids, test_ids = load_split_ids(args.data_dir)
    vocab_size = len(word2id)
    punct_ids = punctuation_ids(word2id)

    print(f"training {args.order}-gram: vocab={vocab_size}, alpha={alpha}")
    start = time.time()
    model = train_ngram(train_ids, args.order, vocab_size, word2id[EOS], alpha)
    train_time = time.time() - start
    print(f"train time: {train_time:.1f}s, distinct contexts: {len(model.context_counts)}")

    val_metrics = evaluate_ngram(model, val_ids, punct_ids=punct_ids)
    test_metrics = evaluate_ngram(model, test_ids, punct_ids=punct_ids)
    print_eval("val", val_metrics)
    print_eval("test", test_metrics)

    config = {
        "model_type": "Ngram",
        "order": args.order,
        "alpha": args.alpha if len(args.alpha) > 1 else args.alpha[0],
        "vocab_size": vocab_size,
        "data_dir": str(args.data_dir),
        "train_time_s": round(train_time, 1),
        "n_contexts": len(model.context_counts),
    }
    metrics = {"val": val_metrics, "test": test_metrics}

    out_path = Path(args.checkpoints_dir) / f"ngram_{args.order}_a{alpha_tag(alpha)}"
    save_pickle_checkpoint(model, config, metrics, out_path)
    print(f"saved checkpoint: {out_path}.pkl + {out_path}.json")


if __name__ == "__main__":
    main()
