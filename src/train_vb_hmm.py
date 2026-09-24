import argparse
import time
from pathlib import Path

from checkpoints import save_pickle_checkpoint
from data_utils import load_split_ids, punctuation_ids
from evaluation import evaluate_hmm, print_eval
from vb_hmm import train_vb_hmm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--n-states", type=int, default=16)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--alpha0", type=float, default=1.0, help="Dirichlet prior concentration for transitions")
    parser.add_argument("--beta0", type=float, default=1.0, help="Dirichlet prior concentration for emissions")
    parser.add_argument("--pi0", type=float, default=1.0, help="Dirichlet prior concentration for initial state")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    word2id, id2word, train_ids, val_ids, test_ids = load_split_ids(args.data_dir)
    vocab_size = len(word2id)
    punct_ids = punctuation_ids(word2id)

    print(f"training VB-HMM: n_states={args.n_states}, vocab={vocab_size}")
    start = time.time()
    model = train_vb_hmm(
        train_ids, args.n_states, vocab_size, args.n_iter, args.seed,
        args.alpha0, args.beta0, args.pi0, val_sequences=val_ids,
    )
    train_time = time.time() - start
    print(f"train time: {train_time:.1f}s, converged: {model.monitor_.converged}, iters: {model.monitor_.iter}")

    val_metrics = evaluate_hmm(model, val_ids, punct_ids=punct_ids)
    test_metrics = evaluate_hmm(model, test_ids, punct_ids=punct_ids)
    print_eval("val", val_metrics)
    print_eval("test", test_metrics)

    config = {
        "model_type": "VB-HMM",
        "n_states": args.n_states,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "alpha0": args.alpha0,
        "beta0": args.beta0,
        "pi0": args.pi0,
        "data_dir": str(args.data_dir),
        "train_time_s": round(train_time, 1),
        "converged": model.monitor_.converged,
        "n_iter_used": model.monitor_.iter,
        "total_train_tokens": sum(len(s) for s in train_ids),
    }
    metrics = {"val": val_metrics, "test": test_metrics}

    out_path = Path(args.checkpoints_dir) / f"vbhmm_{args.n_states}"
    save_pickle_checkpoint(model, config, metrics, out_path,
                            history=model.monitor_.history, val_history=model.monitor_.val_history)
    print(f"saved checkpoint: {out_path}.pkl + {out_path}.json")


if __name__ == "__main__":
    main()
