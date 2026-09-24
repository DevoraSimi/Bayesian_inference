import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from checkpoints import load_rnn_training_state, save_rnn_checkpoint, with_ext
from data_utils import EOS, load_split_ids, punctuation_ids
from evaluation import evaluate_rnn, print_eval
from rnn import LSTMLanguageModel, batchify, sentences_to_stream, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--bptt", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="resume from an existing checkpoint at the target path, if one exists")
    parser.add_argument("--save-every", type=int, default=1, help="save a resumable checkpoint every N epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word2id, id2word, train_ids, val_ids, test_ids = load_split_ids(args.data_dir)
    vocab_size = len(word2id)
    eos_id = word2id[EOS]
    punct_ids = punctuation_ids(word2id)

    train_data = batchify(sentences_to_stream(train_ids, eos_id), args.batch_size, device)
    val_data = batchify(sentences_to_stream(val_ids, eos_id), args.batch_size, device)
    test_data = batchify(sentences_to_stream(test_ids, eos_id), args.batch_size, device)

    config = {
        "model_type": "RNN",
        "vocab_size": vocab_size,
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "bptt": args.bptt,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "data_dir": str(args.data_dir),
    }

    out_path = Path(args.checkpoints_dir) / f"rnn_{args.hidden_size}"
    history = []
    start_epoch = 1
    if args.resume and with_ext(out_path, ".pt").exists():
        model, optimizer, start_epoch, history = load_rnn_training_state(out_path, device, args.lr)
        print(f"resuming from epoch {start_epoch} ({len(history)} epochs of history loaded)")
        if start_epoch > args.epochs:
            print(f"checkpoint is already at epoch {start_epoch - 1} >= --epochs {args.epochs}; nothing to train")
    else:
        model = LSTMLanguageModel(
            vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    print(f"training RNN: hidden={args.hidden_size}, layers={args.num_layers}, vocab={vocab_size}")
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, criterion, args.bptt)
        val_metrics = evaluate_rnn(model, val_data, criterion, args.bptt, punct_ids=punct_ids)
        print(
            f"epoch {epoch}: train_loss={train_loss:.3f} "
            f"val_ppl={val_metrics['perplexity']:.2f} "
            f"val_top5(all)={val_metrics['topk_acc'][5]:.3%} "
            f"val_top5(words)={val_metrics['topk_acc_words'][5]:.3%}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_perplexity": val_metrics["perplexity"],
            "val_top5_acc": val_metrics["topk_acc"][5],
            "val_top5_acc_words": val_metrics["topk_acc_words"][5],
        })

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_rnn_checkpoint(
                model, optimizer, epoch, config, {"val": val_metrics, "test": None},
                out_path, history=history,
            )
    train_time = time.time() - start
    print(f"train time: {train_time:.1f}s")

    val_metrics = evaluate_rnn(model, val_data, criterion, args.bptt, punct_ids=punct_ids)
    test_metrics = evaluate_rnn(model, test_data, criterion, args.bptt, punct_ids=punct_ids)
    print_eval("test", test_metrics)

    config["train_time_s"] = round(train_time, 1)
    save_rnn_checkpoint(
        model, optimizer, args.epochs, config, {"val": val_metrics, "test": test_metrics},
        out_path, history=history,
    )
    print(f"saved checkpoint: {out_path}.pt + {out_path}.json")


if __name__ == "__main__":
    main()
