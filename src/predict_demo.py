import argparse
import pickle
import sys
from pathlib import Path

import torch

from data_utils import load_vocab, tokenize, encode_sentence
from hmm_model import suggest_next_words as hmm_suggest
from rnn_model import LSTMLanguageModel
from rnn_model import suggest_next_words as rnn_suggest


def load_hmm(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_rnn(path, device):
    ckpt = torch.load(path, map_location=device)
    model = LSTMLanguageModel(
        ckpt["vocab_size"], ckpt["embed_size"], ckpt["hidden_size"],
        ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def print_suggestions(prefix, word2id, id2word, hmm_model, rnn_model, device, k):
    tokens = tokenize(prefix)
    if not tokens:
        print("(empty prefix, skipping)")
        return
    ids = encode_sentence(tokens, word2id)

    print(f"\nprefix: {prefix!r}")
    if hmm_model is not None:
        print(f"  HMM: {hmm_suggest(hmm_model, ids, id2word, k)}")
    if rnn_model is not None:
        print(f"  RNN: {rnn_suggest(rnn_model, word2id, id2word, prefix, device, k)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--hmm-model", default=None, help="path to a saved hmm_*.pkl")
    parser.add_argument("--rnn-model", default=None, help="path to a saved rnn_*.pt")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--text", default=None, help="single prefix; if omitted, reads lines from stdin")
    args = parser.parse_args()

    if not args.hmm_model and not args.rnn_model:
        parser.error("provide at least one of --hmm-model or --rnn-model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    word2id, id2word = load_vocab(data_dir / "vocab.json")

    hmm_model = load_hmm(args.hmm_model) if args.hmm_model else None
    rnn_model = load_rnn(args.rnn_model, device) if args.rnn_model else None

    if args.text is not None:
        print_suggestions(args.text, word2id, id2word, hmm_model, rnn_model, device, args.k)
        return

    print("enter a text prefix (Ctrl-D to quit):")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        print_suggestions(line, word2id, id2word, hmm_model, rnn_model, device, args.k)


if __name__ == "__main__":
    main()
