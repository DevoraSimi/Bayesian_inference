import argparse
import sys
from pathlib import Path

import torch

from checkpoints import load_pickle_checkpoint, load_rnn_checkpoint
from data_utils import EOS, UNK, encode_sentence, load_vocab, punctuation_ids, tokenize
from hmm import suggest_next_words as hmm_suggest
from ngram import suggest_next_words as ngram_suggest
from rnn import mc_dropout_predict, suggest_next_words as rnn_suggest


def print_suggestions(prefix, word2id, id2word, models, device, k, mc_samples, exclude_ids):
    tokens = tokenize(prefix)
    if not tokens:
        print("(empty prefix, skipping)")
        return
    ids = encode_sentence(tokens, word2id)

    print(f"\nprefix: {prefix!r}")
    if models.get("hmm") is not None:
        print(f"  HMM:    {hmm_suggest(models['hmm'], ids, id2word, k, exclude_ids)}")
    if models.get("vbhmm") is not None:
        print(f"  VB-HMM: {hmm_suggest(models['vbhmm'], ids, id2word, k, exclude_ids)}")
    if models.get("ngram") is not None:
        print(f"  Ngram:  {ngram_suggest(models['ngram'], ids, id2word, k, exclude_ids)}")
    if models.get("rnn") is not None:
        # the RNN reads each paragraph starting from <eos> (see rnn.paragraph_batches)
        rnn_ids = [word2id[EOS]] + ids
        print(f"  RNN:    {rnn_suggest(models['rnn'], rnn_ids, id2word, device, k, exclude_ids)}")
        if mc_samples:
            mc = mc_dropout_predict(models["rnn"], rnn_ids, id2word, device, k, mc_samples, exclude_ids)
            formatted = ", ".join(f"{w} (p={p:.3f} +/-{s:.3f})" for w, p, s in mc)
            print(f"  RNN MC-Dropout ({mc_samples} samples): {formatted}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--hmm-checkpoint", default=None, help="e.g. checkpoints/hmm_16 (no extension)")
    parser.add_argument("--vbhmm-checkpoint", default=None, help="e.g. checkpoints/vbhmm_16 (no extension)")
    parser.add_argument("--ngram-checkpoint", default=None, help="e.g. checkpoints/ngram_3_a30.0 (no extension)")
    parser.add_argument("--rnn-checkpoint", default=None, help="e.g. checkpoints/rnn_h256_l2_d0.293 (no extension)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--mc-dropout", type=int, default=0, metavar="N_SAMPLES",
        help="if set (with --rnn-checkpoint), also show MC-Dropout uncertainty using N_SAMPLES stochastic passes",
    )
    parser.add_argument("--no-punct", action="store_true",
                        help="suggest words only, hiding punctuation tokens (<unk>/<eos> are always hidden)")
    parser.add_argument("--text", default=None, help="single prefix; if omitted, reads lines from stdin")
    args = parser.parse_args()

    checkpoint_args = (args.hmm_checkpoint, args.vbhmm_checkpoint, args.ngram_checkpoint, args.rnn_checkpoint)
    if not any(checkpoint_args):
        parser.error("provide at least one of --hmm-checkpoint, --vbhmm-checkpoint, --ngram-checkpoint, --rnn-checkpoint")
    if args.mc_dropout and not args.rnn_checkpoint:
        parser.error("--mc-dropout requires --rnn-checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    word2id, id2word = load_vocab(data_dir / "vocab.json")
    # <unk>/<eos> are never useful suggestions; punctuation optionally
    exclude_ids = {word2id[UNK], word2id[EOS]}
    if args.no_punct:
        exclude_ids |= punctuation_ids(word2id)

    models = {}
    if args.hmm_checkpoint:
        models["hmm"], _ = load_pickle_checkpoint(args.hmm_checkpoint)
    if args.vbhmm_checkpoint:
        models["vbhmm"], _ = load_pickle_checkpoint(args.vbhmm_checkpoint)
    if args.ngram_checkpoint:
        models["ngram"], _ = load_pickle_checkpoint(args.ngram_checkpoint)
    if args.rnn_checkpoint:
        models["rnn"], _ = load_rnn_checkpoint(args.rnn_checkpoint, device)

    if args.text is not None:
        print_suggestions(args.text, word2id, id2word, models, device, args.k, args.mc_dropout, exclude_ids)
        return

    print("enter a text prefix (Ctrl-D to quit):")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        print_suggestions(line, word2id, id2word, models, device, args.k, args.mc_dropout, exclude_ids)


if __name__ == "__main__":
    main()
