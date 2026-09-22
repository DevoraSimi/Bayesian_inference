import argparse
import re
from pathlib import Path

from data_utils import (
    build_vocab,
    encode_sentences,
    save_sentences,
    save_vocab,
    text_to_sentences,
    train_val_test_split,
)

START_RE = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg_boilerplate(text):
    start_match = START_RE.search(text)
    end_match = END_RE.search(text)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end]


def load_corpus(raw_dir):
    texts = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        texts.append(strip_gutenberg_boilerplate(raw))
    return "\n\n".join(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--vocab-size", type=int, default=15000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.raw_dir)
    sentences = text_to_sentences(corpus)
    train, val, test = train_val_test_split(
        sentences, args.val_frac, args.test_frac, args.seed
    )

    word2id = build_vocab(train, args.vocab_size) # build vocab from training set only
    save_vocab(word2id, out_dir / "vocab.json")

    save_sentences(train, out_dir / "train.txt")
    save_sentences(val, out_dir / "val.txt")
    save_sentences(test, out_dir / "test.txt")

    unk_id = word2id["<unk>"]
    train_ids = encode_sentences(train, word2id)
    unk_count = sum(tok == unk_id for sent in train_ids for tok in sent)
    total_count = sum(len(sent) for sent in train_ids)

    print(f"sentences: train={len(train)} val={len(val)} test={len(test)}")
    print(f"vocab size: {len(word2id)}")
    print(f"train tokens: {total_count}, unk rate: {unk_count / total_count:.3%}")


if __name__ == "__main__":
    main()
