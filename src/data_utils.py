import json
import random
import re
from pathlib import Path

UNK = "<unk>"
EOS = "<eos>"

WORD_RE = re.compile(r"[a-zA-Z’']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def split_sentences(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def text_to_sentences(text, min_len=2, max_len=50):
    sentences = []
    for raw in split_sentences(text):
        tokens = tokenize(raw)
        if min_len <= len(tokens) <= max_len: # remove outliers that are too short or too long
            sentences.append(tokens)
    return sentences


def build_vocab(sentences, vocab_size):
    from collections import Counter

    counts = Counter(tok for sent in sentences for tok in sent)
    most_common = counts.most_common(vocab_size - 2)
    word2id = {UNK: 0, EOS: 1}
    for word, _ in most_common:
        word2id[word] = len(word2id)
    return word2id


def save_vocab(word2id, path):
    Path(path).write_text(json.dumps(word2id, indent=2, ensure_ascii=False))


def load_vocab(path):
    word2id = json.loads(Path(path).read_text())
    id2word = [None] * len(word2id)
    for word, idx in word2id.items():
        id2word[idx] = word
    return word2id, id2word


def encode_sentence(tokens, word2id):
    unk = word2id[UNK]
    return [word2id.get(tok, unk) for tok in tokens]


def save_sentences(sentences, path):
    with open(path, "w") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")


def load_sentences(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def encode_sentences(sentences, word2id):
    return [encode_sentence(sent, word2id) for sent in sentences]


def split_books(book_sentences, val_frac=0.1, test_frac=0.1, seed=42):
    """Assign whole books/stories to train/val/test so no story's sentences
    leak across splits. Units are shuffled, then greedily filled into test,
    then val, then train, until each split's token-count target is reached."""
    items = list(book_sentences.items())
    random.Random(seed).shuffle(items)

    total_tokens = sum(len(sent) for _, sents in items for sent in sents)
    target_val = total_tokens * val_frac
    target_test = total_tokens * test_frac

    train, val, test = [], [], []
    val_tokens = test_tokens = 0
    assignment = {}
    for name, sents in items:
        n_tokens = sum(len(sent) for sent in sents)
        if test_tokens < target_test:
            test.extend(sents)
            test_tokens += n_tokens
            assignment[name] = "test"
        elif val_tokens < target_val:
            val.extend(sents)
            val_tokens += n_tokens
            assignment[name] = "val"
        else:
            train.extend(sents)
            assignment[name] = "train"
    return train, val, test, assignment
