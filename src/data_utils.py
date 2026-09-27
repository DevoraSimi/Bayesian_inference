import json
import random
import re
from pathlib import Path

import numpy as np

UNK = "<unk>"
EOS = "<eos>"

# Preprocessing follows WikiText (Merity et al., 2016), which was tokenized
# with the Moses tokenizer: one paragraph per sequence (no sentence
# splitting), original case kept, every punctuation mark a token, numbers
# kept, vocab = every training token seen at least 3 times (we default to 2,
# since this corpus is ~4x smaller than WikiText-2).

# Title abbreviations whose period is part of the word ("Mr." is one token),
# like Moses's non-breaking prefixes. Excludes ambiguous ones like "No." and
# "etc." that often end a sentence.
ABBREVIATIONS = ("Mrs", "Mr", "Ms", "Dr", "St", "Prof", "Rev", "Col", "Capt",
                 "Lt", "Gen", "Sgt", "Jr", "Sr")

# Unify typographic variants so the same symbol is one vocab entry, not several
# (the Gutenberg texts mix curly and straight quotes, "—" and "--", etc.).
NORMALIZE_MAP = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})

# WikiText's "@x@" markers: a hyphen inside a word and a separator inside a
# number become their own tokens ("well @-@ known", "1 @,@ 000", "8 @.@ 5"),
# so both halves stay in the vocab instead of one rare compound.
INNER_HYPHEN_RE = re.compile(r"(?<=[^\W_])-(?=[^\W_])")
NUMBER_SEP_RE = re.compile(r"(?<=\d)([.,])(?=\d)")

# A token is one of: an abbreviation with its period; an @x@ marker; a number
# (digits plus any letter suffix: "1890s", "221B"); a clitic split off the
# word before it, as Moses does ("don't" -> "don 't", "Holmes's" -> "Holmes
# 's"); a word; an ellipsis; or any other single non-space, non-word
# character (punctuation, no fixed list). "_" (Gutenberg's italics markup)
# matches none of these and is dropped.
_LETTER = r"[^\W\d_]"
TOKEN_RE = re.compile(
    r"\b(?i:" + "|".join(ABBREVIATIONS) + r")\."
    + r"|@[-,.]@"
    + r"|\d[^\W_]*"
    + r"|(?<=" + _LETTER + r")'" + _LETTER + r"+"
    + r"|" + _LETTER + r"+"
    + r"|\.\.\."
    + r"|[^\w\s]"
)


def normalize_text(text):
    text = text.translate(NORMALIZE_MAP)
    text = text.replace("--", "—").replace("…", "...")
    return text


def tokenize(text):
    text = normalize_text(text)
    text = INNER_HYPHEN_RE.sub(" @-@ ", text)
    text = NUMBER_SEP_RE.sub(r" @\1@ ", text)
    return TOKEN_RE.findall(text)


def punctuation_ids(word2id):
    """Vocabulary ids that are not real words: punctuation tokens and <eos>.
    Used to exclude them from the words-only metrics and from word-only
    suggestions. (<unk> stands for rare words, so it counts as a word.)"""
    punct = {idx for w, idx in word2id.items()
             if w not in (UNK, EOS) and not any(c.isalnum() for c in w)}
    return punct | {word2id[EOS]}


CHAPTER_HEADING_RE = re.compile(r"(?i)(?:chapter|part)\s+(?:\d+|[IVXLC]+)\b")


def is_heading(paragraph):
    """Titles and front matter, not prose: all-caps paragraphs ("THE
    RED-HEADED LEAGUE", "CHAPTER I.") and ones with no final punctuation
    ("Chapter 2. The Curse of the Baskervilles", "By A. Conan Doyle",
    illustration captions). Prose and dialogue always end in punctuation."""
    all_caps = any(c.isalpha() for c in paragraph) and paragraph == paragraph.upper()
    numbered = CHAPTER_HEADING_RE.match(paragraph) is not None
    return all_caps or numbered or paragraph[-1].isalnum()


def split_paragraphs(text):
    """Paragraphs are separated by blank lines; line breaks inside one are
    just the book's hard wrapping."""
    paragraphs = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    return [p for p in paragraphs if p and not is_heading(p)]


def text_to_paragraphs(text, min_len=2):
    paragraphs = []
    for raw in split_paragraphs(text):
        tokens = tokenize(raw)
        if len(tokens) >= min_len:
            paragraphs.append(tokens)
    return paragraphs


def build_vocab(sequences, min_count=2, max_size=None):
    """Every token seen at least `min_count` times (WikiText uses 3; 2 suits
    this smaller corpus), most frequent first; optionally capped at
    `max_size` entries in total."""
    from collections import Counter

    counts = Counter(tok for seq in sequences for tok in seq)
    kept = [w for w, c in counts.most_common() if c >= min_count]
    if max_size is not None:
        kept = kept[:max_size - 2]
    word2id = {UNK: 0, EOS: 1}
    for word in kept:
        word2id[word] = len(word2id)
    return word2id


def save_vocab(word2id, path):
    Path(path).write_text(json.dumps(word2id, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vocab(path):
    word2id = json.loads(Path(path).read_text(encoding="utf-8"))
    id2word = [None] * len(word2id)
    for word, idx in word2id.items():
        id2word[idx] = word
    return word2id, id2word


def encode_sentence(tokens, word2id):
    unk = word2id[UNK]
    return [word2id.get(tok, unk) for tok in tokens]


def save_sentences(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def encode_sentences(sentences, word2id):
    return [encode_sentence(sent, word2id) for sent in sentences]


def load_split_ids(data_dir):
    """Loads vocab.json plus train/val/test.txt from a preprocess.py output
    directory and encodes each split to word-id sequences, each ending in
    <eos> (as in WikiText), so every model is trained and scored on the same
    tokens, including predicting where a paragraph ends. Shared by every
    training/experiment script so the load-and-encode boilerplate lives in
    exactly one place."""
    data_dir = Path(data_dir)
    word2id, id2word = load_vocab(data_dir / "vocab.json")
    eos = word2id[EOS]

    def load(name):
        return [seq + [eos] for seq in encode_sentences(load_sentences(data_dir / name), word2id)]

    return word2id, id2word, load("train.txt"), load("val.txt"), load("test.txt")


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


def top_k_words(dist, id2word, k, exclude_ids=()):
    """The k most likely (word, probability) pairs, skipping exclude_ids
    (e.g. <unk>, or punctuation for word-only suggestions)."""
    dist = np.array(dist, dtype=np.float64)
    dist[list(exclude_ids)] = -np.inf
    top_idx = np.argsort(-dist)[:k]
    return [(id2word[i], float(dist[i])) for i in top_idx]
