import argparse
import json
import re
from pathlib import Path

from data_utils import (
    build_vocab,
    encode_sentences,
    save_sentences,
    save_vocab,
    split_books,
    text_to_sentences,
)

START_RE = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)

CONTENTS_HEADING_RE = re.compile(r"^\s*CONTENTS\s*$", re.IGNORECASE | re.MULTILINE)
ROMAN_PREFIX_RE = re.compile(r"^\s*[IVXLCDM]+\.?\s+")


def strip_gutenberg_boilerplate(text):
    start_match = START_RE.search(text)
    end_match = END_RE.search(text)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end]


def extract_story_titles(text):
    """Find a 'Contents' listing and return (ordered titles, index right after
    the listing), or ([], None) if no such listing is found (e.g. a novel
    with no internal short-story structure)."""
    heading_match = CONTENTS_HEADING_RE.search(text)
    if not heading_match:
        return [], None

    after = text[heading_match.end():]
    titles = []
    blank_run = 0
    consumed_chars = 0
    for line in after.split("\n"):
        consumed_chars += len(line) + 1
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if titles and blank_run >= 2:
                break
            continue
        blank_run = 0
        title = ROMAN_PREFIX_RE.sub("", stripped).strip()
        if title:
            titles.append(title)

    body_start = heading_match.end() + consumed_chars
    return titles, body_start


FUZZY_TITLE_WORD_RE = re.compile(r"[A-Za-z’']+")
FUZZY_TITLE_SEP = r"[\s_*\"'‘’“”-]*"


def build_fuzzy_title_pattern(title):
    """A title's own words, joined by a separator that tolerates the extra
    markup (italics underscores, quote marks) Gutenberg's body headings
    sometimes add that the contents listing doesn't have (or vice versa)."""
    words = FUZZY_TITLE_WORD_RE.findall(title)
    if not words:
        return None
    return FUZZY_TITLE_SEP.join(re.escape(w) for w in words)


def split_into_stories(text):
    """Split an anthology's text into per-story chunks by locating each
    contents-listed title as a heading in the body text, in order. Falls
    back to a single '__whole__' chunk if fewer than 2 titles are found
    (e.g. a novel) or too few headings can be located."""
    titles, body_start = extract_story_titles(text)
    if len(titles) < 2:
        return {"__whole__": text}

    chapter_like = sum(1 for t in titles if re.match(r"(?i)^(chapter|part)\b", t))
    if chapter_like / len(titles) > 0.5:
        # this "Contents" listing is a novel's chapter list, not distinct
        # stories -- treat the whole novel as one unit
        return {"__whole__": text}

    body = text[body_start:]
    cursor = 0
    positions = []
    missing = []
    for title in titles:
        search_title = title.split(":")[0].strip()
        pattern = build_fuzzy_title_pattern(search_title)
        match = re.search(pattern, body[cursor:], re.IGNORECASE) if pattern else None
        if match is None:
            missing.append(title)
            continue
        start = cursor + match.start()
        positions.append((title, start))
        cursor = start + (match.end() - match.start())

    if missing:
        print(f"  warning: could not locate heading(s), merged into neighbors: {missing}")

    if len(positions) < 2:
        return {"__whole__": text}

    stories = {}
    for i, (title, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(body)
        stories[title] = body[start:end]
    return stories


def load_units(raw_dir):
    """Load every book, splitting anthologies into per-story units and
    leaving novels as one unit each. Returns {unit_name: text}."""
    units = {}
    for path in sorted(Path(raw_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        text = strip_gutenberg_boilerplate(raw)
        stories = split_into_stories(text)
        if list(stories.keys()) == ["__whole__"]:
            units[path.stem] = text
        else:
            for title, story_text in stories.items():
                units[f"{path.stem}::{title}"] = story_text
    return units


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

    units = load_units(args.raw_dir)
    n_files = len(list(Path(args.raw_dir).glob("*.txt")))
    print(f"split {n_files} files into {len(units)} units (novels + individual stories)")

    unit_sentences = {name: text_to_sentences(text) for name, text in units.items()}
    train, val, test, assignment = split_books(
        unit_sentences, args.val_frac, args.test_frac, args.seed
    )

    for split_name in ("test", "val"):
        names = sorted(n for n, s in assignment.items() if s == split_name)
        print(f"{split_name} units ({len(names)}):")
        for n in names:
            print(f"  {n}")

    metadata = {
        name: {
            "split": split_name,
            "n_sentences": len(unit_sentences[name]),
            "n_tokens": sum(len(sent) for sent in unit_sentences[name]),
        }
        for name, split_name in assignment.items()
    }
    (out_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    word2id = build_vocab(train, args.vocab_size)  # build vocab from training set only
    save_vocab(word2id, out_dir / "vocab.json")

    save_sentences(train, out_dir / "train.txt")
    save_sentences(val, out_dir / "val.txt")
    save_sentences(test, out_dir / "test.txt")

    unk_id = word2id["<unk>"]
    print(f"sentences: train={len(train)} val={len(val)} test={len(test)}")
    print(f"vocab size: {len(word2id)}")
    for split_name, sentences in (("train", train), ("val", val), ("test", test)):
        ids = encode_sentences(sentences, word2id)
        total = sum(len(sent) for sent in ids)
        unk = sum(tok == unk_id for sent in ids for tok in sent)
        print(f"{split_name} tokens: {total}, unk rate: {unk / total:.3%}")


if __name__ == "__main__":
    main()
