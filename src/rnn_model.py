import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from data_utils import (
    encode_sentence,
    encode_sentences,
    load_sentences,
    load_vocab,
    punctuation_ids,
    tokenize,
    EOS,
)


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.decoder(self.dropout(out))
        return logits, hidden


def sentences_to_stream(id_sequences, eos_id):
    stream = []
    for seq in id_sequences:
        stream.extend(seq)
        stream.append(eos_id)
    return torch.tensor(stream, dtype=torch.long)


def batchify(data, batch_size, device):
    n_batches = len(data) // batch_size
    data = data[: n_batches * batch_size]
    return data.view(batch_size, -1).to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, source.size(1) - 1 - i)
    x = source[:, i : i + seq_len]
    y = source[:, i + 1 : i + 1 + seq_len]
    return x, y


def train_epoch(model, data, optimizer, criterion, bptt, clip=0.25):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, criterion, bptt, k_list=(1, 5, 10), punct_ids=frozenset()):
    """Reports top-k accuracy both over all tokens and restricted to
    positions where the true next token isn't in punct_ids (real words only)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hits_all = {k: 0 for k in k_list}
    hits_words = {k: 0 for k in k_list}
    n_predictions = 0
    n_word_predictions = 0
    punct_tensor = (
        torch.tensor(sorted(punct_ids), dtype=torch.long, device=data.device)
        if punct_ids else None
    )

    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n

        if punct_tensor is not None:
            word_mask = ~torch.isin(y, punct_tensor)
        else:
            word_mask = torch.ones_like(y, dtype=torch.bool)

        topk = logits.topk(max(k_list), dim=-1).indices
        correct = (topk == y.unsqueeze(-1))
        for k in k_list:
            hit_k = correct[..., :k].any(-1)
            hits_all[k] += hit_k.sum().item()
            hits_words[k] += (hit_k & word_mask).sum().item()
        n_predictions += n
        n_word_predictions += word_mask.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    topk_acc = {k: hits_all[k] / n_predictions for k in k_list}
    topk_acc_words = {k: hits_words[k] / n_word_predictions for k in k_list}
    return {"perplexity": perplexity, "topk_acc": topk_acc, "topk_acc_words": topk_acc_words}


@torch.no_grad()
def suggest_next_words(model, word2id, id2word, prefix_text, device, k=5):
    tokens = tokenize(prefix_text)
    ids = encode_sentence(tokens, word2id)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_idx = probs.topk(k)
    return [(id2word[i], float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--bptt", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    word2id, id2word = load_vocab(data_dir / "vocab.json")
    vocab_size = len(word2id)
    eos_id = word2id[EOS]
    punct_ids = punctuation_ids(word2id)

    train_sents = load_sentences(data_dir / "train.txt")
    val_sents = load_sentences(data_dir / "val.txt")
    test_sents = load_sentences(data_dir / "test.txt")

    train_stream = sentences_to_stream(encode_sentences(train_sents, word2id), eos_id)
    val_stream = sentences_to_stream(encode_sentences(val_sents, word2id), eos_id)
    test_stream = sentences_to_stream(encode_sentences(test_sents, word2id), eos_id)

    train_data = batchify(train_stream, args.batch_size, device)
    val_data = batchify(val_stream, args.batch_size, device)
    test_data = batchify(test_stream, args.batch_size, device)

    model = LSTMLanguageModel(
        vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"training RNN: hidden={args.hidden_size}, layers={args.num_layers}, vocab={vocab_size}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, criterion, args.bptt)
        val_metrics = evaluate(model, val_data, criterion, args.bptt, punct_ids=punct_ids)
        print(
            f"epoch {epoch}: train_loss={train_loss:.3f} "
            f"val_ppl={val_metrics['perplexity']:.2f} "
            f"val_top5(all)={val_metrics['topk_acc'][5]:.3%} "
            f"val_top5(words)={val_metrics['topk_acc_words'][5]:.3%}"
        )
    train_time = time.time() - start
    print(f"train time: {train_time:.1f}s")

    test_metrics = evaluate(model, test_data, criterion, args.bptt, punct_ids=punct_ids)
    print(
        f"test: perplexity={test_metrics['perplexity']:.2f} "
        f"top-5 acc (all)={test_metrics['topk_acc'][5]:.3%} "
        f"top-5 acc (words only)={test_metrics['topk_acc_words'][5]:.3%}"
    )

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"rnn_{args.hidden_size}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_size": args.embed_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        out_path,
    )
    print(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
