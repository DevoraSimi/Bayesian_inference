import torch
import torch.nn as nn


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


def sentences_to_stream(id_sequences):
    """Concatenates the sequences into one stream; each already ends in <eos>
    (added by data_utils.load_split_ids)."""
    stream = []
    for seq in id_sequences:
        stream.extend(seq)
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


def detach_hidden(hidden):
    """Cuts the graph at a window boundary: the state's values carry over to
    the next window (each batchify row is contiguous text), but gradients
    stop here, so BPTT stays truncated at bptt steps."""
    return tuple(h.detach() for h in hidden)


def train_epoch(model, data, optimizer, criterion, bptt, clip=0.25):
    model.train()
    total_loss = 0.0
    n_batches = 0
    hidden = None
    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)
        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        hidden = detach_hidden(hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def _masked(probs, exclude_ids):
    """probs with exclude_ids (e.g. <unk>, punctuation) set to -inf, so
    topk skips them; the returned probabilities are unchanged."""
    if not exclude_ids:
        return probs
    probs = probs.clone()
    probs[list(exclude_ids)] = float("-inf")
    return probs


@torch.no_grad()
def suggest_next_words(model, prefix_ids, id2word, device, k=5, exclude_ids=()):
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    logits, _ = model(x) # .eval() so dropout is disabled.
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_idx = _masked(probs, exclude_ids).topk(k)
    return [(id2word[i], float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]


@torch.no_grad()
def mc_dropout_predict(model, prefix_ids, id2word, device, k=5, n_samples=50, exclude_ids=()):
    """Monte Carlo Dropout (Gal & Ghahramani, 2016): keeps dropout ACTIVE at
    prediction time (model.train() instead of model.eval()) and runs
    n_samples independent stochastic forward passes on the same input, each
    with a different random dropout mask. Averaging the resulting softmax
    distributions approximates Bayesian model averaging over an implicit
    posterior over network weights; the standard deviation across samples
    for each word gives an approximate predictive uncertainty that a single
    deterministic forward pass (suggest_next_words) cannot provide.

    Degenerate case: if the model was trained with dropout=0, every sample
    is identical and std will be ~0 for all words -- expected, not a bug.
    """
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    was_training = model.training
    model.train()
    samples = []
    for _ in range(n_samples):
        logits, _ = model(x)
        samples.append(torch.softmax(logits[0, -1], dim=-1))
    model.train(was_training)

    probs_stack = torch.stack(samples)
    mean_probs = probs_stack.mean(dim=0)
    std_probs = probs_stack.std(dim=0)

    top_probs, top_idx = _masked(mean_probs, exclude_ids).topk(k)
    return [
        (id2word[i], float(p), float(std_probs[i]))
        for i, p in zip(top_idx.tolist(), top_probs.tolist())
    ]
