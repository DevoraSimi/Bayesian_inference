import random
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


# target value for padding; nn.CrossEntropyLoss's default ignore_index, so
# padded positions drop out of the loss without configuring the criterion
PAD_TARGET = -100


def paragraph_batches(id_sequences, batch_size, eos_id, device, shuffle=False, seed=0):
    """Pads paragraphs into mini-batches of `batch_size` paragraphs, one
    paragraph per row, each row starting from a fresh (zero) hidden state --
    the same "every paragraph on its own" rule the HMM and n-gram follow, in
    training and in scoring, so the RNN never sees context from a previous
    paragraph.

    A row's input is <eos> followed by the paragraph minus its last token,
    and its target is the whole paragraph (which ends in <eos>, added by
    data_utils.load_split_ids). So every token is predicted, the first one
    from <eos> alone -- the paragraph-start context, the RNN's counterpart of
    the HMM's startprob_ -- and each split is scored on exactly the same
    tokens as evaluate_hmm/evaluate_ngram. Rows are padded at the end: the
    LSTM reads left to right, so padding never affects the real positions,
    and pad targets are PAD_TARGET, which the loss ignores.

    Paragraphs are grouped by length to keep padding small. With shuffle,
    the grouping and the batch order are randomized (seeded), so each
    training epoch sees different batches."""
    order = list(range(len(id_sequences)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
        # sort by length only within large chunks, so batches still vary
        chunk = batch_size * 50
        order = [i for start in range(0, len(order), chunk)
                 for i in sorted(order[start:start + chunk], key=lambda i: len(id_sequences[i]))]
    else:
        order.sort(key=lambda i: len(id_sequences[i]))

    batches = []
    for start in range(0, len(order), batch_size):
        seqs = [id_sequences[i] for i in order[start:start + batch_size]]
        max_len = max(len(seq) for seq in seqs)
        x = torch.full((len(seqs), max_len), eos_id, dtype=torch.long)
        y = torch.full((len(seqs), max_len), PAD_TARGET, dtype=torch.long)
        for row, seq in enumerate(seqs):
            seq = torch.tensor(seq, dtype=torch.long)
            x[row, 1:len(seq)] = seq[:-1]
            y[row, :len(seq)] = seq
        batches.append((x.to(device), y.to(device)))
    if shuffle:
        rng.shuffle(batches)
    return batches


def train_epoch(model, batches, optimizer, criterion, clip=0.25):
    """batches: from paragraph_batches(..., shuffle=True). Returns the
    token-weighted average loss, with dropout ON and the weights changing
    during the epoch -- not comparable to val; see evaluation.rnn_nll."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for x, y in batches:
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        n = (y != PAD_TARGET).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens


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
    """prefix_ids should start with <eos>, the paragraph-start input the
    model is trained with (see paragraph_batches); same for mc_dropout_predict."""
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
