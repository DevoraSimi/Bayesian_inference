import torch
import torch.nn as nn

from data_utils import encode_sentence, tokenize


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
def suggest_next_words(model, word2id, id2word, prefix_text, device, k=5):
    tokens = tokenize(prefix_text)
    ids = encode_sentence(tokens, word2id)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_idx = probs.topk(k)
    return [(id2word[i], float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]


@torch.no_grad()
def mc_dropout_predict(model, word2id, id2word, prefix_text, device, k=5, n_samples=50):
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
    tokens = tokenize(prefix_text)
    ids = encode_sentence(tokens, word2id)
    x = torch.tensor([ids], dtype=torch.long, device=device)

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

    top_probs, top_idx = mean_probs.topk(k)
    return [
        (id2word[i], float(p), float(std_probs[i]))
        for i, p in zip(top_idx.tolist(), top_probs.tolist())
    ]
