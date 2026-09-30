import numpy as np
import torch

from rnn import PAD_TARGET


class TopKAccumulator:
    """Accumulates top-k hit counts (all tokens + words-only) from either
    scalar updates (one prediction at a time, for HMM) or batched tensor
    updates (for RNN), then reduces to the shared metrics shape."""

    def __init__(self, k_list):
        self.k_list = k_list
        self.hits_all = {k: 0 for k in k_list}
        self.hits_words = {k: 0 for k in k_list}
        self.n_predictions = 0
        self.n_word_predictions = 0

    def update_one(self, ranked_ids, true_id, is_word):
        """HMM path: ranked_ids is a 1D array of vocab ids, best first."""
        for k in self.k_list:
            if true_id in ranked_ids[:k]:
                self.hits_all[k] += 1
                if is_word:
                    self.hits_words[k] += 1
        self.n_predictions += 1
        self.n_word_predictions += is_word

    def update_batch(self, correct_per_k, scored_mask, word_mask):
        """RNN path: correct_per_k[k] is a bool tensor, True where the true
        token appeared in that position's top-k; only positions in
        scored_mask count (word_mask is the words-only subset of it)."""
        for k in self.k_list:
            hit_k = correct_per_k[k]
            self.hits_all[k] += (hit_k & scored_mask).sum().item()
            self.hits_words[k] += (hit_k & word_mask).sum().item()
        self.n_predictions += scored_mask.sum().item()
        self.n_word_predictions += word_mask.sum().item()

    def finalize(self, perplexity):
        topk_acc = {k: self.hits_all[k] / self.n_predictions for k in self.k_list}
        topk_acc_words = {k: self.hits_words[k] / self.n_word_predictions for k in self.k_list}
        return {"perplexity": perplexity, "topk_acc": topk_acc, "topk_acc_words": topk_acc_words}


def print_eval(label, metrics):
    print(
        f"{label}: perplexity={metrics['perplexity']:.2f} "
        f"top-5 acc (all)={metrics['topk_acc'][5]:.3%} "
        f"top-5 acc (words only)={metrics['topk_acc_words'][5]:.3%}"
    )


def metrics_to_row(metrics):
    """Flattens a metrics dict into the rounded scalar fields used in
    experiment.py's sweep result rows / comparison.csv."""
    return {
        "perplexity": round(metrics["perplexity"], 2),
        "top1_acc": round(metrics["topk_acc"][1], 4),
        "top5_acc": round(metrics["topk_acc"][5], 4),
        "top1_acc_words": round(metrics["topk_acc_words"][1], 4),
        "top5_acc_words": round(metrics["topk_acc_words"][5], 4),
    }


def evaluate_hmm(model, id_sequences, k_list=(1, 5, 10), punct_ids=frozenset()):
    """Single incremental forward pass per sequence: at each step predict the
    next word from the prefix seen so far, then fold in the true observation.
    Reports top-k accuracy both over all tokens and restricted to positions
    where the true next token isn't in punct_ids (real words only)."""
    A = model.transmat_
    B = model.emissionprob_
    total_log_prob = 0.0
    total_tokens = 0
    acc = TopKAccumulator(k_list)

    for seq in id_sequences:
        alpha = model.startprob_ * B[:, seq[0]]
        norm = alpha.sum()
        total_log_prob += np.log(norm)
        total_tokens += 1
        alpha = alpha / norm

        for o in seq[1:]:
            next_state_dist = alpha @ A
            word_dist = next_state_dist @ B

            ranked = np.argsort(-word_dist)
            acc.update_one(ranked, o, o not in punct_ids)

            alpha = next_state_dist * B[:, o]
            norm = alpha.sum()
            total_log_prob += np.log(norm)
            total_tokens += 1
            alpha = alpha / norm

    perplexity = float(np.exp(-total_log_prob / total_tokens))
    return acc.finalize(perplexity)


def evaluate_ngram(model, id_sequences, k_list=(1, 5, 10), punct_ids=frozenset()):
    """Perplexity is computed over every token (including each sequence's
    first, predicted from the model's paragraph-start context, which
    NgramModel adds itself -- the n-gram equivalent of the HMM's startprob_
    contribution). Top-k accuracy only
    counts tokens with i > 0, matching evaluate_hmm/evaluate_rnn's
    convention of not scoring "predict the very first word from nothing"."""
    total_log_prob = 0.0
    total_tokens = 0
    acc = TopKAccumulator(k_list)

    for seq in id_sequences:
        for i, true_id in enumerate(seq):
            dist = model.next_word_distribution(seq[:i])
            total_log_prob += np.log(dist[true_id])
            total_tokens += 1

            if i > 0:
                ranked = np.argsort(-dist)
                acc.update_one(ranked, true_id, true_id not in punct_ids)

    perplexity = float(np.exp(-total_log_prob / total_tokens))
    return acc.finalize(perplexity)


def _update_topk_batch(acc, scores, y, punct_ids):
    """Top-k update for one padded batch; scores are logits or probabilities
    (same ranking). Skips padding and each paragraph's first token (column
    0), matching evaluate_hmm/evaluate_ngram, which don't score "predict the
    very first word from nothing" for top-k either."""
    scored = y != PAD_TARGET
    scored[:, 0] = False
    word_mask = scored
    if punct_ids:
        punct_tensor = torch.tensor(sorted(punct_ids), dtype=torch.long, device=y.device)
        word_mask = scored & ~torch.isin(y, punct_tensor)
    topk = scores.topk(max(acc.k_list), dim=-1).indices
    correct = (topk == y.unsqueeze(-1))
    correct_per_k = {k: correct[..., :k].any(-1) for k in acc.k_list}
    acc.update_batch(correct_per_k, scored, word_mask)


@torch.no_grad()
def evaluate_rnn(model, batches, criterion, k_list=(1, 5, 10), punct_ids=frozenset()):
    """batches: from rnn.paragraph_batches -- each paragraph scored on its
    own from a fresh hidden state, like evaluate_hmm/evaluate_ngram, so all
    models are scored on the same tokens under the same rule. criterion
    must ignore PAD_TARGET (nn.CrossEntropyLoss() does by default).
    Reports top-k accuracy both over all tokens and restricted to
    positions where the true next token isn't in punct_ids (real words only)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    acc = TopKAccumulator(k_list)

    for x, y in batches:
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n = (y != PAD_TARGET).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
        _update_topk_batch(acc, logits, y, punct_ids)

    perplexity = float(np.exp(total_loss / total_tokens))
    return acc.finalize(perplexity)


@torch.no_grad()
def rnn_nll(model, batches, criterion):
    """Average per-token NLL (nats) in eval mode (dropout off, fixed weights)
    -- the same scoring as evaluate_rnn's perplexity (log of it), minus the
    top-k work. Used for per-epoch train NLL, since train_epoch's returned
    loss is averaged with dropout ON while the weights are still changing,
    so it isn't comparable to val."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in batches:
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n = (y != PAD_TARGET).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens
