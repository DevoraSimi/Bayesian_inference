import numpy as np
import torch

from rnn import detach_hidden, get_batch


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

    def update_batch(self, correct_per_k, word_mask):
        """RNN path: correct_per_k[k] is a bool tensor, True where the true
        token appeared in that position's top-k."""
        for k in self.k_list:
            hit_k = correct_per_k[k]
            self.hits_all[k] += hit_k.sum().item()
            self.hits_words[k] += (hit_k & word_mask).sum().item()
        self.n_predictions += word_mask.numel()
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
    first, using whatever short/empty context is available -- the n-gram
    equivalent of the HMM's startprob_ contribution). Top-k accuracy only
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


@torch.no_grad()
def evaluate_rnn(model, data, criterion, bptt, k_list=(1, 5, 10), punct_ids=frozenset()):
    """Reports top-k accuracy both over all tokens and restricted to
    positions where the true next token isn't in punct_ids (real words only)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    acc = TopKAccumulator(k_list)
    punct_tensor = (
        torch.tensor(sorted(punct_ids), dtype=torch.long, device=data.device)
        if punct_ids else None
    )

    hidden = None
    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)
        logits, hidden = model(x, hidden)
        hidden = detach_hidden(hidden)
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
        correct_per_k = {k: correct[..., :k].any(-1) for k in k_list}
        acc.update_batch(correct_per_k, word_mask)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return acc.finalize(perplexity)


@torch.no_grad()
def rnn_nll(model, data, criterion, bptt):
    """Average per-token NLL (nats) in eval mode (dropout off, fixed weights)
    -- the same scoring as evaluate_rnn's perplexity (log of it), minus the
    top-k work. Used for per-epoch train NLL, since train_epoch's returned
    loss is averaged with dropout ON while the weights are still changing,
    so it isn't comparable to val."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden = None
    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)
        logits, hidden = model(x, hidden)
        hidden = detach_hidden(hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / total_tokens


@torch.no_grad()
def evaluate_rnn_mc_dropout(model, data, bptt, k_list=(1, 5, 10), punct_ids=frozenset(), n_samples=20):
    """Like evaluate_rnn, but keeps dropout ACTIVE (model.train()) and
    averages n_samples stochastic forward passes' softmax probabilities per
    batch before scoring -- MC Dropout as a cheap approximate-Bayesian
    ensemble, testing whether averaging over the implicit weight-posterior
    samples changes perplexity/accuracy, not just prediction uncertainty
    (that's rnn.mc_dropout_predict, used for demo-time suggestions).

    No criterion/loss module needed here: perplexity is computed directly
    from log(mean predictive probability at the true token), which is the
    mathematically correct way to score a Bayesian model average -- NOT
    the same as averaging each sample's log-probability."""
    was_training = model.training
    model.train()

    total_log_prob = 0.0
    total_tokens = 0
    acc = TopKAccumulator(k_list)
    punct_tensor = (
        torch.tensor(sorted(punct_ids), dtype=torch.long, device=data.device)
        if punct_ids else None
    )

    # one carried state per MC sample, so each sample's context comes only
    # from its own earlier passes
    hiddens = [None] * n_samples
    for i in range(0, data.size(1) - 1, bptt):
        x, y = get_batch(data, i, bptt)

        prob_sum = None
        for s in range(n_samples):
            logits, hidden = model(x, hiddens[s])
            hiddens[s] = detach_hidden(hidden)
            probs = torch.softmax(logits, dim=-1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
        mean_probs = prob_sum / n_samples

        true_probs = mean_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        total_log_prob += torch.log(true_probs).sum().item()
        total_tokens += y.numel()

        if punct_tensor is not None:
            word_mask = ~torch.isin(y, punct_tensor)
        else:
            word_mask = torch.ones_like(y, dtype=torch.bool)

        topk = mean_probs.topk(max(k_list), dim=-1).indices
        correct = (topk == y.unsqueeze(-1))
        correct_per_k = {k: correct[..., :k].any(-1) for k in k_list}
        acc.update_batch(correct_per_k, word_mask)

    model.train(was_training)

    perplexity = float(np.exp(-total_log_prob / total_tokens))
    return acc.finalize(perplexity)
