from collections import Counter, defaultdict

import numpy as np


class NgramModel:
    """An order-n Markov / n-gram language model with Dirichlet (Bayesian
    additive) smoothing:

        P(w | context) = (count(context, w) + alpha) / (count(context) + alpha * V)

    This is the posterior MEAN of a Dirichlet(alpha) prior placed over the
    categorical distribution of "next word given this context" -- and by
    Dirichlet-multinomial conjugacy, the posterior mean is also exactly the
    Bayesian posterior predictive probability for a single next draw. So
    this smoothing isn't just a heuristic: it's the Bayes estimator under a
    symmetric Dirichlet prior, the same idea already used for the HMM's
    emission/transition smoothing (hmm.py), just applied to raw n-gram
    counts instead of latent-state statistics.

    order=2 is a bigram model (1 word of context), order=3 a trigram
    (2 words of context), etc. Contexts shorter than order-1 (at the start
    of a sequence) simply use whatever context is available.
    """

    def __init__(self, order, vocab_size, alpha=1.0):
        self.order = order
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.context_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)

    def fit(self, id_sequences):
        ctx_len = self.order - 1
        for seq in id_sequences:
            for i, word in enumerate(seq):
                context = tuple(seq[max(0, i - ctx_len):i])
                self.context_counts[context][word] += 1
                self.context_totals[context] += 1
        return self

    def next_word_distribution(self, context):
        """context: a sequence of word ids (only the last order-1 are used)."""
        ctx_len = self.order - 1
        context = tuple(context[-ctx_len:]) if ctx_len > 0 else ()

        V = self.vocab_size
        alpha = self.alpha
        total = self.context_totals.get(context, 0)
        denom = total + alpha * V

        dist = np.full(V, alpha / denom)
        counts = self.context_counts.get(context)
        if counts:
            for word_id, c in counts.items():
                dist[word_id] = (c + alpha) / denom
        return dist


def train_ngram(id_sequences, order, vocab_size, alpha=1.0):
    return NgramModel(order, vocab_size, alpha).fit(id_sequences)


def suggest_next_words(model, prefix_ids, id2word, k=5):
    dist = model.next_word_distribution(prefix_ids)
    top_idx = np.argsort(-dist)[:k]
    return [(id2word[i], float(dist[i])) for i in top_idx]
