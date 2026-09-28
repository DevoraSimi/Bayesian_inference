from collections import Counter, defaultdict

import numpy as np

from data_utils import top_k_words


class NgramModel:
    """An order-n language model with hierarchical Dirichlet smoothing
    (MacKay & Peto, 1995, "A hierarchical Dirichlet language model"):

        P_k(w | last k words) = (count(ctx_k, w) + alpha * P_{k-1}(w | last k-1 words))
                                / (count(ctx_k) + alpha)

    down to P_{-1}(w) = 1/V (uniform). Each context's next-word distribution
    gets a Dirichlet prior with concentration `alpha` centred on the
    next-shorter context's distribution (trigram -> bigram -> unigram ->
    uniform), 

    Centring the prior on the shorter context rather than on uniform is what
    makes higher orders usable: for a rare or unseen context (count ~ 0) the
    estimate falls back smoothly to the bigram/unigram estimate instead of to
    a near-uniform guess over the whole vocabulary. With a flat prior, a
    trigram scores worse than a bigram on this corpus, since most test
    contexts are rare.

    `alpha` is the prior's total concentration (pseudo-count mass), not a
    per-word pseudo-count: larger alpha trusts the shorter context more.

    order=2 is a bigram model (1 word of context), order=3 a trigram
    (2 words of context), etc. Near the start of a sequence, the longest
    available context is used.
    """

    def __init__(self, order, vocab_size, alpha=1.0):
        self.order = order
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.context_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)

    def fit(self, id_sequences):
        # counts for every context length 0..order-1 at every position, since
        # each level of the hierarchy is estimated from its own counts
        ctx_len = self.order - 1
        for seq in id_sequences:
            for i, word in enumerate(seq):
                for k in range(min(i, ctx_len) + 1):
                    context = tuple(seq[i - k:i])
                    self.context_counts[context][word] += 1
                    self.context_totals[context] += 1
        return self

    def next_word_distribution(self, context):
        """context: a sequence of word ids (only the last order-1 are used)."""
        ctx_len = min(self.order - 1, len(context))
        context = tuple(context[len(context) - ctx_len:])

        alpha = self.alpha
        dist = np.full(self.vocab_size, 1.0 / self.vocab_size)
        for k in range(ctx_len + 1):  # unigram first, then ever longer contexts
            ctx = context[ctx_len - k:]
            total = self.context_totals.get(ctx, 0)
            if total == 0:
                # never seen: posterior = prior, and every longer context
                # containing this one is unseen too
                break
            dist *= alpha / (total + alpha)
            for word_id, c in self.context_counts[ctx].items():
                dist[word_id] += c / (total + alpha)
        return dist


def train_ngram(id_sequences, order, vocab_size, alpha=1.0):
    return NgramModel(order, vocab_size, alpha).fit(id_sequences)


def suggest_next_words(model, prefix_ids, id2word, k=5, exclude_ids=()):
    dist = model.next_word_distribution(prefix_ids)
    return top_k_words(dist, id2word, k, exclude_ids)
