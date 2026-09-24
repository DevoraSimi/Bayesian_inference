import numpy as np


def sequence_nll(startprob, transmat, emissionprob, sequences):
    """Average per-token negative log-likelihood (nats) of `sequences` under
    the given (startprob, transmat, emissionprob) -- a forward-pass-only,
    cheap computation (no backward pass/gamma/xi, since we're not training
    from this, just scoring). Standalone (not a method) so both
    CategoricalHMM and vb_hmm.VariationalBayesHMM can share it -- both
    expose startprob_/transmat_/emissionprob_ via the same interface."""
    total_ll = 0.0
    total_tokens = 0
    for obs in sequences:
        obs = np.asarray(obs)
        alpha = startprob * emissionprob[:, obs[0]]
        norm = alpha.sum()
        total_ll += np.log(norm)
        alpha = alpha / norm
        for o in obs[1:]:
            alpha = (alpha @ transmat) * emissionprob[:, o]
            norm = alpha.sum()
            total_ll += np.log(norm)
            alpha = alpha / norm
        total_tokens += len(obs)
    return -total_ll / total_tokens


class ConvergenceMonitor:
    def __init__(self):
        self.converged = False
        self.iter = 0
        self.history = []
        self.val_history = []


class CategoricalHMM:
    """A from-scratch categorical (discrete-emission) HMM trained with Baum-Welch.

    Exposes the same attribute names hmmlearn uses (startprob_, transmat_,
    emissionprob_, monitor_) so callers don't need to know which
    implementation trained the model.
    """

    def __init__(self, n_components, n_features, n_iter=30, tol=1e-2, random_state=42, smoothing=1e-3):
        self.n_components = n_components
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.smoothing = smoothing
        self.monitor_ = ConvergenceMonitor()

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        N, V = self.n_components, self.n_features

        startprob = rng.random(N)
        self.startprob_ = startprob / startprob.sum()

        transmat = rng.random((N, N))
        self.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)

        emission = rng.random((N, V))
        self.emissionprob_ = emission / emission.sum(axis=1, keepdims=True)

    def _forward_backward(self, obs):
        """Scaled forward-backward for one sequence. Returns per-position state
        posteriors (gamma), summed pairwise transition posteriors (xi_sum),
        and the sequence's log-likelihood."""
        T = len(obs)
        N = self.n_components
        A = self.transmat_
        B = self.emissionprob_

        alpha = np.empty((T, N))
        scale = np.empty(T)

        alpha[0] = self.startprob_ * B[:, obs[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]

        beta = np.empty((T, N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / scale[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((N, N))
        for t in range(T - 1):
            xi_t = (alpha[t][:, None] * A) * (B[:, obs[t + 1]] * beta[t + 1])[None, :] / scale[t + 1]
            xi_sum += xi_t

        log_likelihood = np.log(scale).sum()
        return gamma, xi_sum, log_likelihood

    def fit(self, sequences, val_sequences=None):
        """val_sequences, if given, is scored (average NLL/token, forward-pass
        only -- no backward pass/gamma/xi, since validation only needs a
        likelihood, not the training statistics) after every EM iteration and
        appended to monitor_.val_history, for overfitting monitoring. This is
        never used to influence training itself, only recorded."""
        self._init_params()
        N, V = self.n_components, self.n_features
        prev_ll = None

        for iteration in range(1, self.n_iter + 1):
            start_num = np.zeros(N)
            trans_num = np.zeros((N, N))
            trans_denom = np.zeros(N)
            emit_num = np.zeros((N, V))
            emit_denom = np.zeros(N)
            total_ll = 0.0

            for obs in sequences:
                obs = np.asarray(obs)
                gamma, xi_sum, log_likelihood = self._forward_backward(obs)
                total_ll += log_likelihood

                start_num += gamma[0]
                trans_num += xi_sum
                trans_denom += gamma[:-1].sum(axis=0)
                np.add.at(emit_num, (slice(None), obs), gamma.T)
                emit_denom += gamma.sum(axis=0)

            eps = self.smoothing
            self.startprob_ = (start_num + eps) / (start_num.sum() + N * eps)
            self.transmat_ = (trans_num + eps) / (trans_denom[:, None] + N * eps)
            self.emissionprob_ = (emit_num + eps) / (emit_denom[:, None] + V * eps)

            self.monitor_.iter = iteration
            self.monitor_.history.append(total_ll)
            if val_sequences is not None:
                self.monitor_.val_history.append(sequence_nll(self.startprob_, self.transmat_, self.emissionprob_, val_sequences))
            if prev_ll is not None and abs(total_ll - prev_ll) < self.tol:
                self.monitor_.converged = True
                break
            prev_ll = total_ll

        return self


def train_hmm(id_sequences, n_states, vocab_size, n_iter=50, seed=42, val_sequences=None):
    model = CategoricalHMM(
        n_components=n_states,
        n_features=vocab_size,
        n_iter=n_iter,
        random_state=seed,
    )
    model.fit(id_sequences, val_sequences=val_sequences)
    return model


def filtered_state_distribution(model, obs_ids):
    """P(state_t | obs_1..t) via the forward algorithm, normalized at each step."""
    A = model.transmat_
    B = model.emissionprob_
    alpha = model.startprob_ * B[:, obs_ids[0]]
    alpha = alpha / alpha.sum()
    for o in obs_ids[1:]:
        alpha = (alpha @ A) * B[:, o]
        alpha = alpha / alpha.sum()
    return alpha


def next_word_distribution(model, obs_ids):
    alpha = filtered_state_distribution(model, obs_ids)
    next_state_dist = alpha @ model.transmat_
    return next_state_dist @ model.emissionprob_


def suggest_next_words(model, prefix_ids, id2word, k=5):
    dist = next_word_distribution(model, prefix_ids)
    top_idx = np.argsort(-dist)[:k]
    return [(id2word[i], float(dist[i])) for i in top_idx]
