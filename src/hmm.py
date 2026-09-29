import numpy as np
from data_utils import top_k_words


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
        self.train_nll_history = []
        self.val_history = []


class CategoricalHMM:
    """A from-scratch categorical (discrete-emission) HMM trained with Baum-Welch.

    Exposes the same attribute names hmmlearn uses (startprob_, transmat_,
    emissionprob_, monitor_) so callers don't need to know which
    implementation trained the model.
    """

    def __init__(self, n_components, n_features, n_iter=200, tol=1e-4, random_state=42, smoothing=1e-3,
                 tol_patience=3):
        self.n_components = n_components # number of hidden states
        self.n_features = n_features # vocab size
        self.n_iter = n_iter # number of EM iterations
        self.tol = tol # relative log-likelihood change threshold for convergence
        self.tol_patience = tol_patience # consecutive below-tol iterations needed to declare convergence
        self.random_state = random_state
        self.smoothing = smoothing # additive smoothing for start/trans/emission probabilities
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
        T = len(obs) # sequence length
        N = self.n_components # number of hidden states
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

        # xi_t(i,j) = alpha_t(i) A(i,j) B(j,o_{t+1}) beta_{t+1}(j) / c_{t+1};
        # A(i,j) factors out of the sum over t, leaving one matrix product
        w = B[:, obs[1:]].T * beta[1:] / scale[1:, None]
        xi_sum = A * (alpha[:-1].T @ w)

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
        below_tol = 0

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
                # train scored the same way as val (same post-M-step params, same
                # function), unlike total_ll, which used the pre-M-step params
                self.monitor_.train_nll_history.append(sequence_nll(self.startprob_, self.transmat_, self.emissionprob_, sequences))
                self.monitor_.val_history.append(sequence_nll(self.startprob_, self.transmat_, self.emissionprob_, val_sequences))
            # relative tolerance, as in VBHMM: total_ll is a sum over the whole
            # corpus, so an absolute threshold would depend on corpus size.
            # Requires tol_patience consecutive small steps, so one noisy dip
            # below tol doesn't end training while the LL is still climbing
            if prev_ll is not None and abs(total_ll - prev_ll) < self.tol * abs(prev_ll):
                below_tol += 1
                if below_tol >= self.tol_patience:
                    self.monitor_.converged = True
                    break
            else:
                below_tol = 0
            prev_ll = total_ll

        return self


def train_hmm(id_sequences, n_states, vocab_size, n_iter=200, seed=42, val_sequences=None):
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


def suggest_next_words(model, prefix_ids, id2word, k=5, exclude_ids=()):
    return top_k_words(next_word_distribution(model, prefix_ids), id2word, k, exclude_ids)
