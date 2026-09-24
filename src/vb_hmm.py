import numpy as np

from hmm import ConvergenceMonitor, sequence_nll


def _digamma(x):
    """Numerically stable digamma (psi) function via the standard shift-up
    recurrence (psi(x) = psi(x+1) - 1/x) plus an asymptotic expansion for
    x >= 6. Implemented locally rather than importing scipy.special.digamma,
    to avoid adding scipy as a project dependency for a single function and
    to keep this model, like hmm.py, implemented from first principles."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    while np.any(x < 6):
        mask = x < 6
        result = np.where(mask, result - 1.0 / x, result)
        x = np.where(mask, x + 1.0, x)
    inv = 1.0 / x
    inv2 = inv * inv
    result += np.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252)))
    return result


class VariationalBayesHMM:
    """A fully Bayesian HMM: Dirichlet priors on the initial-state
    distribution, each row of the transition matrix, and each row of the
    emission matrix, fit by mean-field Variational Bayes EM (VB-EM) --
    unlike hmm.CategoricalHMM, which fits point-estimate (MAP) parameters,
    this model maintains DIRICHLET POSTERIOR DISTRIBUTIONS over pi/A/B.

    self.pi_post, self.A_post, self.B_post hold Dirichlet concentration
    parameters (prior + expected sufficient statistics), not probabilities.
    The E-step (forward-backward) plugs in the expected log-probabilities
    under the current posterior, exp(E[log theta]) = exp(digamma(alpha) -
    digamma(sum(alpha))), rather than point-estimate probabilities -- the
    standard mean-field trick for conjugate-exponential models (Beal, 2003,
    "Variational Algorithms for Approximate Bayesian Inference", Ch. 3;
    Bishop, "Pattern Recognition and Machine Learning", Ch. 10). This
    quantity is always <= the naive ratio alpha_i / sum(alpha), a "Bayesian
    shrinkage" effect from Jensen's inequality that makes VB-EM naturally
    more conservative about rare states/observations than plain MAP-EM.

    startprob_/transmat_/emissionprob_ expose the POSTERIOR MEAN of each
    Dirichlet (a simple ratio, unlike the exp(E[log .]) used internally
    during fitting). By Dirichlet-multinomial conjugacy, the posterior mean
    is exactly the Bayesian posterior predictive distribution for a single
    next draw -- so a fitted model is drop-in compatible with every
    function written against hmm.CategoricalHMM's interface (filtered_state_
    distribution, next_word_distribution, suggest_next_words, evaluate_hmm).
    """

    def __init__(self, n_components, n_features, n_iter=30, tol=1e-4, random_state=42,
                 alpha0=1.0, beta0=1.0, pi0=1.0):
        self.n_components = n_components
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.pi0 = pi0
        self.monitor_ = ConvergenceMonitor()

    def _init_posteriors(self):
        rng = np.random.default_rng(self.random_state)
        N, V = self.n_components, self.n_features
        self.pi_post = np.full(N, self.pi0) + rng.random(N) * 0.1
        self.A_post = np.full((N, N), self.alpha0) + rng.random((N, N)) * 0.1
        self.B_post = np.full((N, V), self.beta0) + rng.random((N, V)) * 0.1

    def _expected_probs(self):
        pi_tilde = np.exp(_digamma(self.pi_post) - _digamma(self.pi_post.sum()))
        A_tilde = np.exp(_digamma(self.A_post) - _digamma(self.A_post.sum(axis=1, keepdims=True)))
        B_tilde = np.exp(_digamma(self.B_post) - _digamma(self.B_post.sum(axis=1, keepdims=True)))
        return pi_tilde, A_tilde, B_tilde

    def _forward_backward(self, obs, pi_tilde, A_tilde, B_tilde):
        """Same scaled forward-backward recursion as hmm.CategoricalHMM,
        but using exp(E[log .]) in place of point-estimate probabilities.
        Returns per-position state posteriors (gamma), summed pairwise
        transition posteriors (xi_sum), and the log of the forward-pass
        scaling factors -- the data-fit term of the variational lower
        bound, used here as a monotonically-improving convergence signal
        (the full ELBO would additionally subtract KL(posterior||prior)
        terms for pi/A/B; omitted for simplicity since it doesn't change
        which direction the algorithm is converging)."""
        T = len(obs)
        N = self.n_components

        alpha = np.empty((T, N))
        scale = np.empty(T)

        alpha[0] = pi_tilde * B_tilde[:, obs[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A_tilde) * B_tilde[:, obs[t]]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]

        beta = np.empty((T, N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A_tilde @ (B_tilde[:, obs[t + 1]] * beta[t + 1])) / scale[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((N, N))
        for t in range(T - 1):
            xi_t = (alpha[t][:, None] * A_tilde) * (B_tilde[:, obs[t + 1]] * beta[t + 1])[None, :] / scale[t + 1]
            xi_sum += xi_t

        data_term = np.log(scale).sum()
        return gamma, xi_sum, data_term

    def fit(self, sequences, val_sequences=None):
        """val_sequences, if given, is scored (via sequence_nll, using the
        posterior-mean startprob_/transmat_/emissionprob_ after each
        iteration's M-step) and appended to monitor_.val_history, for
        overfitting monitoring -- never used to influence training itself."""
        self._init_posteriors()
        N, V = self.n_components, self.n_features
        prev_bound = None

        for iteration in range(1, self.n_iter + 1):
            pi_tilde, A_tilde, B_tilde = self._expected_probs()

            start_num = np.zeros(N)
            trans_num = np.zeros((N, N))
            emit_num = np.zeros((N, V))
            total_bound = 0.0

            for obs in sequences:
                obs = np.asarray(obs)
                gamma, xi_sum, data_term = self._forward_backward(obs, pi_tilde, A_tilde, B_tilde)
                total_bound += data_term

                start_num += gamma[0]
                trans_num += xi_sum
                np.add.at(emit_num, (slice(None), obs), gamma.T)

            # M-step: Dirichlet posteriors = prior + expected sufficient
            # statistics (no separate "denominator" needed -- the row-sum of
            # each posterior recovers it automatically, since e.g.
            # trans_num[i, :].sum() == sum_t gamma_t(i) by construction)
            self.pi_post = self.pi0 + start_num
            self.A_post = self.alpha0 + trans_num
            self.B_post = self.beta0 + emit_num

            self.monitor_.iter = iteration
            self.monitor_.history.append(total_bound)
            if val_sequences is not None:
                self.monitor_.val_history.append(sequence_nll(self.startprob_, self.transmat_, self.emissionprob_, val_sequences))
            if prev_bound is not None and abs(total_bound - prev_bound) < self.tol:
                self.monitor_.converged = True
                break
            prev_bound = total_bound

        return self

    @property
    def startprob_(self):
        return self.pi_post / self.pi_post.sum()

    @property
    def transmat_(self):
        return self.A_post / self.A_post.sum(axis=1, keepdims=True)

    @property
    def emissionprob_(self):
        return self.B_post / self.B_post.sum(axis=1, keepdims=True)


def train_vb_hmm(id_sequences, n_states, vocab_size, n_iter=50, seed=42,
                  alpha0=1.0, beta0=1.0, pi0=1.0, val_sequences=None):
    model = VariationalBayesHMM(
        n_components=n_states, n_features=vocab_size, n_iter=n_iter, random_state=seed,
        alpha0=alpha0, beta0=beta0, pi0=pi0,
    )
    model.fit(id_sequences, val_sequences=val_sequences)
    return model
