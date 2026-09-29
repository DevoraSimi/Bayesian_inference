import numpy as np
from scipy.special import digamma as _digamma, gammaln as _log_gamma

from hmm import ConvergenceMonitor, sequence_nll


def _dirichlet_kl(post, prior):
    """Sum over rows of KL(Dir(post_row) || Dir(prior, ..., prior)), for a
    symmetric scalar prior; post is a vector (one Dirichlet) or a matrix
    (one Dirichlet per row)."""
    post = np.atleast_2d(post)
    K = post.shape[1]
    post_sum = post.sum(axis=1, keepdims=True)
    kl = (_log_gamma(post_sum[:, 0]) - _log_gamma(post).sum(axis=1)
          - _log_gamma(K * prior) + K * _log_gamma(prior)
          + ((post - prior) * (_digamma(post) - _digamma(post_sum))).sum(axis=1))
    return kl.sum()


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

    def __init__(self, n_components, n_features, n_iter=200, tol=1e-4, random_state=42,
                 alpha0=1.0, beta0=0.1, pi0=1.0, tol_patience=3):
        self.n_components = n_components
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol  # relative ELBO change threshold for convergence
        self.tol_patience = tol_patience  # consecutive below-tol iterations needed to declare convergence
        self.random_state = random_state
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.pi0 = pi0
        self.monitor_ = ConvergenceMonitor()

    def _init_posteriors(self, n_tokens):
        """Prior + random pseudo-counts on the scale of the real data, as if
        each state had already been assigned ~n_tokens/N tokens at random.
        The random part must be this large: a tiny perturbation of the prior
        (e.g. +U(0, 0.1) on 11k emission entries) leaves all states nearly
        identical, so they receive the same expected counts every E-step and
        never break symmetry -- the model degenerates to a single state."""
        rng = np.random.default_rng(self.random_state)
        N, V = self.n_components, self.n_features
        per_state = n_tokens / N
        self.pi_post = self.pi0 + rng.dirichlet(np.ones(N)) * N
        self.A_post = self.alpha0 + rng.dirichlet(np.ones(N), size=N) * per_state
        self.B_post = self.beta0 + rng.dirichlet(np.ones(V), size=N) * per_state

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
        bound (fit() subtracts the KL(posterior||prior) terms for pi/A/B
        to get the full ELBO; the data term alone is not monotone)."""
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

        # xi_t(i,j) = alpha_t(i) A(i,j) B(j,o_{t+1}) beta_{t+1}(j) / c_{t+1};
        # A(i,j) factors out of the sum over t, leaving one matrix product
        w = B_tilde[:, obs[1:]].T * beta[1:] / scale[1:, None]
        xi_sum = A_tilde * (alpha[:-1].T @ w)

        data_term = np.log(scale).sum()
        return gamma, xi_sum, data_term

    def fit(self, sequences, val_sequences=None):
        """val_sequences, if given, is scored (via sequence_nll, using the
        posterior-mean startprob_/transmat_/emissionprob_ after each
        iteration's M-step) and appended to monitor_.val_history, for
        overfitting monitoring -- never used to influence training itself."""
        self._init_posteriors(sum(len(obs) for obs in sequences))
        N, V = self.n_components, self.n_features
        prev_bound = None
        below_tol = 0

        for iteration in range(1, self.n_iter + 1):
            pi_tilde, A_tilde, B_tilde = self._expected_probs()

            start_num = np.zeros(N)
            trans_num = np.zeros((N, N))
            emit_num = np.zeros((N, V))
            # ELBO = sum of data terms - KL(q(theta) || p(theta)), with the KL
            # taken for the posterior used in this E-step (i.e. before the M-step)
            total_bound = -(_dirichlet_kl(self.pi_post, self.pi0)
                            + _dirichlet_kl(self.A_post, self.alpha0)
                            + _dirichlet_kl(self.B_post, self.beta0))

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
                # train scored the same way as val (posterior-mean params, same
                # function), unlike total_bound, which is the ELBO
                self.monitor_.train_nll_history.append(sequence_nll(self.startprob_, self.transmat_, self.emissionprob_, sequences))
                self.monitor_.val_history.append(sequence_nll(self.startprob_, self.transmat_, self.emissionprob_, val_sequences))
            # relative tolerance: the ELBO is a sum over the whole corpus, so
            # an absolute threshold would depend on corpus size. Requires
            # tol_patience consecutive small steps, so one noisy dip below tol
            # doesn't end training while the ELBO is still climbing
            if prev_bound is not None and abs(total_bound - prev_bound) < self.tol * abs(prev_bound):
                below_tol += 1
                if below_tol >= self.tol_patience:
                    self.monitor_.converged = True
                    break
            else:
                below_tol = 0
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


def train_vb_hmm(id_sequences, n_states, vocab_size, n_iter=200, seed=42,
                  alpha0=1.0, beta0=0.1, pi0=1.0, val_sequences=None):
    model = VariationalBayesHMM(
        n_components=n_states, n_features=vocab_size, n_iter=n_iter, random_state=seed,
        alpha0=alpha0, beta0=beta0, pi0=pi0,
    )
    model.fit(id_sequences, val_sequences=val_sequences)
    return model
