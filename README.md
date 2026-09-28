# Automatic Text Completion — Baum-Welch, Bayesian HMM, N-gram, and RNN

A smartphone-keyboard-style next-word prediction system, built to compare several
language-modeling approaches on the same corpus: a classical Dirichlet-smoothed
n-gram model, an HMM trained with Baum-Welch (MAP-EM), a fully Bayesian HMM
(Variational Bayes EM), and an LSTM RNN (with an optional MC-Dropout uncertainty
mode). Given a text prefix, each model suggests likely next words; all are trained
and evaluated on the same corpus and vocabulary for a fair comparison.

## Corpus

The complete Arthur Conan Doyle Sherlock Holmes canon (public domain, via Project
Gutenberg): 4 novels and 5 short-story anthologies, 9 files in `data/raw/`.
Anthologies are automatically split into their individual stories (not just left as
one blob per book) so that train/val/test splits happen at the *story* level —
whole stories are held out, never individual paragraphs — avoiding the leakage that
paragraph-level shuffling would cause.

## Preprocessing

Follows WikiText (Merity et al., 2016), which was tokenized with the Moses tokenizer:

- **One paragraph per sequence** (blank-line separated), no sentence splitting;
  `<eos>` marks the end of a paragraph. Headings, chapter titles, contents
  listings and "By A. Conan Doyle" lines are dropped (unlike WikiText, which
  keeps section titles as `= Title =` lines: in these books they are layout and
  front matter, not content).
- **Original case kept** (`Holmes`, `The` and `the` are distinct tokens).
- **Every punctuation mark is a token**; curly/straight quote variants are unified,
  `--` becomes `—`, `…` becomes `...`.
- **Moses/WikiText splitting**: `don't` → `don 't`, `Holmes's` → `Holmes 's`,
  `well-known` → `well @-@ known`, `1,000` → `1 @,@ 000`, `3.5` → `3 @.@ 5`;
  title abbreviations stay whole (`Mr.`, `Dr.`, `St.`).
- **Numbers are kept** as tokens (`1889`, `221B`).
- **Vocabulary** = every token seen at least `--min-count` times in the training
  split (default 2; WikiText uses 3, but this corpus is ~4x smaller); everything
  else becomes `<unk>`.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Build the corpus: clean Gutenberg boilerplate, split into stories and
#    paragraphs, tokenize, build vocab, split train/val/test at the story level.
#    Changing preprocessing changes the vocab: retrain every model afterwards,
#    and first move old outputs aside (experiment.py MERGES into an existing
#    results/comparison.csv, and old checkpoints would sit next to new ones):
#      mv checkpoints checkpoints_old; mv results results_old
python3 src/preprocess.py            # --min-count 3 for the WikiText default

# 2. Train individual models (optional -- experiment.py below does all of this
#    as a parameter sweep, but these are useful for one-off runs):
python3 src/train_ngram.py --order 3
python3 src/train_hmm.py --n-states 16
python3 src/train_vb_hmm.py --n-states 16
python3 src/train_rnn.py --hidden-size 256 --epochs 10

# 3. (Optional) search for good RNN hyperparameters before the full sweep --
#    only embed_size/dropout/lr are searched; hidden_size/num_layers are fixed
#    so the result stays valid when the sweep below varies them.
python3 src/tune_rnn.py --hidden-size 128 --num-layers 1 --n-trials 20

# 4. Run the full comparison: sweeps every model across its own parameters
#    (n-gram order & alpha, HMM/VB-HMM states, RNN hidden/layers/dropout, and
#    optionally MC-Dropout at various sample counts), saving a checkpoint,
#    metrics, and training-loss history for every configuration.
python3 src/experiment.py \
  --hmm-states 4 8 16 32 64 --vbhmm-states 4 8 16 32 64 \
  --ngram-orders 2 3 4 --ngram-alpha 10 30 100 \
  --rnn-hidden 64 128 256 --rnn-layers 1 2 --rnn-dropout 0.2 --rnn-epochs 15 \
  --mc-dropout-samples 5 20 50

# 5. Inspect results
python3 src/list_checkpoints.py --out results/checkpoint_manifest.csv
python3 src/plot_history.py

# 6. Try live next-word suggestions from any trained model(s). Without --text it
#    runs interactively: type a prefix, get suggestions, Ctrl-D to quit.
#    Input is case-sensitive, like the training data ("Mr. Holmes", not "mr. holmes").
#    <unk>/<eos> are never suggested; add --no-punct to suggest words only.
python3 src/predict_demo.py \
  --hmm-checkpoint checkpoints/hmm_16 --vbhmm-checkpoint checkpoints/vbhmm_16 \
  --ngram-checkpoint checkpoints/ngram_3_a30.0 --rnn-checkpoint checkpoints/rnn_256 \
  --mc-dropout 50 --text "Sherlock Holmes said that"
```

Every `train_*.py` script and `experiment.py` accept `--data-dir`, `--checkpoints-dir`,
and `--results-dir` if you want non-default locations; run any script with `--help`
for its full option list.

## Models

- **N-gram** (`ngram.py`) — order-*n* Markov model with hierarchical Dirichlet
  smoothing (MacKay & Peto, 1995): `P(w|ctx) = (count(ctx,w) + α·P(w|shorter ctx))/(count(ctx) + α)`,
  recursively trigram → bigram → unigram → uniform. Each level is the posterior
  mean of a Dirichlet prior centred on the next-shorter context, and by conjugacy
  exactly the Bayesian posterior predictive for one draw; rare contexts fall back
  smoothly to shorter ones. `order` and `α` (total prior mass) are both swept.
- **HMM** (`hmm.py`) — a from-scratch categorical HMM trained with Baum-Welch
  (MAP-EM): the forward-backward algorithm plus M-step point estimates with
  light Dirichlet smoothing, implemented without `hmmlearn`.
- **VB-HMM** (`vb_hmm.py`) — the same model made fully Bayesian: Dirichlet
  *posterior distributions* (not point estimates) over the initial-state,
  transition, and emission distributions, fit by mean-field Variational Bayes
  EM. States start from random data-scale pseudo-counts (a near-uniform start
  never breaks symmetry: all states stay identical). The emission prior
  defaults to `beta0 = 0.1` per word: 1.0 adds ~V pseudo-counts to every state
  and pulls them all toward uniform. Exposes the same
  interface as the MAP-EM HMM (via the posterior mean, which is the Bayesian
  posterior predictive by conjugacy), so it's a drop-in alternative wherever
  the MAP-EM HMM is used.
- **RNN** (`rnn.py`) — an LSTM language model (PyTorch), with an optional
  **MC-Dropout** mode (`mc_dropout_predict`, `evaluate_rnn_mc_dropout`) that
  keeps dropout active at inference time and averages multiple stochastic
  passes — an approximate Bayesian treatment of the network's weights (Gal &
  Ghahramani, 2016), giving both a prediction and an uncertainty estimate.

## Project layout

```
src/
  data_utils.py       WikiText-style tokenization, paragraph splitting, vocab, encoding
  preprocess.py        corpus loading, story-boundary splitting, CLI

  ngram.py              n-gram model
  hmm.py                 MAP-EM HMM (Baum-Welch)
  vb_hmm.py               Variational Bayes HMM
  rnn.py                   LSTM model + MC-Dropout

  evaluation.py             perplexity / top-k accuracy for every model,
                             shared TopKAccumulator, MC-Dropout evaluation
  checkpoints.py             save/load: model weights + config + metrics +
                             training history, one self-contained pair of
                             files per checkpoint

  train_ngram.py              standalone single-run training scripts
  train_hmm.py
  train_vb_hmm.py
  train_rnn.py                 (supports --resume for interrupted runs)
  tune_rnn.py                   Optuna hyperparameter search for the RNN

  experiment.py                  full parameter sweep across all models
  list_checkpoints.py             tabular summary of every saved checkpoint
  plot_history.py                  training-loss curves from saved history
  predict_demo.py                   live next-word suggestions from any
                                     trained model(s), with optional MC-Dropout

data/
  raw/                the 9 Gutenberg source files
  processed/           vocab.json, train/val/test.txt, split_metadata.json

checkpoints/    <name>.pkl or .pt (weights) + <name>.json (config, metrics,
                training history, timestamp) per trained model
results/        comparison.csv, per-model plots, checkpoint manifest
```

## Metrics

Every model is scored with:
- **Perplexity** (lower is better) — `exp(average negative log-likelihood per token)`.
- **Top-k accuracy** (k = 1, 5, 10) — was the true next word in the model's top-k
  predictions — reported both over *all* tokens and restricted to real words only
  (excluding punctuation tokens), since punctuation is highly predictable and
  otherwise inflates the "all tokens" number.

`experiment.py`'s RNN sweep and `tune_rnn.py`'s final retrain both use early
stopping (tracking the best validation-perplexity epoch and restoring those
weights before the final evaluation) rather than training for a fixed epoch count
regardless of overfitting.
