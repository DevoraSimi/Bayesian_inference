import json
import pickle
import time
from pathlib import Path

import torch

from rnn import LSTMLanguageModel


def with_ext(path, ext):
    """Appends ext to path. Deliberately not Path.with_suffix(), which treats
    everything after the LAST dot as a replaceable suffix -- a checkpoint
    path built with e.g. a dropout value ("rnn_h4_l1_d0.1") would silently
    get truncated to "rnn_h4_l1_d0.json", and two different dropout values
    could even collide onto the same file."""
    return Path(str(path) + ext)


def save_pickle_checkpoint(model, config, metrics, path, history=None, val_history=None, train_nll_history=None):
    """Saves any plain-Python/NumPy model (MAP-EM HMM, VB-HMM, n-gram -- not
    the torch RNN, see save_rnn_checkpoint) to <path>.pkl via pickle, plus
    its run config + metrics + training/validation history to <path>.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(with_ext(path, ".pkl"), "wb") as f:
        pickle.dump(model, f)
    _save_metadata(path, config, metrics, history, val_history, train_nll_history)


def save_metrics_checkpoint(config, metrics, path):
    """Saves a metrics-only record (config + metrics, no model weights and no
    .pkl/.pt) to <path>.json -- for results like RNN-MCDropout that re-score
    an already-saved model's weights rather than training/saving a new one,
    so there's no separate model file for them, just a JSON of what was
    computed. Load it back the same way as any other checkpoint's metadata,
    via load_metadata(path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_metadata(path, config, metrics)


def load_pickle_checkpoint(path):
    path = Path(path)
    with open(with_ext(path, ".pkl"), "rb") as f:
        model = pickle.load(f)
    return model, load_metadata(path)


def save_rnn_checkpoint(model, optimizer, epoch, config, metrics, path, history=None):
    """Saves everything needed both for inference (model weights + config, via
    load_rnn_checkpoint) and for resuming training (+ optimizer state + epoch,
    via load_rnn_training_state), plus config/metrics/history to <path>.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": config,
    }, with_ext(path, ".pt"))
    _save_metadata(path, config, metrics, history)


def load_rnn_checkpoint(path, device):
    """Load for inference only. Returns (model, metadata)."""
    path = Path(path)
    ckpt = torch.load(with_ext(path, ".pt"), map_location=device)
    model = _build_rnn(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, load_metadata(path)


def load_rnn_training_state(path, device, lr):
    """Load to resume training. Returns (model, optimizer, start_epoch, history)."""
    path = Path(path)
    ckpt = torch.load(with_ext(path, ".pt"), map_location=device)
    model = _build_rnn(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    # history may run past the saved epoch (early-stopped runs keep the
    # post-best epochs for plotting); drop those so resumed training
    # doesn't duplicate epoch numbers
    history = [h for h in load_metadata(path).get("history", [])
               if h["epoch"] < start_epoch]
    return model, optimizer, start_epoch, history


def _build_rnn(cfg, device):
    return LSTMLanguageModel(
        cfg["vocab_size"], cfg["embed_size"], cfg["hidden_size"], cfg["num_layers"], cfg["dropout"],
    ).to(device)


def _save_metadata(path, config, metrics, history=None, val_history=None, train_nll_history=None):
    metadata = {
        "config": config,
        "metrics": metrics,
        "history": history or [],
        "train_nll_history": train_nll_history or [],
        "val_history": val_history or [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with_ext(path, ".json").write_text(json.dumps(metadata, indent=2))


def load_metadata(path):
    """Loads a checkpoint's <path>.json. JSON object keys are always strings,
    so topk_acc/topk_acc_words dicts (naturally keyed by int k) get their
    keys restored to ints here, once, so every caller can rely on int keys
    whether metrics came fresh from evaluate_*() or were reloaded from disk."""
    metadata = json.loads(with_ext(path, ".json").read_text())
    for split_metrics in metadata.get("metrics", {}).values():
        if not split_metrics:
            continue
        for key in ("topk_acc", "topk_acc_words"):
            if key in split_metrics:
                split_metrics[key] = {int(k): v for k, v in split_metrics[key].items()}
    return metadata
