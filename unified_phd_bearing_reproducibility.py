#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED PhD PROJECT: Reproducible Bearing Fault Diagnosis Research Suite
=======================================================================
Paper tracks (conceptual mapping, not claims of exact published reproduction):
  P1: Multiscale Feature Fusion by Entropy-Augmented KCFP for Rolling Bearing Fault Diagnosis
  P2: Physics-inspired deep learning network using dilated kronecker convolution
      for rotary machines under variable operating conditions
  P3: Frequency-aware transformer fusion for intelligent fault diagnosis of
      rolling bearings under variable operating conditions

Original source elements integrated here:
  - Entropy-augmented 1D CNN and multiscale convolution (P1's supplied code)
  - Median denoising / wavelet-Volterra / KCFP-LSTM baseline
  - Stagewise embedding visualizations and Random Forest / XGBoost comparisons

IMPORTANT REPRODUCIBILITY NOTES
  * P2 and P3 are NEW reference implementations motivated by their TITLES.
    No source implementations of those papers were supplied. Do not describe
    their numerical results as reproductions of the papers.
  * The supplied CNN code tested on its own training data; this implementation
    uses isolated, group-separated train/validation/test samples.
  * The supplied LSTM code standardized before splitting and augmented before
    splitting, creating data leakage. This implementation fixes both.
  * Each file is treated as one fault class, as in the provided nine-file maps.
    Meaningful speed/generalization studies REQUIRE real RPM/load metadata.
  * --demo generates SYNTHETIC signals for a software smoke test ONLY. Do not
    report its scores as measurements on real bearings.

Install: pip install numpy scipy matplotlib scikit-learn torch PyWavelets
Optional: pip install xgboost

Quick smoke test:
  python unified_phd_bearing_reproducibility.py --demo --epochs 2 --runs 1 --models entropy_cnn kcfp_lstm physics_dkc fatf --output demo_out

Real data:
  python unified_phd_bearing_reproducibility.py --data-root "D:/Datasets/Testing Data/Shandong University Fault Diagnosis Bearing Dataset/Dataset/【0000】汇总和说明文件" --epochs 100 --runs 5 --output phd_results

See --help for more controls. Figures saved as PNG (600 dpi), PDF, and EPS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import medfilt
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import pywt
except ImportError:
    pywt = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

LOG = logging.getLogger('phd_bearing')
CLASS_FILES = ('0031', '0042', '0051', '0063', '0083', '0092', '0093', '0043', '0032')
CLASS_NAMES = tuple(f'{k}' for k in CLASS_FILES)
ALL_MODELS = ('entropy_cnn', 'kcfp_lstm', 'physics_dkc', 'fatf')
PALETTE = ('#3564AD', '#D76839', '#228C78', '#9A57A5', '#D5A12F',
           '#5F7592', '#AF5879', '#6F9A47', '#8E6C49')


@dataclass(frozen=True)
class Config:
    data_root: str
    output: str
    demo: bool = False
    models: Tuple[str, ...] = ALL_MODELS
    signal_length: int = 480
    group_chunks: int = 12
    max_chunks_per_class: int = 0
    epochs: int = 100
    runs: int = 3
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 20
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    noise_std: float = 0.02
    seed: int = 2026
    num_workers: int = 0
    device: str = 'auto'
    figures: bool = True
    tsne: bool = True
    classical: bool = True
    num_classes: int = len(CLASS_FILES)


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default='.', help='Folder containing the 9 .mat files')
    ap.add_argument('--output', default='phd_reproducibility_output')
    ap.add_argument('--demo', action='store_true', help='Synthetic smoke-test data only')
    ap.add_argument('--models', nargs='+', choices=ALL_MODELS, default=list(ALL_MODELS))
    ap.add_argument('--signal-length', type=int, default=480)
    ap.add_argument('--group-chunks', type=int, default=12)
    ap.add_argument('--max-chunks-per-class', type=int, default=0, help='0 means all')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--learning-rate', type=float, default=5e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--test-fraction', type=float, default=0.20)
    ap.add_argument('--noise-std', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--no-tsne', action='store_true')
    ap.add_argument('--no-classical', action='store_true')
    a = ap.parse_args()
    if not (0 < a.val_fraction < 0.5 and 0 < a.test_fraction < 0.5 and
            a.val_fraction + a.test_fraction < 0.8):
        ap.error('validation and test fractions must be positive and sum to < 0.8')
    for nm in ('signal_length', 'group_chunks', 'epochs', 'runs', 'batch_size'):
        if getattr(a, nm) < 1:
            ap.error(f'--{nm.replace("_", "-")} must be >= 1')
    if a.signal_length < 128:
        ap.error('--signal-length must be at least 128 for STFT and pooling')
    return Config(data_root=a.data_root, output=a.output, demo=a.demo,
                  models=tuple(dict.fromkeys(a.models)), signal_length=a.signal_length,
                  group_chunks=a.group_chunks, max_chunks_per_class=a.max_chunks_per_class,
                  epochs=a.epochs, runs=a.runs, batch_size=a.batch_size,
                  learning_rate=a.learning_rate, weight_decay=a.weight_decay,
                  patience=a.patience, val_fraction=a.val_fraction,
                  test_fraction=a.test_fraction, noise_std=a.noise_std,
                  seed=a.seed, num_workers=a.num_workers, device=a.device,
                  figures=not a.no_figures, tsne=not a.no_tsne,
                  classical=not a.no_classical)


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def detect_device(choice: str) -> torch.device:
    if choice != 'auto':
        d = torch.device(choice)
        if choice == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        if choice == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is not available')
        return d
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_raw_signals(cfg: Config) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    if cfg.demo:
        rng = np.random.default_rng(cfg.seed)
        signals = {}
        for c in range(cfg.num_classes):
            n = cfg.signal_length * max(72, 7 * cfg.group_chunks)
            tt = np.arange(n, dtype=np.float64)
            carrier = 0.018 + c * 0.0019
            rotation = 1.0 + 0.00006 * tt / cfg.signal_length
            s = (np.sin(2 * np.pi * carrier * rotation * tt) +
                 0.43 * np.sin(2 * np.pi * (2.2 * carrier) * tt + 0.3 * c) +
                 0.26 * np.sin(2 * np.pi * (c + 1) * 0.004 * tt))
            impulse_interval = 35 + 4 * c
            impulse = np.zeros(n)
            impulse[::impulse_interval] = 0.28 + 0.07 * c
            kernel = np.exp(-np.arange(30) / 7.0) * np.sin(np.arange(30) * 1.4)
            s += np.convolve(impulse, kernel, mode='same')
            s += rng.normal(0, 0.16, n)
            signals[c] = s.astype('float32')
        return signals, {'mode': 'SYNTHETIC_DEMO_ONLY', 'files': {}}
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f'Data directory not found: {root}')
    result, manifests = {}, {}
    for c, stem in enumerate(CLASS_FILES):
        found = sorted(root.rglob(stem + '.mat'))
        if not found:
            raise FileNotFoundError(f'Missing {stem}.mat under {root}. All nine files required.')
        if len(found) > 1:
            LOG.warning('Multiple matches for %s: using %s', stem, found[0])
        p = found[0]
        data = loadmat(p)
        key = 'y' + stem
        if key not in data:
            raise KeyError(f'{p}: expected MATLAB variable {key!r}, found {list(data)}')
        sig = np.asarray(data[key], dtype=np.float64).ravel()
        if not np.isfinite(sig).all() or len(sig) < cfg.signal_length * cfg.group_chunks * 4:
            raise ValueError(f'{p}: insufficient samples or non-finite samples')
        result[c] = sig.astype(np.float32)
        manifests[stem] = {'path': str(p.resolve()), 'sha256': file_checksum(p),
                           'raw_samples': int(len(sig)), 'mat_key': key}
    return result, {'mode': 'REAL_MAT_DATA', 'files': manifests}


def segment_signals(signals: Dict[int, np.ndarray], cfg: Config):
    """Non-overlapping windows; groups isolate contiguous recording regions."""
    segments, y, groups, metadata = [], [], [], []
    for c, raw in sorted(signals.items()):
        n = len(raw) // cfg.signal_length
        if cfg.max_chunks_per_class:
            n = min(n, cfg.max_chunks_per_class)
        if n < cfg.group_chunks * 4:
            raise ValueError(f'Class {c}: need >=4 groups, have {n // cfg.group_chunks}')
        for idx in range(n):
            start = idx * cfg.signal_length
            segments.append(raw[start:start + cfg.signal_length].copy())
            y.append(c)
            groups.append(f'{c}:{idx // cfg.group_chunks}')
            metadata.append({'class_id': c, 'file': CLASS_FILES[c], 'sample_offset': start,
                             'group': groups[-1]})
    return (np.stack(segments).astype(np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(groups), metadata)


def group_stratified_indices(labels: np.ndarray, groups: np.ndarray, cfg: Config):
    """Split whole nonoverlapping groups within each class, never samplewise."""
    rng = np.random.default_rng(cfg.seed)
    splits = [[], [], []]
    info = {}
    for cls in np.unique(labels):
        class_idx = np.flatnonzero(labels == cls)
        uniq = np.unique(groups[class_idx])
        if len(uniq) < 4:
            raise ValueError(f'Class {cls}: too few independent groups ({len(uniq)})')
        perm = rng.permutation(uniq)
        nt = max(1, int(round(len(perm) * cfg.test_fraction)))
        nv = max(1, int(round(len(perm) * cfg.val_fraction)))
        if nt + nv >= len(perm):
            raise ValueError('Not enough groups to split into train/val/test')
        sets = (perm[nt+nv:], perm[nt:nt+nv], perm[:nt])
        for si, gset in enumerate(sets):
            ix = class_idx[np.isin(groups[class_idx], gset)]
            splits[si].extend(ix.tolist())
        info[int(cls)] = {'train_groups': sets[0].tolist(), 'val_groups': sets[1].tolist(),
                          'test_groups': sets[2].tolist()}
    out = [np.asarray(sorted(s), dtype=np.int64) for s in splits]
    ga, gb, gc = (set(groups[s]) for s in out)
    assert not (ga & gb or ga & gc or gb & gc), 'group leakage detected'
    assert len(np.unique(np.concatenate(out))) == len(labels)
    return out, info


def normalized_signal(x: np.ndarray) -> np.ndarray:
    """Per-window normalization: no statistics derived from held-out windows."""
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean()
    return np.clip(x / (x.std() + 1e-6), -8.0, 8.0)


def shannon_entropy(x: np.ndarray, bins: int = 64) -> float:
    hist = np.histogram(np.clip(x, -8, 8), bins=bins, range=(-8, 8))[0]
    p = hist[hist > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def wavelet_volterra_features(x: np.ndarray) -> np.ndarray:
    x = normalized_signal(x).astype(np.float64)
    if pywt is not None:
        coeffs = pywt.wavedec(x, 'db4', level=3)
        wave = np.array([z for c in coeffs for z in (np.mean(c), np.std(c))])
    else:
        # Workable dependency-free summary, logged as NOT equivalent to db4.
        spec = np.abs(np.fft.rfft(x))
        wave = np.array([z for seg in np.array_split(spec, 4)
                         for z in (np.mean(seg), np.std(seg))])
    volterra2 = x * np.roll(x, 1)
    volterra3 = x * np.roll(x, 2)
    return np.concatenate((wave, volterra2, volterra3)).astype(np.float32)


class BearingDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, index: np.ndarray,
                 training: bool = False, noise_std: float = 0.02, seed: int = 0):
        self.x = signals[index]
        self.y = labels[index]
        self.training = training
        self.noise_std = noise_std
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i: int):
        x = normalized_signal(self.x[i])
        if self.training and self.noise_std:
            # Different but fully deterministic augmentation per epoch/sample.
            rng = np.random.default_rng(self.seed + self.epoch * 1000003 + int(i))
            x = x + rng.normal(0, self.noise_std, len(x)).astype('float32')
            x = x * np.float32(rng.uniform(0.96, 1.04))
        e = shannon_entropy(x)
        dual = np.stack((x, np.full_like(x, e / 6.0)))
        return torch.from_numpy(dual.copy()), torch.tensor(int(self.y[i]), dtype=torch.long)


class FSC(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, cout, 3, padding=dilation,
                                          dilation=dilation, bias=False),
                                 nn.BatchNorm1d(cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class EntropyAugmentedKCFPCNN(nn.Module):
    """Memory-efficient adaptation of the supplied entropy-input CNN."""
    def __init__(self, classes=9):
        super().__init__()
        self.fsc1 = FSC(2, 32)
        self.fsc2 = FSC(32, 64, dilation=2)
        self.multiscale = nn.ModuleList([FSC(64, 64, dilation=d) for d in (1, 2, 4)])
        self.mix = nn.Sequential(nn.Conv1d(64 * 3, 128, 1), nn.BatchNorm1d(128), nn.GELU())
        self.features = nn.Sequential(FSC(128, 256), nn.MaxPool1d(2),
                                      FSC(256, 256), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(),
                                  nn.Dropout(0.35), nn.Linear(128, classes))

    def embed(self, x):
        x = F.max_pool1d(self.fsc1(x), 2)
        x = F.max_pool1d(self.fsc2(x), 2)
        x = self.mix(torch.cat([m(x) for m in self.multiscale], dim=1))
        return self.features(x).squeeze(-1)

    def forward(self, x):
        return self.head(self.embed(x))


class KroneckerDilatedConv1d(nn.Module):
    """Explicit outer-product Kronecker-structured 1D convolution kernels.

    Each (out,in) kernel K = kron(A, B), where A has a elements and B has b.
    The resulting (a*b)-tap kernel uses a selected signal-domain dilation.
    This is a concrete factorization, not merely a dilated ordinary Conv1d.
    """
    def __init__(self, cin: int, cout: int, a: int = 3, b: int = 3, dilation: int = 1):
        super().__init__()
        self.A = nn.Parameter(torch.empty(cout, cin, a))
        self.B = nn.Parameter(torch.empty(cout, cin, b))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.dilation = dilation
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        w = torch.einsum('oia,oib->oiab', self.A, self.B)
        w = w.reshape(self.A.shape[0], self.A.shape[1], -1)
        pad_total = (w.shape[-1] - 1) * self.dilation
        x = F.pad(x, (pad_total // 2, pad_total - pad_total // 2))
        return F.conv1d(x, w, self.bias, dilation=self.dilation)


class DKCBlock(nn.Module):
    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.net = nn.Sequential(KroneckerDilatedConv1d(cin, cout, 3, 3, dilation),
                                 nn.BatchNorm1d(cout), nn.GELU(),
                                 nn.Conv1d(cout, cout, 1), nn.BatchNorm1d(cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))


class SpectralResonanceGate(nn.Module):
    """Trainable non-negative Gaussian resonances, a physically motivated prior.

    This uses normalized frequency (not bearing-specific BPFO/BPFI values),
    since RPM, geometry, and sample rate were not provided.
    """
    def __init__(self, bands: int = 6):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-2.0, 2.0, bands))
        self.width_logit = nn.Parameter(torch.full((bands,), -1.0))
        self.amplitudes = nn.Parameter(torch.zeros(bands))
        self.residual_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        freqs = torch.linspace(0, 1, spectrum.shape[-1], device=x.device)
        centers = torch.sigmoid(self.centers)
        widths = 0.025 + F.softplus(self.width_logit) * 0.12
        kernels = torch.exp(-0.5 * ((freqs[None, :] - centers[:, None]) /
                                    widths[:, None]).square())
        response = (F.softplus(self.amplitudes)[:, None] * kernels).sum(0)
        response = response / response.amax().clamp_min(1e-6)
        gated = torch.fft.irfft(spectrum * response[None, None, :], n=x.shape[-1], dim=-1)
        alpha = torch.sigmoid(self.residual_logit)
        return x + alpha * gated.to(x.dtype)


class PhysicsInspiredDKCNet(nn.Module):
    def __init__(self, classes=9):
        super().__init__()
        self.resonance = SpectralResonanceGate()
        self.stem = nn.Sequential(nn.Conv1d(2, 24, 5, stride=2, padding=2),
                                  nn.BatchNorm1d(24), nn.GELU())
        self.layers = nn.Sequential(DKCBlock(24, 32, 1), nn.MaxPool1d(2),
                                    DKCBlock(32, 48, 2), nn.MaxPool1d(2),
                                    DKCBlock(48, 64, 4), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(64, classes))

    def embed(self, x):
        raw = self.resonance(x[:, :1, :])
        x = torch.cat((raw, x[:, 1:, :]), 1)
        return self.layers(self.stem(x)).squeeze(-1)

    def forward(self, x):
        return self.head(self.embed(x))


class KCFPLSTM(nn.Module):
    """Repaired depthwise/grouped KCFP-LSTM baseline adapted from source."""
    def __init__(self, classes=9):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, 16, 3, padding=1), nn.BatchNorm1d(16),
                                  nn.LeakyReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(16, 16, 3, padding=1, groups=16),
                                  nn.Conv1d(16, 32, 1), nn.BatchNorm1d(32),
                                  nn.LeakyReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 32, 3, padding=1, groups=32),
                                  nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64),
                                  nn.LeakyReLU(), nn.AdaptiveAvgPool1d(48))
        self.lstm = nn.LSTM(input_size=64, hidden_size=96, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(96, classes))

    def embed(self, x):
        h = self.stem(x).transpose(1, 2)
        h, _ = self.lstm(h)
        return h[:, -1]

    def forward(self, x):
        return self.head(self.embed(x))


class FrequencyAwareTransformerFusion(nn.Module):
    """New dual-stream time-frequency transformer; NOT in supplied sources.

    Raw sequence tokens: strided convolution.
    Spectral tokens: log-magnitude STFT pooled across time windows.
    Frequency token positions: normalized frequency coordinate embeddings.
    Cross-attention: waveform queries attend to spectral representations.
    """
    def __init__(self, classes=9, width=64, heads=4, layers=2):
        super().__init__()
        self.nfft = 64
        self.hop = 16
        self.register_buffer('hann', torch.hann_window(self.nfft), persistent=False)
        self.temporal = nn.Sequential(nn.Conv1d(2, width, 7, stride=4, padding=3),
                                      nn.BatchNorm1d(width), nn.GELU(),
                                      nn.Conv1d(width, width, 3, stride=2, padding=1),
                                      nn.GELU())
        self.spectral_embed = nn.Linear(2, width)
        self.cross = nn.MultiheadAttention(width, heads, dropout=0.1, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
                                           dim_feedforward=width * 3, dropout=0.12,
                                           activation='gelu', batch_first=True,
                                           norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width * 2)
        self.head = nn.Sequential(nn.Linear(width * 2, 96), nn.GELU(),
                                  nn.Dropout(0.25), nn.Linear(96, classes))

    def embed(self, x):
        raw = x[:, 0].float()
        stft = torch.stft(raw, n_fft=self.nfft, hop_length=self.hop,
                          window=self.hann.to(raw.device), return_complex=True)
        logpower = torch.log1p(stft.abs()).mean(dim=-1)
        freqs = torch.linspace(0, 1, logpower.shape[1],
                               device=x.device, dtype=logpower.dtype)
        stokens = self.spectral_embed(torch.stack(
            (logpower, freqs[None, :].expand(raw.shape[0], -1)), dim=-1))
        ttokens = self.temporal(x).transpose(1, 2)
        # Shared sinusoidal temporal coordinates to preserve token ordering.
        pos = torch.arange(ttokens.shape[1], device=x.device, dtype=ttokens.dtype)
        div = torch.exp(torch.arange(0, ttokens.shape[-1], 2,
                                      device=x.device, dtype=ttokens.dtype) *
                        (-math.log(10000.) / ttokens.shape[-1]))
        pe = torch.zeros_like(ttokens[0])
        pe[:, 0::2] = torch.sin(pos[:, None] * div[None, :])
        pe[:, 1::2] = torch.cos(pos[:, None] * div[None, :])
        q = ttokens + pe.unsqueeze(0)
        attended, _ = self.cross(q, stokens, stokens, need_weights=False)
        out = self.encoder(q + attended)
        return self.norm(torch.cat((out.mean(1), stokens.mean(1)), dim=-1))

    def forward(self, x):
        return self.head(self.embed(x))


def create_model(name: str, classes: int) -> nn.Module:
    return {'entropy_cnn': EntropyAugmentedKCFPCNN,
            'kcfp_lstm': KCFPLSTM,
            'physics_dkc': PhysicsInspiredDKCNet,
            'fatf': FrequencyAwareTransformerFusion}[name](classes)


def build_loaders(signals, labels, splits, cfg, seed):
    loaders = {}
    for name, ids in zip(('train', 'val', 'test'), splits):
        ds = BearingDataset(signals, labels, ids, training=(name == 'train'),
                            noise_std=cfg.noise_std, seed=seed)
        gen = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(ds, batch_size=cfg.batch_size,
                                   shuffle=(name == 'train'), num_workers=cfg.num_workers,
                                   generator=gen, pin_memory=False)
    return loaders


@torch.no_grad()
def evaluate_torch(model, loader, device):
    model.eval()
    all_y, all_p, all_prob, all_emb = [], [], [], []
    loss_sum, n = 0., 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        p = torch.softmax(logits, 1)
        all_y.append(y.cpu().numpy())
        all_p.append(p.argmax(1).cpu().numpy())
        all_prob.append(p.cpu().numpy())
        all_emb.append(model.embed(x).cpu().numpy())
        n += len(y)
    yy = np.concatenate(all_y)
    pp = np.concatenate(all_p)
    return {'loss': loss_sum / max(n, 1), 'y': yy, 'pred': pp,
            'prob': np.concatenate(all_prob), 'embeddings': np.concatenate(all_emb),
            **basic_metrics(yy, pp)}


def basic_metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y, pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y, pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y, pred, average='macro', zero_division=0))}


def train_once(model_name, run, signals, labels, splits, cfg, device, out):
    seed_everything(cfg.seed + 37 * run)
    loaders = build_loaders(signals, labels, splits, cfg, cfg.seed + 37 * run)
    model = create_model(model_name, cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    best_val, best_epoch = float('inf'), -1
    state = None
    history = []
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        loaders['train'].dataset.epoch = epoch
        loss_total, n, correct = 0., 0, 0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        scheduler.step()
        val = evaluate_torch(model, loaders['val'], device)
        history.append({'epoch': epoch + 1, 'train_loss': loss_total / n,
                        'train_accuracy': correct / n, 'val_loss': val['loss'],
                        'val_accuracy': val['accuracy'], 'lr': optimizer.param_groups[0]['lr']})
        if val['loss'] < best_val - 1e-5:
            best_val, best_epoch = val['loss'], epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            LOG.info('%s run %d epoch %d/%d train=%.3f val=%.3f', model_name, run + 1,
                     epoch + 1, cfg.epochs, correct / n, val['accuracy'])
        if epoch - best_epoch >= cfg.patience:
            break
    assert state is not None
    model.load_state_dict(state)
    test = evaluate_torch(model, loaders['test'], device)
    test['seconds'] = time.perf_counter() - start
    test['best_epoch'] = best_epoch + 1
    subdir = out / model_name / f'run_{run+1:02d}'
    subdir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state, 'model': model_name, 'config': asdict(cfg),
                'best_epoch': best_epoch + 1}, subdir / 'best_checkpoint.pt')
    write_csv(subdir / 'history.csv', history)
    np.savez_compressed(subdir / 'test_predictions.npz',
                        y_true=test['y'], y_pred=test['pred'],
                        probabilities=test['prob'], embeddings=test['embeddings'],
                        test_sample_indices=splits[2])
    (subdir / 'classification_report.txt').write_text(
        classification_report(test['y'], test['pred'], labels=list(range(cfg.num_classes)),
                              target_names=CLASS_NAMES, zero_division=0), encoding='utf-8')
    metrics = {k: v for k, v in test.items() if isinstance(v, (float, int))}
    metrics.update({'model': model_name, 'run': run + 1})
    save_json(subdir / 'metrics.json', metrics)
    if cfg.figures:
        plot_training(history, subdir / 'learning_curves')
        plot_cm(test['y'], test['pred'], cfg.num_classes, subdir / 'confusion_matrix')
        if cfg.tsne:
            plot_tsne(test['embeddings'], test['y'], subdir / 'tsne_embeddings',
                      seed=cfg.seed)
    return metrics


def write_csv(path: Path, rows: Sequence[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def save_json(path: Path, data: Any):
    def to_native(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               default=to_native), encoding='utf-8')


def apply_fig_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.labelsize': 9, 'axes.linewidth': 0.8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.facecolor': 'white'})


def save_fig(fig, stem: Path):
    for ext in ('png', 'pdf', 'eps'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig.savefig(stem.with_suffix('.' + ext), dpi=600,
                        bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def plot_training(history, dest):
    ep = [h['epoch'] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    for i, pair in enumerate((('train_loss', 'val_loss'),
                              ('train_accuracy', 'val_accuracy'))):
        for nm, color, linestyle in zip(pair, (PALETTE[0], PALETTE[1]), ('-', '--')):
            ax[i].plot(ep, [h[nm] for h in history], color=color, ls=linestyle,
                       lw=1.5, label=nm.replace('_', ' ').title())
        ax[i].set(xlabel='Epoch', ylabel=('Cross-entropy loss' if i == 0 else 'Accuracy'))
        ax[i].grid(alpha=0.2, linestyle=':')
        ax[i].legend(frameon=False)
        if i == 1:
            ax[i].set_ylim(0, 1.03)
    save_fig(fig, dest)


def plot_cm(y, pred, classes, dest):
    cm = confusion_matrix(y, pred, labels=list(range(classes)), normalize='true')
    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    im = ax.imshow(cm * 100, cmap='Blues', vmin=0, vmax=100)
    for i in range(classes):
        for j in range(classes):
            v = cm[i, j] * 100
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if v > 55 else '#24364c')
    ax.set_xticks(range(classes), CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(classes), CLASS_NAMES)
    ax.set_xlabel('Predicted class (file ID)')
    ax.set_ylabel('True class (file ID)')
    fig.colorbar(im, ax=ax, fraction=0.04, label='Class recall (%)')
    save_fig(fig, dest)


def plot_tsne(features, labels, dest, seed=42):
    n = len(labels)
    if n < 8:
        LOG.warning('Skipping t-SNE: too few test samples')
        return
    rng = np.random.default_rng(seed)
    keep = rng.choice(n, size=min(900, n), replace=False)
    xx = StandardScaler().fit_transform(features[keep])
    if xx.shape[1] > 45 and len(xx) > 50:
        xx = PCA(n_components=min(45, xx.shape[1], len(xx) - 1), random_state=seed).fit_transform(xx)
    xy = TSNE(n_components=2, perplexity=min(30, max(3, (len(xx)-1)//3)),
              init='pca', learning_rate='auto', random_state=seed).fit_transform(xx)
    fig, ax = plt.subplots(figsize=(4.15, 3.05), constrained_layout=True)
    for c in np.unique(labels[keep]):
        ix = labels[keep] == c
        ax.scatter(xy[ix, 0], xy[ix, 1], s=16, color=PALETTE[int(c) % len(PALETTE)],
                   alpha=0.7, edgecolors='none', label=CLASS_NAMES[int(c)])
    ax.set(xlabel='t-SNE dimension 1', ylabel='t-SNE dimension 2')
    ax.legend(ncol=3, fontsize=7, frameon=False, title='File class', title_fontsize=7)
    ax.grid(alpha=0.13, linestyle=':')
    save_fig(fig, dest)


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return float(vals.mean()), None, None
    mean = float(vals.mean())
    half = float(student_t.ppf(0.975, len(vals)-1) *
                 vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, mean-half, mean+half


def benchmark_classical(signals, labels, splits, cfg, out):
    if pywt is None:
        LOG.warning('PyWavelets absent: using FFT statistical summaries instead of db4 coefficients')
    # Feature construction occurs on each split, scaling is fitted TRAIN ONLY.
    def feat(ids):
        return np.stack([wavelet_volterra_features(medfilt(signals[i], kernel_size=3))
                         for i in ids])
    train, val, test = [feat(ids) for ids in splits]
    scaler = StandardScaler().fit(train)
    xtr = scaler.transform(train)
    xte = scaler.transform(test)
    yt = labels[splits[0]]
    ytest = labels[splits[2]]
    results = []
    candidates = [('RandomForest', RandomForestClassifier(n_estimators=200,
                   n_jobs=-1, random_state=cfg.seed, class_weight='balanced_subsample'))]
    if XGBClassifier is not None:
        candidates.append(('XGBoost', XGBClassifier(n_estimators=150, max_depth=4,
                            learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
                            eval_metric='mlogloss', tree_method='hist', n_jobs=4,
                            random_state=cfg.seed)))
    for name, clf in candidates:
        LOG.info('Fitting classical baseline: %s', name)
        clf.fit(xtr, yt)
        pred = clf.predict(xte)
        met = basic_metrics(ytest, pred)
        met.update({'model': name, 'run': 1})
        results.append(met)
        np.savez_compressed(out / f'{name}_predictions.npz', y_true=ytest, y_pred=pred,
                            test_sample_indices=splits[2])
        if cfg.figures:
            plot_cm(ytest, pred, cfg.num_classes, out / f'{name}_confusion_matrix')
    return results


def compile_summary(rows, dest: Path, cfg: Config):
    write_csv(dest / 'all_run_metrics.csv', rows)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)
    metrics = ('accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
               'precision_macro', 'recall_macro')
    summary = []
    for name, records in by_model.items():
        row = {'model': name, 'runs': len(records)}
        for key in metrics:
            mean, lo, hi = ci95([a[key] for a in records])
            row.update({key + '_mean': mean, key + '_ci95_lower': lo,
                        key + '_ci95_upper': hi})
        summary.append(row)
    write_csv(dest / 'summary_95ci.csv', summary)
    save_json(dest / 'summary_95ci.json', summary)
    if cfg.figures and summary:
        fig, ax = plt.subplots(figsize=(max(5.3, len(summary)*1.15), 3.1),
                                constrained_layout=True)
        vals = [z['accuracy_mean']*100 for z in summary]
        lower = [(z['accuracy_mean']-z['accuracy_ci95_lower'])*100
                 if z['accuracy_ci95_lower'] is not None else 0 for z in summary]
        upper = [(z['accuracy_ci95_upper']-z['accuracy_mean'])*100
                 if z['accuracy_ci95_upper'] is not None else 0 for z in summary]
        xx = np.arange(len(summary))
        bars = ax.bar(xx, vals, color=[PALETTE[i % len(PALETTE)] for i in xx],
                      alpha=0.85, width=0.6, yerr=[lower, upper], capsize=4,
                      edgecolor='#35455e', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.7, f'{val:.1f}%',
                    fontsize=8, ha='center')
        ax.set_xticks(xx, [z['model'].replace('_', '\n') for z in summary])
        ax.set_ylabel('Held-out test accuracy (%)')
        ax.set_ylim(0, max(110, max(vals)+12))
        ax.grid(axis='y', alpha=0.18)
        ax.set_axisbelow(True)
        save_fig(fig, dest / 'model_comparison_95ci')
    return summary


def environment_info():
    return {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': __import__('sklearn').__version__, 'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda, 'pywavelets': getattr(pywt, '__version__', None),
            'xgboost': getattr(__import__('sys').modules.get('xgboost'), '__version__', None)}


def main():
    cfg = parse_args()
    dest = Path(cfg.output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(dest / 'execution.log', encoding='utf-8')])
    seed_everything(cfg.seed)
    apply_fig_style()
    device = detect_device(cfg.device)
    LOG.info('Device=%s, models=%s, synthetic_demo=%s', device, cfg.models, cfg.demo)
    if cfg.demo:
        LOG.warning('SYNTHETIC DEMO: results must not be used as bearing benchmark evidence')
    signals, manifest = load_raw_signals(cfg)
    segments, labels, groups, metadata = segment_signals(signals, cfg)
    splits, group_info = group_stratified_indices(labels, groups, cfg)
    save_json(dest / 'experiment_manifest.json', {'config': asdict(cfg),
              'environment': environment_info(), 'data': manifest,
              'class_names': CLASS_NAMES, 'groups': group_info,
              'counts': {k: dict(Counter(map(int, labels[ids])))
                         for k, ids in zip(('train', 'val', 'test'), splits)},
              'methodological_notes': [
                  'No window overlap. Whole contiguous groups assigned to only one split.',
                  'Nine input MAT files define nine classes; RPM/load labels not inferred.',
                  'Validation selects checkpoints; test only evaluated after selection.',
                  'Runs repeat model initialization with the same held-out group split.',
                  '95% t intervals across seeds are not confidence intervals across machines.',
                  'Physics DKC and FATF are newly authored title-inspired reference models.',
                  'RF/XGB use train-only fit for preprocessing; no leakage of test statistics.']})
    np.savez_compressed(dest / 'split_indices.npz', train=splits[0],
                        val=splits[1], test=splits[2], labels=labels, groups=groups)
    write_csv(dest / 'sample_metadata.csv', metadata)
    LOG.info('Split sizes: train=%d, val=%d, test=%d', *(len(q) for q in splits))
    rows = []
    for model in cfg.models:
        for run in range(cfg.runs):
            LOG.info('START model=%s run=%d/%d', model, run + 1, cfg.runs)
            result = train_once(model, run, segments, labels, splits, cfg, device, dest)
            rows.append(result)
            LOG.info('FINISHED %s run=%d test_accuracy=%.4f f1_macro=%.4f',
                     model, run+1, result['accuracy'], result['f1_macro'])
    if cfg.classical:
        rows.extend(benchmark_classical(segments, labels, splits, cfg, dest))
    summary = compile_summary(rows, dest, cfg)
    for item in summary:
        LOG.info('%s: accuracy %.2f%% (95%% CI: %s to %s)', item['model'],
                 item['accuracy_mean']*100,
                 ('n/a' if item['accuracy_ci95_lower'] is None else
                  f"{item['accuracy_ci95_lower']*100:.2f}%"),
                 ('n/a' if item['accuracy_ci95_upper'] is None else
                  f"{item['accuracy_ci95_upper']*100:.2f}%"))
    LOG.info('Done. Outputs: %s', dest)


if __name__ == '__main__':
    main()
