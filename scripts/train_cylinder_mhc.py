"""
train_cylinder_mhc.py — KBS-mHC-FNO experiment on Cylinder1.mat

This script implements the KBS-mHC-FNO architecture for Cylinder flow prediction.
Supports comprehensive performance comparison across multiple dimensions:
- With/without mHC
- Discrete vs continuous mHC mode
- Different expansion ratios
- Different KDB bandwidths

Data layout : Cylinder1.mat  key='u'  shape [N, T, C, H, W]
                N=747, T=20, C=3 (field / vx / mask), H=W=64

Design choices:
  * Uses mHC-enhanced FNO (channels-first [B,C,H,W])
  * positional_embedding="grid"  →  FNO appends (x,y) coords internally
  * Gaussian normalisation       →  per-channel (per time-step), full image
  * Loss: per-sample relative L2  ||pred - true|| / ||true||
  * Scheduler: OneCycleLR (10% linear warmup + cosine annealing)
  * Prediction: step=1 autoregressive rollout
    - training  : teacher-forcing (use ground truth to advance window)
    - evaluation: free-running   (use model output to advance window)
  * mHC architecture: controlled by --use_mhc (True/False)
    - mhc_mode: "discrete" or "continuous"
    - mhc_expansion_ratio: expansion ratio for manifold (n in n×C)
    - mhc_sinkhorn_iter: number of Sinkhorn iterations (fixed at 20)
    - mhc_kdb_bandwidth: bandwidth for kernel density balancing
"""

import argparse
import logging
import os
import sys
from timeit import default_timer

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralop.models import FNO

# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("KBS-mHC-FNO Cylinder experiment")
# ── data ──────────────────────────────────────────────────────────────────────
parser.add_argument("--ntrain",    type=int,   default=600)
parser.add_argument("--ntest",     type=int,   default=147)
parser.add_argument("--in_dim",    type=int,   default=10,
    help="T_in: number of input time steps")
parser.add_argument("--out_dim",   type=int,   default=1,
    help="step: output time steps per forward pass (1 = autoregressive)")
parser.add_argument("--data_path", type=str,
    default="/home/leeshu/wmm/neuraloperator-main/neuralop/data/datasets/data/Cylinder/Cylinder1.mat")
# ── model ─────────────────────────────────────────────────────────────────────
parser.add_argument("--modes",            type=int,   default=12)
parser.add_argument("--width",            type=int,   default=32,
    help="hidden_channels for FNO")
parser.add_argument("--n_layers",         type=int,   default=4)
parser.add_argument("--use_channel_mlp",  type=int,   default=1,
    help="1=True / 0=False")
parser.add_argument("--fno_skip",         type=str,   default="linear")
parser.add_argument("--padding",          type=int,   default=0,
    help="domain padding (0 = disabled)")
# ── mHC ───────────────────────────────────────────────────────────────────────
parser.add_argument("--use_mhc",         type=int,   default=1,
    help="1=True / 0=False - Enable mHC branch")
parser.add_argument("--mhc_mode",        type=str,   default="continuous",
    choices=["discrete", "continuous"],
    help="mHC operation mode")
parser.add_argument("--mhc_expansion_ratio", type=int, default=4,
    help="Expansion ratio for manifold (n in n×C). "
         "Controls the dimensionality of dual-stochastic constraint space. "
         "Common values: 2, 4, 8. Higher = more expressive but more parameters.")
parser.add_argument("--mhc_kdb_bandwidth",   type=float, default=0.5,
    help="Bandwidth for kernel density balancing in continuous mode. "
         "Controls spatial smoothing scale. "
         "Smaller (0.1) = less smoothing, preserve details. "
         "Larger (1.0+) = more smoothing, smoother gradients.")
parser.add_argument("--mhc_sinkhorn_iter",  type=int,   default=20,
    help="Number of Sinkhorn iterations for bistochastic projection. "
         "Fixed at 20 for stable performance with torch.compile and DDP.")
parser.add_argument("--mhc_kernel_size",    type=int,   default=5,
    help="Size of truncated kernel for KDB in continuous mode. "
         "Must be odd for symmetric padding.")
parser.add_argument("--mhc_weight_decay",   type=float, default=0.0,
    help="weight_decay for mHC parameters (if separated)")
# ── training ──────────────────────────────────────────────────────────────────
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--epochs",        type=int,   default=500)
parser.add_argument("--weight_decay",  type=float, default=1e-4)
parser.add_argument("--batch_size",    type=int,   default=10)
parser.add_argument("--max_grad_norm", type=float, default=None)
# ── curriculum learning ─────────────────────────────────────────────────────────
parser.add_argument("--use_curriculum",      type=int,   default=0,
    help="Enable curriculum learning (1=True, 0=False)")
parser.add_argument("--use_pushforward",     type=int,   default=0,
    help="Use pushforward trick (epoch-bound rollout expansion) (1=True, 0=False)")
parser.add_argument("--teacher_forcing_start", type=float, default=1.0,
    help="Initial teacher forcing ratio (1.0=always, 0.0=never)")
parser.add_argument("--teacher_forcing_end",   type=float, default=0.5,
    help="Final teacher forcing ratio")
parser.add_argument("--warmup_epochs",        type=int,   default=100,
    help="Number of epochs for warmup with 100%% teacher forcing")
parser.add_argument("--max_rollout_steps",    type=int,   default=5,
    help="Maximum rollout steps for pushforward trick")
parser.add_argument("--temporal_gamma",       type=float, default=0.8,
    help="Temporal discounting factor for rollout loss (0.8-0.95)")
# ── misc ──────────────────────────────────────────────────────────────────────
parser.add_argument("--gpu",       type=str, default="0")
parser.add_argument("--save_name", type=str, default="cylinder_fno_mhc")
parser.add_argument("--eval",      type=int, default=0,
    help="1 = load checkpoint and run evaluation only")
parser.add_argument("--seed",      type=int,   default=42,
    help="Random seed for reproducibility")
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────────
_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
log_dir = os.path.join(_repo_root, "logs", "Cylinder_mHC")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{args.save_name}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info(f"Args: {args}")

# ───────────────────────────────────────────────────────────────────────────────
# Checkpoint directory
# ───────────────────────────────────────────────────────────────────────────────
ckpt_dir = os.path.join(_repo_root, "checkpoints", "Cylinder_mHC")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, f"{args.save_name}.pt")

# ───────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ───────────────────────────────────────────────────────────────────────────────
ntrain = args.ntrain
ntest  = args.ntest
T_in   = args.in_dim    # 10
T      = 10             # future steps to predict
step   = args.out_dim   # 1
S      = 64

# ───────────────────────────────────────────────────────────────────────────────
# Gaussian Normalizer  (per-channel, channels-first [N, C, H, W])
# ───────────────────────────────────────────────────────────────────────────────
class GaussianNormalizer:
    """Normalise [N, C, H, W] to zero mean / unit std, per channel C."""

    def __init__(self, x, eps=1e-6):
        # average over batch and spatial dims, keep channel dim
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)   # [1, C, 1, 1]
        self.std  = x.std (dim=(0, 2, 3), keepdim=True).clamp(min=eps)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean

    def to(self, dev):
        self.mean = self.mean.to(dev)
        self.std  = self.std .to(dev)
        return self


# ───────────────────────────────────────────────────────────────────────────────
# Data loading  [N, T_total, C, H, W] = [747, 20, 3, 64, 64]
# ───────────────────────────────────────────────────────────────────────────────
log.info(f"Loading data from {args.data_path} ...")
raw = scipy.io.loadmat(args.data_path)["u"]            # [747, 20, 3, 64, 64]
log.info(f"  raw shape : {raw.shape}")
assert raw.shape == (747, 20, 3, 64, 64), \
    f"Expected (747,20,3,64,64), got {raw.shape}"


def _field(arr, idx_n, t0, t1):
    """Physical field C=0 → [n, T, H, W]  (channels-first)."""
    x = arr[idx_n, t0:t1, 0]                           # [n, T, H, W]
    return torch.from_numpy(x.astype(np.float32))


def _mask(arr, idx_n, t0, t1):
    """Mask C=2 (clip to [0,1]) → [n, T, H, W]  (channels-first)."""
    x = arr[idx_n, t0:t1, -1:]                         # [n, T, 1, H, W]
    n_actual = x.shape[0]
    x = x.reshape(n_actual, t1 - t0, S, S)             # [n, T, H, W]
    return torch.from_numpy(np.clip(x, 0.0, 1.0).astype(np.float32))


train_idx = slice(0, ntrain)
test_idx  = slice(-ntest, None)

train_a  = _field(raw, train_idx, 0,    T_in)           # [600, 10, 64, 64]
train_u  = _field(raw, train_idx, T_in, T_in + T)       # [600, 10, 64, 64]
train_am = _mask (raw, train_idx, 0,    T_in)           # [600, 10, 64, 64]
train_um = _mask (raw, train_idx, T_in, T_in + T)       # [600, 10, 64, 64]

test_a   = _field(raw, test_idx,  0,    T_in)           # [147, 10, 64, 64]
test_u   = _field(raw, test_idx,  T_in, T_in + T)       # [147, 10, 64, 64]
test_am  = _mask (raw, test_idx,  0,    T_in)           # [147, 10, 64, 64]
test_um  = _mask (raw, test_idx,  T_in, T_in + T)       # [147, 10, 64, 64]

log.info(f"  train_a : {train_a.shape}   train_u : {train_u.shape}")
log.info(f"  test_a  : {test_a.shape}    test_u  : {test_u.shape}")
assert train_a.shape == (ntrain, T_in, S, S)
assert test_a.shape  == (ntest,  T_in, S, S)

# ── Gaussian normalisation (stats from training set, full image) ──────────────
a_normalizer = GaussianNormalizer(train_a)
u_normalizer = GaussianNormalizer(train_u)

train_a = a_normalizer.encode(train_a)
train_u = u_normalizer.encode(train_u)
test_a  = a_normalizer.encode(test_a)
test_u  = u_normalizer.encode(test_u)

a_normalizer.to(device)
u_normalizer.to(device)

log.info(f"  a_norm  mean={a_normalizer.mean.mean().item():.4f}  "
         f"std={a_normalizer.std.mean().item():.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# Custom Dataset for Sliding Window Single-Step Training
# ───────────────────────────────────────────────────────────────────────────────
class SlidingWindowDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset for single-step FNO training.
    For each sample sequence, extract all possible (10 -> 1) mapping pairs.
    """
    def __init__(self, a_data, am_data, u_data, um_data, window_size=T_in):
        """
        Args:
            a_data: input time steps [N, T_total, H, W]
            am_data: input mask [N, T_total, H, W]
            u_data: target time steps [N, T_total, H, W]
            um_data: target mask [N, T_total, H, W]
            window_size: number of input time steps (10)
        """
        self.a_data = a_data
        self.am_data = am_data
        self.u_data = u_data
        self.um_data = um_data
        self.window_size = window_size
        self.num_windows = a_data.shape[1] - window_size  # 20 - 10 = 10 windows per sample

        # Pre-compute all possible (sample_idx, time_idx) pairs
        self.indices = []
        for n in range(a_data.shape[0]):
            for t in range(self.num_windows):
                self.indices.append((n, t))
        self.indices = np.array(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        # Input: 10 consecutive time steps starting at t
        a = self.a_data[n, t:t+self.window_size]  # [10, H, W]
        am = self.am_data[n, t:t+self.window_size]
        # Target: 1 time step at t+window_size
        u = self.u_data[n, t+self.window_size:t+self.window_size+1]  # [1, H, W]
        um = self.um_data[n, t+self.window_size:t+self.window_size+1]
        return a, am, u, um


# ───────────────────────────────────────────────────────────────────────────────
# Reconstruct full 20-step sequences for sliding window
# ───────────────────────────────────────────────────────────────────────────────
# Merge train_a and train_u back to full 20-step sequences
train_full = torch.cat([train_a, train_u], dim=1)      # [600, 20, 64, 64]
train_full_m = torch.cat([train_am, train_um], dim=1) # [600, 20, 64, 64]

import multiprocessing
# Use 75% of available CPUs for data loading, cap at 8
num_workers = min(8, max(1, multiprocessing.cpu_count() // 4))

train_loader = DataLoader(
    SlidingWindowDataset(train_full, train_full_m, train_full, train_full_m, window_size=T_in),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
)
test_loader = DataLoader(
    # For test, we still use the original format for autoregressive evaluation
    TensorDataset(test_a, test_am, test_u, test_um),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=min(4, num_workers),  # Fewer workers for test
    pin_memory=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# Model  (mHC-enhanced FNO)
# ───────────────────────────────────────────────────────────────────────────────
model = FNO(
    n_modes=(args.modes, args.modes),
    in_channels=T_in,
    out_channels=step,
    hidden_channels=args.width,
    n_layers=args.n_layers,
    positional_embedding="grid",           # appends (x,y) coords internally
    use_channel_mlp=bool(args.use_channel_mlp),
    fno_skip=args.fno_skip,
    domain_padding=args.padding if args.padding > 0 else None,
    use_mhc=bool(args.use_mhc),
    mhc_mode=args.mhc_mode,
    mhc_expansion_ratio=args.mhc_expansion_ratio,
    mhc_sinkhorn_iter=args.mhc_sinkhorn_iter,
    mhc_kdb_bandwidth=args.mhc_kdb_bandwidth,
    mhc_kernel_size=args.mhc_kernel_size,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Model: FNO + {'mHC' if args.use_mhc else 'no mHC'} | "
         f"hidden={args.width} | modes={args.modes} | layers={args.n_layers} | "
         f"mhc_mode={args.mhc_mode if args.use_mhc else 'N/A'} | "
         f"mhc_expansion_ratio={args.mhc_expansion_ratio if args.use_mhc else 'N/A'} | "
         f"mhc_kdb_bandwidth={args.mhc_kdb_bandwidth if args.use_mhc else 'N/A'} | "
         f"use_channel_mlp={args.use_channel_mlp} | params={n_params:,}")

# ── forward shape sanity check ─────────────────────────────────────────────────
with torch.no_grad():
    _xx, _xm, _, _ = next(iter(train_loader))
    _xx_d = (_xx * _xm).to(device)
    _im   = model(_xx_d)
    log.info(f"  [shape check] xx {_xx_d.shape} -> im {_im.shape}")
    assert _xx_d.shape == (args.batch_size, T_in, S, S), \
        f"Input shape mismatch: {_xx_d.shape}"
    assert _im.shape == (args.batch_size, step, S, S), \
        f"Output shape mismatch: {_im.shape}"

    # ── Grid embedding verification ─────────────────────────────────────────────────────
    # Verify grid embedding is working by checking lifting layer input channels
    log.info("  [grid verification]")
    log.info(f"  Input channels: {model.in_channels}")
    log.info(f"  Lifting layer input channels: {model.lifting.in_channels}")
    log.info(f"  Expected with grid: {model.in_channels + model.n_dim}")
    if model.lifting.in_channels == model.in_channels + model.n_dim:
        log.info(f"  ✓ Grid embedding is ACTIVE (coordinates embedded)")
    else:
        log.warning(f"  ✗ Grid embedding may NOT be working!")

    # ── mHC verification ─────────────────────────────────────────────────────────────
    if args.use_mhc:
        log.info("  [mHC verification]")
        log.info(f"  mHC enabled: {model.use_mhc}")
        log.info(f"  mHC mode: {model.mhc_mode}")
        log.info(f"  mHC expansion ratio: {model.mhc_expansion_ratio}")
        log.info(f"  mHC KDB bandwidth: {model.mhc_kdb_bandwidth}")
        log.info("  ✓ mHC branch configured")

    # Clear GPU memory after verification
    del _xx, _xm, _xx_d, _im
    torch.cuda.empty_cache()
log.info("  shape check passed ✓")

# ───────────────────────────────────────────────────────────────────────────────
# Loss: per-sample relative L2
# ───────────────────────────────────────────────────────────────────────────────
def rel_l2(pred, target, eps=1e-8):
    """Per-sample relative L2, averaged over batch.  pred, target: [B, ...]"""
    p = pred  .reshape(pred  .size(0), -1)
    t = target.reshape(target.size(0), -1)
    return ((p - t).norm(dim=1) / t.norm(dim=1).clamp(min=eps)).mean()

# ───────────────────────────────────────────────────────────────────────────────
# Curriculum Learning: Teacher Forcing Schedule
# ───────────────────────────────────────────────────────────────────────────────
def get_teacher_forcing_ratio(epoch, total_epochs, tf_start, tf_end, warmup_epochs):
    """
    Calculate teacher forcing ratio for current epoch.

    Three-stage schedule:
    1. Warmup (0-warmup_epochs): 100% teacher forcing
    2. Decay (warmup_epochs-total_epochs): linear decay from tf_start to tf_end
    3. Final (after total_epochs): stay at tf_end

    Parameters
    ----------
    epoch : int
        Current epoch number
    total_epochs : int
        Total number of training epochs
    tf_start : float
        Initial teacher forcing ratio (after warmup)
    tf_end : float
        Final teacher forcing ratio
    warmup_epochs : int
        Number of epochs for warmup with 100% teacher forcing

    Returns
    -------
    float
        Teacher forcing ratio for current epoch
    """
    if epoch < warmup_epochs:
        # Warmup phase: 100% teacher forcing
        return 1.0
    else:
        # Decay phase: linear from tf_start to tf_end
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return tf_start + (tf_end - tf_start) * progress

def get_rollout_steps(epoch, warmup_epochs, max_rollout_steps, total_epochs):
    """
    Calculate rollout steps for pushforward trick.

    Gradually increase rollout steps based on epoch milestones:
    - Epoch 0-warmup_epochs: 1-step (teacher forcing)
    - Then gradual expansion based on total training progress

    Parameters
    ----------
    epoch : int
        Current epoch number
    warmup_epochs : int
        Number of epochs for warmup with 1-step training
    max_rollout_steps : int
        Maximum rollout steps
    total_epochs : int
        Total number of training epochs

    Returns
    -------
    int
        Number of rollout steps for current epoch
    """
    if epoch < warmup_epochs:
        return 1  # Warmup: single-step

    # Calculate progress through training (excluding warmup)
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)

    # Gradual rollout expansion based on progress
    # 0% -> 25%: 2-step
    # 25% -> 50%: 3-step
    # 50% -> 75%: 4-step
    # 75% -> 100%: 5-step (or max_rollout_steps)
    if progress < 0.25:
        return 2
    elif progress < 0.50:
        return 3
    elif progress < 0.75:
        return 4
    else:
        return min(5, max_rollout_steps)

# ───────────────────────────────────────────────────────────────────────────────
# Error metrics functions
# ───────────────────────────────────────────────────────────────────────────────
def compute_mae(pred, target, mask=None):
    """Mean Absolute Error, optional mask."""
    if mask is not None:
        return torch.abs(pred - target)[mask].mean()
    return torch.abs(pred - target).mean()

def compute_rmse(pred, target, mask=None):
    """Root Mean Square Error, optional mask."""
    if mask is not None:
        return torch.sqrt(((pred - target)**2)[mask].mean())
    return torch.sqrt(((pred - target)**2).mean())

def compute_max_ae(pred, target, mask=None):
    """Maximum Absolute Error, optional mask."""
    if mask is not None:
        return torch.abs(pred - target)[mask].max()
    return torch.abs(pred - target).max()


# ───────────────────────────────────────────────────────────────────────────────
# Eval-only mode
# ───────────────────────────────────────────────────────────────────────────────
if args.eval:
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    log.info(f"Loaded checkpoint: {ckpt_path}")

    pred_all, gt_all = [], []
    # Store per-step metrics
    rel_l2_per_step = np.zeros(T)     # [T]
    mae_per_step = np.zeros(T)
    rmse_per_step = np.zeros(T)
    max_ae_per_step = np.zeros(T)

    with torch.no_grad():
        for xx, xm, yy, ym in test_loader:
            xx, xm, yy, ym = xx.to(device), xm.to(device), yy.to(device), ym.to(device)
            xx_roll = xx * xm                              # [B, T_in, H, W]
            preds   = []

            for t in range(0, T, step):
                im  = model(xx_roll)                        # free-running, no mask
                preds.append(im)
                # advance window along channel dim=1
                xx_roll = torch.cat([xx_roll[:, step:], im], dim=1)

            pred = torch.cat(preds, dim=1)                # [B, T, H, W]

            # Denormalise to physical space
            pred_phys = u_normalizer.decode(pred).cpu()   # [B, T, H, W]
            gt_phys   = u_normalizer.decode(yy).cpu()     # [B, T, H, W]

            # Apply mask ONLY for final error computation
            ym_cpu = ym.cpu()                             # [B, T, H, W]

            # Compute per-step metrics WITH mask
            for t in range(T):
                pred_t = pred_phys[:, t]                  # [B, H, W]
                gt_t   = gt_phys[:, t]                    # [B, H, W]
                mask_t = ym_cpu[:, t]                     # [B, H, W]

                # L2 relative error
                pred_t_masked = pred_t * mask_t
                gt_t_masked = gt_t * mask_t

                pred_norm = torch.norm(pred_t_masked.reshape(pred_t.size(0), -1), dim=1)
                gt_norm = torch.norm(gt_t_masked.reshape(gt_t.size(0), -1), dim=1).clamp(min=1e-8)
                rel_l2 = (torch.norm(pred_t_masked - gt_t_masked, dim=(1, 2)) / gt_norm).mean()

                # Other metrics (with mask)
                mae = compute_mae(pred_t, gt_t, mask_t.bool())
                rmse = compute_rmse(pred_t, gt_t, mask_t.bool())
                max_ae = compute_max_ae(pred_t, gt_t, mask_t.bool())

                rel_l2_per_step[t] += rel_l2.item()
                mae_per_step[t] += mae.item()
                rmse_per_step[t] += rmse.item()
                max_ae_per_step[t] += max_ae.item()

            pred_all.append(pred_phys)
            gt_all.append(gt_phys)

    # Average per-step metrics across batches
    rel_l2_per_step /= len(test_loader)
    mae_per_step /= len(test_loader)
    rmse_per_step /= len(test_loader)
    max_ae_per_step /= len(test_loader)

    pred_all = torch.cat(pred_all, dim=0).numpy()         # [ntest, T, H, W]
    gt_all   = torch.cat(gt_all,   dim=0).numpy()

    # Overall L2 error
    err_all = pred_all - gt_all
    rel_err = float(np.mean(
        np.linalg.norm(err_all.reshape(ntest, -1), axis=-1) /
        np.linalg.norm(gt_all.reshape(ntest, -1), axis=-1).clip(1e-8)
    ))
    log.info(f"Test relative L2 error (overall): {rel_err:.4f}")

    # Print per-step metrics
    log.info("Per-step metrics (averaged across batches):")
    for t in range(T):
        log.info(f"  Step {t+1}: rel_l2={rel_l2_per_step[t]:.4f}, "
                 f"mae={mae_per_step[t]:.4e}, rmse={rmse_per_step[t]:.4e}, "
                 f"max_ae={max_ae_per_step[t]:.4e}")

    results_dir = os.path.join(_repo_root, "results", "Cylinder_mHC")
    os.makedirs(results_dir, exist_ok=True)

    # Save detailed results
    metrics_dict = {
        "pred": pred_all,
        "gt": gt_all,
        "error": err_all,
        "rel_l2_per_step": rel_l2_per_step,
        "mae_per_step": mae_per_step,
        "rmse_per_step": rmse_per_step,
        "max_ae_per_step": max_ae_per_step,
        "overall_rel_l2": rel_err,
        # Add configuration info
        "config": {
            "use_mhc": args.use_mhc,
            "mhc_mode": args.mhc_mode,
            "mhc_expansion_ratio": args.mhc_expansion_ratio,
            "mhc_kdb_bandwidth": args.mhc_kdb_bandwidth,
            "mhc_sinkhorn_iter": args.mhc_sinkhorn_iter,
            "n_params": n_params,
        }
    }

    mat_path = os.path.join(results_dir, f"{args.save_name}_results.mat")
    scipy.io.savemat(mat_path, metrics_dict)
    log.info(f"Results saved: {mat_path}")

    # Save per-step metrics to separate CSV file
    import pandas as pd
    metrics_df = pd.DataFrame({
        "step": np.arange(1, T+1),
        "rel_l2": rel_l2_per_step,
        "mae": mae_per_step,
        "rmse": rmse_per_step,
        "max_ae": max_ae_per_step,
    })
    csv_path = os.path.join(results_dir, f"{args.save_name}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    log.info(f"Per-step metrics saved: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for label, data in [("pred", pred_all), ("gt", gt_all), ("error", err_all)]:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            im_img = ax.imshow(data[0, 0], cmap="viridis")   # first sample, t=0
            plt.colorbar(im_img, ax=ax)
            fig.savefig(
                os.path.join(results_dir, f"{args.save_name}_{label}_t0.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.close(fig)
        log.info(f"Plots saved to {results_dir}/")
    except Exception as exc:
        log.warning(f"Visualization skipped: {exc}")

    sys.exit(0)

# ───────────────────────────────────────────────────────────────────────────────
# Optimiser
# ───────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

log.info(f"Optimizer: Adam with lr={args.lr}, weight_decay={args.weight_decay}")

# OneCycleLR: 10% linear warmup + cosine annealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    epochs=args.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
)

# ───────────────────────────────────────────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────────────────────────────────────────
t_global = default_timer()

# Log training strategy settings
if args.use_pushforward:
    log.info(f"Pushforward Trick enabled:")
    log.info(f"  Warmup epochs: {args.warmup_epochs} (1-step training)")
    log.info(f"  Max rollout steps: {args.max_rollout_steps}")
    log.info(f"  Temporal discounting: γ = {args.temporal_gamma}")
    log.info(f"  Rollout expansion: every 50 epochs increase by 1 step")
elif args.use_curriculum:
    log.info(f"Curriculum Learning enabled (probabilistic):")
    log.info(f"  Teacher forcing: {args.teacher_forcing_start:.2f} -> {args.teacher_forcing_end:.2f}")
    log.info(f"  Warmup epochs: {args.warmup_epochs}")
else:
    log.info("Pure Teacher Forcing (single-step training)")

for ep in range(args.epochs):
    # Calculate teacher forcing ratio for this epoch (for probabilistic CL)
    tf_ratio = get_teacher_forcing_ratio(
        ep, args.epochs, args.teacher_forcing_start,
        args.teacher_forcing_end, args.warmup_epochs
    ) if args.use_curriculum else 1.0

    # Calculate rollout steps for pushforward trick
    rollout_steps = get_rollout_steps(
        ep, args.warmup_epochs, args.max_rollout_steps, args.epochs
    ) if args.use_pushforward else 1

    model.train()
    t1 = default_timer()
    train_loss_sum = 0.0
    n_batches = 0

    for xx, xm, yy, ym in train_loader:
        xx, xm, yy, ym = xx.to(device), xm.to(device), yy.to(device), ym.to(device)

        # Apply mask only to input
        xx = xx * xm  # [B, T_in, H, W]

        optimizer.zero_grad()

        # Choose training strategy
        if args.use_pushforward and rollout_steps > 1:
            # Pushforward trick: epoch-bound rollout with temporal discounting
            xx_roll = xx.clone()
            loss_t = torch.zeros(1, device=device)
            gamma = args.temporal_gamma

            for t_step in range(rollout_steps):
                im = model(xx_roll)  # [B, 1, H, W]
                # Use yy[:, t_step:t_step+1] as target
                if t_step < yy.shape[1]:
                    # Temporal discounting: later steps have less weight
                    weight = gamma ** t_step
                    loss_t += weight * rel_l2(im, yy[:, t_step:t_step+1])
                # Roll forward: use prediction as next input
                xx_roll = torch.cat([xx_roll[:, 1:], im], dim=1)

            # Normalize by sum of weights
            weight_sum = sum(gamma ** i for i in range(rollout_steps))
            loss = loss_t / weight_sum

        elif args.use_curriculum and (torch.rand(1).item() > tf_ratio):
            # Original probabilistic curriculum learning
            xx_roll = xx.clone()
            loss_t = torch.zeros(1, device=device)

            for t_step in range(5):  # Fixed 5-step rollout
                im = model(xx_roll)
                if t_step < yy.shape[1]:
                    loss_t += rel_l2(im, yy[:, t_step:t_step+1])
                xx_roll = torch.cat([xx_roll[:, 1:], im], dim=1)

            loss = loss_t / 5

        else:
            # Pure teacher forcing: single-step prediction
            im = model(xx)  # [B, 1, H, W]
            loss = rel_l2(im, yy)

        loss.backward()
        if args.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        train_loss_sum += loss.item()
        n_batches += 1

    # ── validation (both single-step and autoregressive) ───────────────────────────
    model.eval()
    test_single_loss_sum = 0.0  # 单步预测误差
    test_ar_loss_sum = 0.0        # 自回归推演10步误差（在物理空间计算）
    n_test_batches = 0

    with torch.no_grad():
        for xx, xm, yy, ym in test_loader:
            xx, xm, yy, ym = xx.to(device), xm.to(device), yy.to(device), ym.to(device)

            # ===== 单步预测误差 (teacher forcing, 在归一化空间) =====
            xx_masked = xx * xm
            im_single = model(xx_masked)  # [B, 1, H, W]
            y_single = yy[:, 0:1]           # [B, 1, H, W]
            test_single_loss_sum += rel_l2(im_single, y_single).item()

            # ===== 自回归推演10步误差 (free-running, 在物理空间计算) =====
            xx_roll = xx * xm
            preds = []

            for t in range(0, T, step):
                im = model(xx_roll)        # [B, 1, H, W] - 归一化空间
                preds.append(im)
                xx_roll = torch.cat([xx_roll[:, step:], im], dim=1)

            # 拼接所有预测并反归一化到物理空间
            pred = torch.cat(preds, dim=1)  # [B, T, H, W]
            pred_phys = u_normalizer.decode(pred)  # [B, T, H, W] - 保持在GPU上
            gt_phys = u_normalizer.decode(yy)      # [B, T, H, W] - 保持在GPU上
            ym_gpu = ym                             # [B, T, H, W] - 已经在GPU上

            # 在物理空间计算每一步的相对L2误差
            for t in range(T):
                pred_t = pred_phys[:, t:t+1]  # [B, 1, H, W]
                gt_t = gt_phys[:, t:t+1]      # [B, 1, H, W]
                mask_t = ym_gpu[:, t:t+1]     # [B, 1, H, W]

                # 对mask区域进行屏蔽
                pred_masked = pred_t * mask_t
                gt_masked = gt_t * mask_t

                # 计算相对L2误差（只考虑有效区域）
                error = rel_l2(pred_masked, gt_masked)
                test_ar_loss_sum += error.item()

            n_test_batches += 1

    train_loss = train_loss_sum / n_batches
    test_single_loss = test_single_loss_sum / n_test_batches      # 单步误差
    test_ar_loss = test_ar_loss_sum / (n_test_batches * T)        # 10步平均误差
    t2 = default_timer()
    cur_lr = scheduler.get_last_lr()[0]

    # Build log message with training strategy info
    if args.use_pushforward:
        msg = (f"Epoch {ep:4d} | {t2-t1:.1f}s | lr {cur_lr:.2e} | "
               f"rollout {rollout_steps}d | γ {args.temporal_gamma} | "
               f"train_relL2 {train_loss:.4e} | "
               f"test_single {test_single_loss:.4e} | "
               f"test_ar_10step {test_ar_loss:.4e}")
    elif args.use_curriculum:
        msg = (f"Epoch {ep:4d} | {t2-t1:.1f}s | lr {cur_lr:.2e} | tf {tf_ratio:.2f} | "
               f"train_relL2 {train_loss:.4e} | "
               f"test_single {test_single_loss:.4e} | "
               f"test_ar_10step {test_ar_loss:.4e}")
    else:
        msg = (f"Epoch {ep:4d} | {t2-t1:.1f}s | lr {cur_lr:.2e} | "
               f"train_relL2 {train_loss:.4e} | "
               f"test_single {test_single_loss:.4e} | "
               f"test_ar_10step {test_ar_loss:.4e}")
    log.info(msg)

# ───────────────────────────────────────────────────────────────────────────────
# Save checkpoint
# ───────────────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), ckpt_path)
log.info(f"Checkpoint saved: {ckpt_path}")
log.info(f"Total time: {default_timer() - t_global:.1f}s")
