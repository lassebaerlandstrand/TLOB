"""Q-target generation for the DP-Distilled Value Network (DPVN).

For each timestep t and each action a in {-1, 0, +1}, compute

    V_target[t, a] = a * (z_mid[t+h] - z_mid[t])
                   - |a - pos_DP[t-1]| * z_half_spread[t]
                   + gamma * tail_V_DP[t+h]

where pos_DP is the DP-optimal position trajectory and tail_V_DP[t] is the DP
cumulative net PnL from t onward. This is a truncated-horizon Q-function using
the DP trajectory as a teacher-forced oracle: the immediate reward is causal
(h-step hold cost), while the tail uses DP to bootstrap.

Respects segment boundaries (Battery per-product) — each segment is its own
DP problem and no value flows across borders.
"""
from __future__ import annotations

import numpy as np

from utils.metrics import compute_dp_optimal


def compute_q_targets(
    z_mid: np.ndarray,
    z_half_spread: np.ndarray,
    horizon: int,
    gamma: float = 1.0,
    segment_boundaries: np.ndarray | None = None,
    mean_center: bool = True,
) -> np.ndarray:
    """Return (N, 3) array of Q targets for actions {-1, 0, +1}.

    Beyond the last valid timestep (t > N - horizon) or across a segment
    boundary, targets fall back to the immediate return with no tail.

    When mean_center=True, subtract the per-sample mean across the action axis.
    The argmax decision rule is invariant under this shift, but the action-
    dependent signal dominates the target (SNR gain ~100x on LOB data where
    tail values vastly exceed h-step return magnitudes).
    """
    z_mid = np.asarray(z_mid, dtype=np.float64)
    z_half_spread = np.asarray(np.abs(z_half_spread), dtype=np.float64)
    n = len(z_mid)
    assert len(z_half_spread) == n, "z_mid and z_half_spread must be same length"

    if segment_boundaries is not None:
        boundaries = np.asarray(segment_boundaries, dtype=np.int64)
        segments = []
        prev = 0
        for b in boundaries:
            if b > prev:
                segments.append((prev, b))
            prev = b
        if prev < n:
            segments.append((prev, n))
    else:
        segments = [(0, n)]

    q_targets = np.zeros((n, 3), dtype=np.float32)
    pos_vals = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

    for start, end in segments:
        seg_len = end - start
        if seg_len < 2:
            continue
        mid = z_mid[start:end]
        hs = z_half_spread[start:end]

        dp_result = compute_dp_optimal(mid, hs)
        pos_DP = dp_result["positions"]  # length seg_len, pos_DP[-1] = 0 (force close)

        pos_prev = np.zeros(seg_len, dtype=np.float64)
        pos_prev[1:] = pos_DP[:-1]

        gross_DP = np.zeros(seg_len, dtype=np.float64)
        gross_DP[:-1] = pos_DP[:-1] * np.diff(mid)

        cost_DP = np.zeros(seg_len, dtype=np.float64)
        cost_DP[1:] = np.abs(np.diff(pos_DP)) * hs[1:]
        cost_DP[0] = np.abs(pos_DP[0]) * hs[0]

        net_DP = gross_DP - cost_DP
        tail_V_DP = np.zeros(seg_len + 1, dtype=np.float64)
        tail_V_DP[:-1] = np.cumsum(net_DP[::-1])[::-1]

        for a_idx in range(3):
            a = pos_vals[a_idx]
            t = np.arange(seg_len, dtype=np.int64)
            future_idx = np.minimum(t + horizon, seg_len - 1)
            r_t = a * (mid[future_idx] - mid)
            entry_cost = np.abs(a - pos_prev) * hs
            tail = gamma * tail_V_DP[future_idx]
            valid_tail = (t + horizon) <= (seg_len - 1)
            tail = np.where(valid_tail, tail, 0.0)

            q_seg = r_t - entry_cost + tail
            q_targets[start:end, a_idx] = q_seg.astype(np.float32)

    if mean_center:
        q_targets -= q_targets.mean(axis=1, keepdims=True)

    return q_targets


def extract_z_mid_z_half_spread(input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull z_mid and z_half_spread from LOB columns 0 (ask_price L0) and 2 (bid_price L0).

    Matches engine.test_step's extraction convention: `x[:,-1,0]` = ask, `x[:,-1,2]` = bid.
    Both BTC and Battery LOB-only layouts have ask/bid at indices 0/2 after loading.
    """
    col0 = np.asarray(input_tensor[:, 0], dtype=np.float64)
    col2 = np.asarray(input_tensor[:, 2], dtype=np.float64)
    z_mid = (col0 + col2) / 2.0
    z_half_spread = np.abs(col0 - col2) / 2.0
    return z_mid, z_half_spread
