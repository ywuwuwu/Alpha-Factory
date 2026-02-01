from __future__ import annotations

import numpy as np


def project_to_l1_ball(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection of v onto the L1-ball of radius z (nonnegative version).

    We use a simple projection onto the simplex (v>=0, sum=v=z) after clipping negatives,
    which is enough for an 'alpha weight' use-case where weights are nonnegative.
    """
    v = np.maximum(v, 0.0)
    if v.sum() <= z:
        return v
    # projection onto simplex: https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf (conceptually)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - z))[0]
    if len(rho) == 0:
        return np.full_like(v, z / len(v))
    rho = rho[-1]
    theta = (cssv[rho] - z) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


class OnlineALMAllocator:
    """A simple turnover-aware online allocator for combining factor scores.

    This is intentionally lightweight and interview-friendly:
      - uses recent factor quality signal (e.g., train IC mean)
      - updates weights with learning rate eta
      - projects onto an L1 budget (simplex-like)
      - mixes with previous weights to control turnover
    """

    def __init__(self, n: int, l1_budget: float = 1.0, eta: float = 0.1, tau: float = 0.1):
        self.n = int(n)
        self.l1_budget = float(l1_budget)
        self.eta = float(eta)
        self.tau = float(tau)
        self.w = np.full(self.n, self.l1_budget / self.n, dtype=float)

    def step(self, quality: np.ndarray) -> np.ndarray:
        quality = np.asarray(quality, dtype=float)
        assert quality.shape == (self.n,)
        # gradient-ascent-ish step
        proposal = self.w + self.eta * quality
        proposal = project_to_l1_ball(proposal, self.l1_budget)
        # turnover control: exponential smoothing toward proposal
        self.w = (1.0 - self.tau) * self.w + self.tau * proposal
        # re-project (numerical)
        self.w = project_to_l1_ball(self.w, self.l1_budget)
        return self.w.copy()
