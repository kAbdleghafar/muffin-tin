"""Uniform cubic B-splines in torch (differentiable in x and in the coefficients)."""
import torch


def _bspline_weights(t):
    """The four uniform cubic B-spline weights on an interval, t in [0, 1)."""
    t2, t3 = t * t, t * t * t
    return torch.stack([(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3],
                       dim=-1) / 6.0


class Splines(torch.nn.Module):
    """A bank of n cubic splines on [x0, x1] with uniform knots.

    cutoff=True: f, f', f'' vanish at x1 (the 3 B-splines that are non-zero at x1 are dropped)
                 and f = 0 beyond; below x0, f continues linearly (value and slope at x0).
    cutoff=False: all n_intervals + 3 B-splines; linear continuation outside [x0, x1].
    Coefficients have shape (n, size); evaluate with f(x, which) for function index `which`."""

    def __init__(self, n, x0, x1, n_intervals, cutoff=True, init=None):
        super().__init__()
        self.x0, self.x1, self.m, self.cutoff = float(x0), float(x1), int(n_intervals), cutoff
        self.h = (self.x1 - self.x0) / self.m
        self.size = self.m if cutoff else self.m + 3
        c = torch.zeros(n, self.size, dtype=torch.float64) if init is None else init.clone()
        self.coef = torch.nn.Parameter(c)

    def _full(self):
        if self.cutoff:
            return torch.cat([self.coef, self.coef.new_zeros(self.coef.shape[0], 3)], dim=1)
        return self.coef

    def _inside(self, x, which, c):
        u = ((x - self.x0) / self.h).clamp(0, self.m - 1e-9)
        k = torch.floor(u).long()
        w = _bspline_weights(u - k)
        idx = k[:, None] + torch.arange(4, device=x.device)
        return (c[which[:, None], idx] * w).sum(-1)

    def forward(self, x, which):
        c = self._full()
        y = self._inside(x, which, c)
        lo = x < self.x0
        hi = x > self.x1
        if lo.any() or (hi.any() and not self.cutoff):
            # linear continuation: value and slope at the end point (B-spline end formulas)
            v0 = (c[which, 0] + 4 * c[which, 1] + c[which, 2]) / 6
            s0 = (c[which, 2] - c[which, 0]) / (2 * self.h)
            y = torch.where(lo, v0 + s0 * (x - self.x0), y)
            if not self.cutoff:
                n = c.shape[1]
                v1 = (c[which, n - 3] + 4 * c[which, n - 2] + c[which, n - 1]) / 6
                s1 = (c[which, n - 1] - c[which, n - 3]) / (2 * self.h)
                y = torch.where(hi, v1 + s1 * (x - self.x1), y)
        if self.cutoff:
            y = torch.where(hi, torch.zeros_like(y), y)
        return y

    def curvature_penalty(self):
        """Sum of squared second differences of the coefficients (smoothness)."""
        return (torch.diff(self.coef, n=2, dim=1) ** 2).sum()
