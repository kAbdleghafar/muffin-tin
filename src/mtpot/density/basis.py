"""Uniform cubic B-spline radial basis that vanishes smoothly (C2) at the cutoff."""
import numpy as np
from scipy.interpolate import BSpline


class RadialBasis:
    """f(r) = sum_j c_j B_j(r) on [r0, rc]. Only B-splines with support ending at or before rc
    are kept, so f, f' and f'' are zero at rc; f is free (not forced to zero) at r0."""

    def __init__(self, r0, rc, n_intervals):
        self.r0, self.rc, self.n = float(r0), float(rc), int(n_intervals)
        self.h = (self.rc - self.r0) / self.n
        self.knots = self.r0 + self.h * np.arange(-3, self.n + 4)
        self.size = self.n                      # drops the 3 splines that are non-zero at rc

    def design(self, r):
        """Sparse (len(r), size) matrix of basis values; r must lie in [r0, rc]."""
        r = np.clip(np.asarray(r, float), self.r0, self.rc)
        B = BSpline.design_matrix(r, self.knots, 3).tocsc()
        return B[:, :self.size]

    def __call__(self, coef, r):
        """f(r) for one coefficient vector; zero beyond rc, held at f(r0) below r0."""
        r = np.asarray(r, float)
        out = np.asarray(self.design(r) @ coef).ravel()
        return np.where(r < self.rc, out, 0.0)

    def penalty(self):
        """Second-difference penalty matrix D^T D (P-spline smoothness)."""
        D = np.diff(np.eye(self.size), n=2, axis=0)
        return D.T @ D
