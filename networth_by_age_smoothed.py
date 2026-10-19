#!/usr/bin/env python3
"""
Smoothed net worth percentiles by age, using weighted quantile regression.

Instead of computing each percentile independently within noisy 5-year age
bins (see networth_by_age.py), this fits a weighted quantile regression of
NETWORTH on a cubic B-spline basis of AGE for each target percentile, using
every respondent (weighted by WGT) rather than just the respondents that
happen to fall in one bin. The fitted curves are then evaluated at the
midpoint of each age group, which borrows statistical strength across
nearby ages and smooths out small-sample noise in the tails (p95, p99).

Quantile regression is solved directly as a weighted pinball-loss linear
program (scipy.optimize.linprog), since no quantile-regression package is
installed in this project's virtualenv.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import linprog
from scipy.sparse import eye as sparse_eye
from scipy.sparse import hstack as sparse_hstack

CSV_PATH = "scripts/scfp2025/SCFP2025.csv"

AGE_GROUPS = [
    (30, 34),
    (35, 39),
    (40, 44),
    (45, 49),
    (50, 54),
    (55, 59),
    (60, 64),
    (65, 69),
    (70, 74),
    (75, 80),
]

PERCENTILES = [25, 50, 75, 90, 95, 99]

SPLINE_DEGREE = 3
INTERIOR_KNOT_QUANTILES = [20, 40, 60, 80]


def build_spline_basis(age: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    full_knots = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_basis = len(full_knots) - degree - 1
    basis = np.empty((len(age), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        basis[:, i] = BSpline(full_knots, coeffs, degree, extrapolate=False)(age)
    return basis


def fit_weighted_quantile_regression(X: np.ndarray, y: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
    n, p = X.shape
    # beta split into nonnegative parts (beta = beta_pos - beta_neg) so all LP
    # variables are >= 0; residual also split into positive/negative parts.
    A_eq = sparse_hstack([X, -X, sparse_eye(n), -sparse_eye(n)]).tocsc()
    b_eq = y
    c = np.concatenate([
        np.zeros(2 * p),
        weights * tau,
        weights * (1.0 - tau),
    ])
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"quantile regression failed for tau={tau}: {result.message}")
    beta_pos, beta_neg = result.x[:p], result.x[p:2 * p]
    return beta_pos - beta_neg


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=["AGE", "NETWORTH", "WGT"])

    age = df["AGE"].to_numpy(dtype=float)
    networth = df["NETWORTH"].to_numpy(dtype=float)
    weights = df["WGT"].to_numpy(dtype=float)

    interior_knots = np.percentile(age, INTERIOR_KNOT_QUANTILES)
    knots = np.concatenate([[age.min()], interior_knots, [age.max()]])
    X = build_spline_basis(age, knots, SPLINE_DEGREE)

    midpoints = np.array([(low + high) / 2.0 for low, high in AGE_GROUPS])
    X_pred = build_spline_basis(midpoints, knots, SPLINE_DEGREE)

    predictions = np.empty((len(AGE_GROUPS), len(PERCENTILES)))
    for col, p in enumerate(PERCENTILES):
        beta = fit_weighted_quantile_regression(X, networth, weights, tau=p / 100.0)
        predictions[:, col] = X_pred @ beta

    # Quantile regression curves are fit independently per percentile and can
    # cross; enforce p25 <= p50 <= ... <= p99 at each age via sorting
    # (the standard rearrangement fix for quantile crossing).
    predictions = np.sort(predictions, axis=1)
    predictions = np.round(predictions / 1000.0) * 1000.0

    header = f"{'Age Group':<12}" + "".join(f"{'p' + str(p):>15}" for p in PERCENTILES)
    print(header)
    print("-" * len(header))
    for (low, high), row in zip(AGE_GROUPS, predictions):
        line = f"{f'{low}-{high}':<12}" + "".join(f"{v:>15,.0f}" for v in row)
        print(line)


if __name__ == "__main__":
    main()
