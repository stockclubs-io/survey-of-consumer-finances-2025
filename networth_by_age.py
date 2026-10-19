#!/usr/bin/env python3
"""
Compute weighted net worth percentiles by age group from the SCF 2025 dataset.

Uses the SCF sample weight (WGT) for weighted percentiles, since survey
respondents are not uniformly representative of the population.
"""
import pandas as pd
import numpy as np

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


def weighted_percentile(
        values: np.ndarray,
        weights: np.ndarray,
        percentiles: list[int]
) -> list[float]:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    cutoffs = (cum_weights - 0.5 * weights) / weights.sum()
    return list(np.interp(np.array(percentiles) / 100.0, cutoffs, values))


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=["AGE", "NETWORTH", "WGT"])

    header = f"{'Age Group':<12}" + "".join(f"{'p' + str(p):>15}" for p in PERCENTILES)
    print(header)
    print("-" * len(header))

    for low, high in AGE_GROUPS:
        subset = df[(df["AGE"] >= low) & (df["AGE"] <= high)]
        if subset.empty:
            print(f"{low}-{high:<9}" + "".join(f"{'n/a':>15}" for _ in PERCENTILES))
            continue

        values = weighted_percentile(
            subset["NETWORTH"].to_numpy(dtype=float),
            subset["WGT"].to_numpy(dtype=float),
            PERCENTILES,
        )
        row = f"{f'{low}-{high}':<12}" + "".join(f"{v:>15,.0f}" for v in values)
        print(row)


if __name__ == "__main__":
    main()
