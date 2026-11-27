"""Exact algebra checks; standard library only. Written but not run by the assistant.

Run: python3 output/algebraic_derivation/verify_algebra.py

Rounded benchmark diagnostics are NOT population hypothesis tests.
"""
from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations

TERMS = (
    (1, ("12", "13", "24", "35", "45")),
    (-1, ("12", "13", "25", "34", "45")),
    (-1, ("12", "14", "23", "35", "45")),
    (1, ("12", "14", "25", "34", "35")),
    (1, ("12", "15", "23", "34", "45")),
    (-1, ("12", "15", "24", "34", "35")),
    (1, ("13", "14", "23", "25", "45")),
    (-1, ("13", "14", "24", "25", "35")),
    (-1, ("13", "15", "23", "24", "45")),
    (1, ("13", "15", "24", "25", "34")),
    (-1, ("14", "15", "23", "25", "34")),
    (1, ("14", "15", "23", "24", "35")),
)
Monomial = tuple[int, ...]
Polynomial = dict[Monomial, int]


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for a, ca in left.items():
        for b, cb in right.items():
            key = tuple(x + y for x, y in zip(a, b))
            result[key] = result.get(key, 0) + ca * cb
    return {key: value for key, value in result.items() if value}


def verify_pentad_identity() -> None:
    """Symbolically substitute r_ij = a_i*a_j + b_i*b_j, no numerical tolerance."""
    total: Polynomial = {}
    for sign, edges in TERMS:
        product: Polynomial = {(0,) * 10: sign}
        for edge in edges:
            i, j = (int(c) - 1 for c in edge)
            a, b = [0] * 10, [0] * 10
            a[i] += 1
            a[j] += 1
            b[5 + i] += 1
            b[5 + j] += 1
            product = multiply(product, {tuple(a): 1, tuple(b): 1})
        for key, value in product.items():
            total[key] = total.get(key, 0) + value
    remaining = {key: value for key, value in total.items() if value}
    if remaining:
        raise AssertionError(f"Pentad substitution failed: {remaining}")


def pentad(entries: dict[str, Fraction]) -> Fraction:
    result = Fraction(0)
    for sign, edges in TERMS:
        product = Fraction(sign)
        for edge in edges:
            product *= entries[edge]
        result += product
    return result


def verify_simplex_identity(weights: list[Fraction]) -> dict[str, str]:
    """Check the Gram-based third-tensor norm by two exact rational expressions."""
    if any(w <= 0 for w in weights) or sum(weights) != 1:
        raise ValueError("Weights must be positive and sum to one")
    count = len(weights)
    gram_sum = Fraction(0)
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            inner = 1 / wi - 1 if i == j else Fraction(-1)
            gram_sum += wi * wj * inner**3
    formula = sum(1 / w for w in weights) - 3 * count + 2
    bound = (count - 1) * (count - 2)
    if gram_sum != formula or formula < bound:
        raise AssertionError("Simplex third-moment identity failed")
    return {"weights": str(weights), "squared_norm": str(formula), "bound": str(bound)}


def benchmark_report(values: list[str]) -> dict[str, object]:
    pairs = [f"{i}{j}" for i, j in combinations(range(1, 6), 2)]
    entries = {key: Fraction(value) for key, value in zip(pairs, values)}
    value = pentad(entries)
    return {
        "pentad_exact": str(value),
        "pentad_decimal": float(value),
        "nonzero_for_displayed_rounded_matrix": bool(value),
        "interpretation": "Nonzero excludes exact K<=3 diagonal covariance matching for this displayed matrix only; not a population test or exact minimum count.",
    }


def main() -> None:
    verify_pentad_identity()
    skewness_checks = [
        verify_simplex_identity([Fraction(1, k)] * k) for k in range(2, 6)
    ]
    skewness_checks.append(verify_simplex_identity([Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]))
    reports = {
        "van_der_Linden_rounded": benchmark_report([
            "-0.26", "-0.12", "-0.26", "-0.32", "0.31", "0.18", "0.21", "0.14", "0.14", "0.31"
        ]),
        "Busseri_Erb_rounded": benchmark_report([
            "-0.27", "-0.08", "-0.14", "-0.22", "0.31", "0.27", "0.29", "0.27", "0.22", "0.27"
        ]),
    }
    print(json.dumps({
        "symbolic_pentad_identity": "passed",
        "simplex_checks": skewness_checks,
        "rounded_benchmarks": reports,
        "scope": "No simulations, no significance tests, no global minimum-rank optimization.",
    }, indent=2))


if __name__ == "__main__":
    main()
