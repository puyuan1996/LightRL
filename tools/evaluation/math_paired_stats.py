#!/usr/bin/env python3
"""Paired base-vs-checkpoint statistics for the math RLVR runs.

Design choices and why:

1. PRIMARY TEST IS AN EXACT PAIRED SIGN-FLIP PERMUTATION TEST. Under the null
   "this checkpoint is exchangeable with base on each problem", flipping the sign
   of each problem's paired difference is the exact randomisation. Every
   per-problem difference is a multiple of 1/k (k samples per problem), so scaling
   by k makes them integers and the whole null distribution is enumerable by DP --
   no Monte Carlo error. That matters here because several cells sit within
   ~5e-3 of alpha, which is the same order as the sampling error a B=200,000
   estimate would carry: AIME2024 x Run 1 samples to .0065 where the exact value
   is .006042, and a Holm decision can turn on that gap.

2. BOOTSTRAP IS USED FOR INTERVALS ONLY, and reports BCa rather than percentile.

3. MULTIPLICITY IS CORRECTED. The confirmatory family is declared up front: the
   12 lenient Avg@k cells (4 datasets x 3 runs). Holm controls FWER, BH controls
   FDR; both are reported because they disagree here, and that disagreement is
   itself part of the result.

4. POWER IS REPORTED. "Not significant" carries almost no information at 30
   problems x 1 seed, so every cell carries MDE80, a two-sided CI, and the count
   of problems with a non-zero paired difference (only those carry information,
   and they set each cell's p-value resolution floor).

MATH-500 is restricted to the 349 problems whose ground truth compute_score_dapo
can grade; see per_problem() for why.

Usage:  python paired_stats.py
        RES_DIR=/path python paired_stats.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(os.environ.get("RES_DIR", Path(__file__).resolve().parent / "eval_results"))
RNG = np.random.default_rng(20260801)
B_BOOT = 20_000

# (stem, label, k). MATH-500 is k=4: 500 problems x 16 samples was too many
# generations to be worth it, and 500 problems already dominates the precision.
DATASETS = [("aime-2025", "AIME2025", 16), ("aime-2024", "AIME2024", 16),
            ("amc23", "AMC23", 16), ("math-500", "MATH-500", 4)]
RUNS = [("iter159", "Run 1"), ("a25_iter19", "Run 2"), ("run3_iter19", "Run 3")]
BASE = "T1.0"


def load(stem: str, tag: str, n: int) -> list[dict]:
    for name in (f"{stem}_{tag}_n{n}.detail.json", f"20_detail_{stem}_{tag}_n{n}.json"):
        p = RES / name
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"{stem}/{tag}/n{n} under {RES}")


# MATH-500's stored acc_lenient is not format-neutral. compute_score_dapo needs an
# integer-parseable ground truth; on the 151/500 problems where it is not, the
# except branch hard-sets acc=False, so "strict OR boxed" silently degrades to
# boxed-only -- which penalises exactly the checkpoints that emit `Answer:` rather
# than \boxed{}. Reported effect on that cell: -19.60pp, actual ~0.
#
# The fix keeps compute_score_dapo everywhere (all four datasets must be graded by
# the SAME verifier -- a report arguing that verifier definitions must match cannot
# itself mix two) and restricts MATH-500 to the 349 problems where it can run.
# math500_neutral_check.py cross-checks this against a format-neutral verifier on all 500;
# the two agree in direction and magnitude.
# Accept both the working-tree name and the gist bundle's numbered name.
_FIXED = next((RES / n for n in ("math500_gradeable_subset.json",
                                 "13_math500_gradeable_subset.json")
               if (RES / n).exists()), RES / "math500_gradeable_subset.json")
FIXED_LENIENT = json.loads(_FIXED.read_text()) if _FIXED.exists() else {}


def per_problem(d: list[dict], key: str, stem: str = "", tag: str = "") -> np.ndarray:
    if key == "acc_lenient" and stem == "math-500":
        if tag not in FIXED_LENIENT:
            # Falling back to the stored acc_lenient would silently swap in the
            # all-500 population, whose lenient track is not format-neutral (see
            # above), and still exit 0 -- a reader on an incomplete copy would
            # believe they had reproduced the report. Refuse instead.
            raise SystemExit(
                f"missing {_FIXED}: MATH-500 must use the 349-problem gradeable\n"
                f"subset (see the report's section-2 footnote). Without it this script\n"
                f"would silently fall back to the non-format-neutral all-500 track.\n"
                f"Regenerate with\n"
                f"  (it ships in the evidence bundle as 13_math500_gradeable_subset.json)\n"
                f"or copy math500_gradeable_subset.json next to the data.")
        return np.array(FIXED_LENIENT[tag])
    return np.array([np.mean([s[key] for s in p["samples"]]) for p in d])


def perm_p(diff: np.ndarray, k: int) -> float:
    """EXACT two-sided paired sign-flip p, by DP over the integer null distribution.

    Every per-problem difference is a multiple of 1/k (k samples per problem), so
    scaling by k makes them integers and the entire null distribution of
    sum(+/-d_i) is enumerable -- no Monte Carlo error.

    Sampling would not be good enough here: several cells sit within ~5e-3 of
    alpha, the same order as the Monte Carlo error of a B=200,000 estimate.

    Zero differences are invariant under sign flip -- they scale every branch by 2
    and cancel out of the ratio, so they are dropped.
    """
    scaled = np.asarray(diff, dtype=float) * k
    if not np.allclose(scaled, np.rint(scaled)):
        # Silently quantising off-grid input would return a meaningless p.
        raise ValueError(f"paired differences are not multiples of 1/{k}; "
                         "perm_p's exact enumeration does not apply")
    d = np.rint(scaled).astype(np.int64)
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    obs = int(abs(nz.sum()))
    off = int(np.abs(nz).sum())
    # Halve at each step to keep the distribution normalised; float64 keeps ~15
    # significant digits, far below the smallest p any of these cells reaches.
    dist = np.zeros(2 * off + 1)
    dist[off] = 1.0
    for v in np.abs(nz):
        nxt = np.zeros_like(dist)
        nxt[v:] += dist[:dist.size - v]
        nxt[:dist.size - v] += dist[v:]
        dist = nxt * 0.5
    if obs == 0:
        return 1.0
    return float(dist[off + obs:].sum() + dist[:off - obs + 1].sum())


def bca_ci(diff: np.ndarray, alpha: float = 0.05, b: int = B_BOOT) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI for the mean paired difference."""
    n = diff.size
    theta = diff.mean()
    boots = diff[RNG.integers(0, n, size=(b, n))].mean(axis=1)
    prop = np.mean(boots < theta)
    prop = min(max(prop, 1 / b), 1 - 1 / b)  # keep ppf finite
    z0 = stats.norm.ppf(prop)
    # jackknife acceleration
    jk = (diff.sum() - diff) / (n - 1)
    jkbar = jk.mean()
    num = np.sum((jkbar - jk) ** 3)
    den = 6.0 * (np.sum((jkbar - jk) ** 2) ** 1.5)
    a = num / den if den else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(boots, 100 * stats.norm.cdf(adj))))
    return out[0], out[1]


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def bh(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)[::-1]
    adj = np.empty(m)
    run = 1.0
    for i, idx in enumerate(order):
        run = min(run, m / (m - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def mde80(diff: np.ndarray) -> float:
    """Smallest true effect detected 80% of the time by a paired z-test, alpha=.05.

    NORMAL quantiles, not t: at the small n this is applied to (down to 6) the
    t-based value is 7-25% larger, so this is an optimistic bound. It is also the
    MDE of a z-test, while the p-values reported beside it come from the exact
    sign-flip permutation test -- the two differ by -20% to +35% depending on the
    cell. Read it as an order-of-magnitude statement about resolution, not as the
    power of the test whose p-value sits next to it.

    Being computed from the observed sd, it is a post-hoc MDE: at n=6 that sd
    carries a 0.62x-2.45x CI of its own.
    """
    n = diff.size
    sd = diff.std(ddof=1)
    return float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd / np.sqrt(n))


def main() -> None:
    cells = []
    for stem, label, k in DATASETS:
        base = per_problem(load(stem, BASE, k), "acc_lenient", stem, BASE)
        for tag, run in RUNS:
            ck = per_problem(load(stem, tag, k), "acc_lenient", stem, tag)
            d = ck - base
            lo, hi = bca_ci(d)
            lo90, hi90 = bca_ci(d, alpha=0.10)
            cells.append(dict(
                dataset=label, k=k, run=run, tag=tag, n_problems=int(d.size),
                base=100 * base.mean(), ckpt=100 * ck.mean(), delta=100 * d.mean(),
                ci95=[100 * lo, 100 * hi], ci90=[100 * lo90, 100 * hi90],
                p_perm=perm_p(d, k), mde80=100 * mde80(d),
                # Only problems with a non-zero paired difference carry
                # information; a cell with few of them has a coarse p-value floor.
                n_informative=int(np.count_nonzero(np.rint(d * k))),
                p_floor=2.0 ** (1 - max(1, int(np.count_nonzero(np.rint(d * k)))))))

    ps = [c["p_perm"] for c in cells]
    for c, h, b in zip(cells, holm(ps), bh(ps)):
        c["p_holm"], c["q_bh"] = h, b

    print(f"CONFIRMATORY FAMILY: {len(cells)} lenient Avg@k cells (4 datasets x 3 runs)")
    print(f"permutation: EXACT (full sign-flip enumeration by DP), BCa bootstrap B={B_BOOT:,}\n")
    hdr = f"{'dataset':<10}{'run':<7}{'Δpp':>8}{'95% CI':>18}{'p_perm':>9}{'p_holm':>8}{'q_BH':>8}{'MDE80':>7}"
    print(hdr + "\n" + "-" * len(hdr))
    for c in cells:
        star = "*" if c["q_bh"] < 0.05 else " "
        star += "H" if c["p_holm"] < 0.05 else " "
        ci = "[%+.2f,%+.2f]" % (c["ci95"][0], c["ci95"][1])
        print(f"{c['dataset']:<10}{c['run']:<7}{c['delta']:+8.2f}{ci:>18}"
              f"{c['p_perm']:>10.6f}{c['p_holm']:>8.3f}{c['q_bh']:>8.3f}{c['mde80']:>7.2f} {star}")
    print("\n* = survives BH(FDR .05)   H = survives Holm(FWER .05)")

    # Direct run-vs-run contrasts. The issue argued "verifier flips the outcome"
    # and "training-set size does not matter" from two separate vs-base tests,
    # which is not a test of the difference. These are.
    print("\nDIRECT CONTRASTS (report section 5.3)")
    contrasts = [("run3_iter19", "iter159", "Run3 - Run1  (verifier+dyn)"),
                 ("run3_iter19", "a25_iter19", "Run3 - Run2  (train set)")]
    for stem, label, k in DATASETS:
        for a_tag, b_tag, name in contrasts:
            a = per_problem(load(stem, a_tag, k), "acc_lenient", stem, a_tag)
            b = per_problem(load(stem, b_tag, k), "acc_lenient", stem, b_tag)
            d = a - b
            lo, hi = bca_ci(d)
            print(f"  {label:<10}{name:<28}{100*d.mean():+7.2f}pp  "
                  f"95% CI [{100*lo:+6.2f},{100*hi:+6.2f}]  p={perm_p(d, k):.6f}")

    # Write beside the data, not beside the script: in the gist bundle there is
    # no eval_results/ and the old hardcoded path made the script exit 1 after
    # having already printed the table -- which read as "it works".
    out = RES / "paired_stats.json"
    out.write_text(json.dumps(cells, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
