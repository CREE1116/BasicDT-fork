# BasicDT — Changes vs Original

Summary of all changes in this fork relative to the original `BasicDT`
(`/Users/leejongmin/code/BasicDT`, base commit `bd96670`). Goal of the fork:
turn BasicDT into a **complete, standard, well-rounded GBDT** by adding the
proven techniques of XGBoost / LightGBM / CatBoost that the original lacked,
plus a distinctive unified-path explainer.

---

## 1. New standard hyperparameters

All added to `BasicDTClassifier.__init__` and threaded into the C++ engine.
Defaults are chosen so existing behavior is unchanged unless a knob is set.

| Param | Source idiom | Was | Now |
| :-- | :-- | :-- | :-- |
| `reg_alpha` | XGB/LGBM | — (L2 only) | L1 soft-threshold on the gradient, applied **consistently** in split gain and leaf Newton step (`thr_l1`) |
| `gamma` | XGB/LGBM | split if gain>0 | minimum split gain threshold |
| `min_child_weight` | XGB/LGBM | hardcoded `0.1` | exposed parameter (default 1.0) |
| `max_leaves` | LGBM | hardcoded `2^depth` | leaf-wise leaf budget |
| `max_bin` | XGB/LGBM | hardcoded `AX_BINS=256` | **runtime** bin count, engine-level (see §4) |
| `colsample_bynode` | XGB/LGBM | — | per-node feature subsampling (node-seeded RNG, deterministic) |
| `goss` / `goss_top_rate` / `goss_other_rate` | LGBM | — | Gradient-based One-Side Sampling (see §5) |
| `multi_strategy` | — | shared only | `"shared"` (fast shared tree, **default**) or `"ovr"` (standard K-trees); see §3 |

Also added: **learned missing-value direction** (`default_left`) per split —
XGB-style for **both numeric and categorical** features. The original imputed
missing values (numeric → column mean; categorical → treated as a ranked
category). Now missing rows are excluded from the split statistics, tried on
both sides of each candidate split, and routed by the higher-gain default
direction (sentinel bin `n_bins-1`); at predict, NaN **and unseen categories**
follow that default. Unifies numeric/categorical missing handling.

---

## 2. Bug fix — `sample_weight` was silently ignored

`fit(..., sample_weight=...)` accepted the argument but `_fit_core` never used
it (a silent no-op, violating the sklearn contract). Now each boosting round
scales the per-sample gradient/hessian by the weight (XGB semantics). Verified:
all-ones weight == unweighted (bit-identical), zero-weighting a class removes it
from predictions, length mismatch raises.

---

## 3. Multiclass architecture

- **Original:** one *shared* tree per round with K-vector leaves for all K
  classes (split chosen by gain summed over classes).
- **`multi_strategy="ovr"`:** standard one-vs-rest — K separate trees per round
  (like XGBoost/LightGBM). K=2 still uses the binary fast path.
- **`multi_strategy="shared"` (current default):** keeps the original
  shared-tree design. It is **~2.3x faster at low K** (1 histogram pass vs K)
  with equal accuracy, because the structure search is paid once. Profiling
  showed the OVR slowdown is inherent (K× histogram passes), not Python
  overhead (which is ~6%). Chosen as the default in the Fast profile (§8).

`predict_proba`, `explain`, and serialization all branch on `self._is_separate_`.

---

## 4. Engine (C++) changes — `basicdt.cpp` / `basicdt_types.h`

- **Runtime `max_bin`:** `BasicDTCtx` gained `n_bins`; `basicdt_ctx_create` takes
  a `max_bin` arg (clamped to `[2,256]`, fits the `uint8` bin codes). Inside
  `basicdt_ctx_create` and `basicdt_build`, a local `const int AX_BINS = ctx->n_bins;`
  **shadows** the compile-time constant, so all 15+ histogram/scan/threshold
  sites use the runtime value with minimal diff. NaN sentinel = `n_bins-1`,
  normal values clamp to `n_bins-2` (no collision). Predict routes on stored
  thresholds, so it is unaffected.
- **`thr_l1(g, alpha)`** helper for L1 (reg_alpha), used in every gain term and
  the leaf weight so split scoring and leaf output stay consistent.
- **`default_left`** field on `BasicDTTree`; learned at split time, honored in
  single-tree and ensemble predict (with `continue` so missing rows are not
  double-routed).
- **`basicdt_build` signature** extended: `max_leaves, reg_lambda, colsample,
  col_seed, gamma, min_child_weight, reg_alpha`.
- **`basicdt_export`** now also exports `default_left`; new **`basicdt_export_gain`**
  exports per-node split gain (additive, serialization ABI unchanged).
- **Perf cleanup:** the dominant-class (`kdom`) scan is skipped entirely when
  there are no categorical features (it only feeds categorical target-rank
  re-encoding) — pure waste removed for numeric data.

---

## 5. GOSS (Gradient-based One-Side Sampling)

LightGBM's signature sampling, implemented in Python on top of the fixed
`sample_weight` + existing subsample machinery (no C++ change). Each round:
keep all top-`goss_top_rate` |gradient| rows, randomly sample `goss_other_rate`
of the rest, and rescale the sampled rows by `(1-top)/other` to keep gain
estimates unbiased. Fewer rows per tree (faster) while focusing on hard
examples. Replaces plain Bernoulli `subsample` when `goss=True`.

---

## 6. Interpretability — new `explain.py` (distinctive feature)

- `clf.feature_importances_` — one **unified gain importance** over all classes.
- `clf.explain(x)` / `explain.explain_prediction` — per-feature contribution to
  **every** class logit, decomposed along the decision path(s), gain-weighted.
  Contributions are **conservative** (sum to the logit shift over base scores,
  verified to ~5e-7); path reconstruction is cross-checked against the engine
  (`exact_paths`). Works for both `ovr` and `shared`.
- Backed by `BasicDTree.export_arrays()` + C++ `basicdt_export_gain`.

---

## 7. Measured performance (K=3, 120k×12, 200 rounds, shared)

| config | time | acc |
| :-- | :-- | :-- |
| baseline (bin256, subsample 0.8) | 2.02s | 0.9712 |
| `max_bin=128` | 1.88s | 0.9716 |
| `goss=True` | 1.75s | 0.9707 |
| `goss=True, max_bin=128` | **1.58s** | 0.9708 |

GOSS (fewer rows → cheaper histogram accumulate) and `max_bin` (fewer bins →
cheaper scan) attack independent costs and **compound**: **22% faster at equal
accuracy.** `multi_strategy="shared"` is a further ~2.3x over `ovr` at low K.

---

## 8. Default profile — "Fast"

Defaults were retuned for speed (user choice). All knobs remain overridable.

| param | original | now |
| :-- | :-- | :-- |
| `n_estimators` | 1000 | 300 |
| `learning_rate` | 0.03 | 0.08 |
| `max_bin` | 256 | 64 |
| `multi_strategy` | (shared only) | `"shared"` |
| `goss` | (n/a) | `True` |

Measured (K=3, 120k): new defaults **2.1s @ 0.9563** vs old-style accurate
defaults (n=1000, lr=0.03, bin256, ovr) **18.4s @ 0.9597** — ~8.7x faster for
~0.3pt accuracy. For max accuracy instead: `n_estimators=1000, learning_rate=0.03,
max_bin=256, multi_strategy="ovr", goss=False`.

## 9. Misc

- `pyproject.toml`: `requires-python` `>=3.10` → `>=3.9` (code is 3.9-safe via
  `from __future__ import annotations`).
- New tests: `tests/test_explain.py`, `tests/test_regularization.py`
  (+ additions to `tests/test_classifier.py`). All pass.
- Benchmark/validation scripts at repo root: `bench_manyclass.py`,
  `bench_colsample.py`, `demo_explain.py`.

> Note: the StandardScaler removal from the `stellar` model wrappers
> (`OQBoost`/`BasicDT`/`ICST`) lives in the separate `stellar` repo, not here.
