# Vendored SHAP (modified)

This directory is a copy of the SHAP library, version 0.46.0
(https://github.com/shap/shap), MIT License, Copyright (c) 2018 Scott Lundberg.
See `LICENSE` in this directory.

Diff-SPORT imports this copy directly (`import libs.shap.shap as shap` in
`osp_utils/shap/obs2D-osp-shap.py`); the `shap` package from PyPI is **not** used
and is not listed in `requirements.txt`.

## What was modified

One method: `KernelExplainer.explain()` in `shap/explainers/_kernel.py`.

Upstream KernelSHAP assigns the Shapley kernel weight
`(M - 1) / (|C| (M - |C|))` to every coalition size `|C|` from 1 to `M - 1`.
The modified version restricts sampling to medium-sized coalitions,
`k_min <= |C| <= k_max` with `k_min = floor(0.1 M)` and `k_max = floor(0.4 M)`
(`M` = number of candidate sensor subregions): the kernel weight is set to zero
outside this window, renormalised over the window, and coalitions are enumerated
or sampled from the allowed sizes only. This is the modified kernel described in
the Methods section of the paper ("SHAP-based optimal sensor placement").

Everything else in the package is unchanged from upstream 0.46.0.
