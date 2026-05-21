"""R1 disproof experiment: GBM baseline on aggregated [6] features per complex.

Train sklearn.ensemble.GradientBoostingRegressor on (mean, std, min, max) of the
6 channels per complex -> affinity. Report val Pearson r.

If r >= 0.3 the binding signal IS in our features -> safe to proceed.
If r <  0.1 the features are too thin -> add channels 7-8 (H-bonds, contact entropy).

Stub: implement during hour 2-4.
"""
# TODO(hour 2-4): implement
