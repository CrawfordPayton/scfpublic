"""
scfpublic.scfcombo
=================

A package for properly correcting for multiple imputation and sampling variances.

Author: Payton Crawford
"""

from .scfcombo import scfcombo_result, prepare_scf_data, scf_estimate
from .stats import scf_mean, scf_quantile, scf_regression, scf_total

__all__ = [
    "scfcombo_result",
    "prepare_scf_data",
    "scf_estimate",
    "scf_mean",
    "scf_quantile",
    "scf_regression",
    "scf_total",
]