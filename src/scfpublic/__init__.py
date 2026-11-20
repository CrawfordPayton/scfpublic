"""
scfpublic
=================

A package for loading the Survey of Consumer Finances and properly applying multiple imputation and sampling variances.

Author: Payton Crawford
"""

from .scfpublic import download_year, download_years, download_main_extr_years, _get_url, _get_dta_path, _match_vars, _optimize_dtypes, minYear, maxYear, YEARS
from .scfcombo import scfcombo_result, prepare_scf_data, scf_estimate, scf_mean, scf_quantile, scf_regression, scf_total
from .cpi import read_excel_from_url, cpi_adjust
from .util import mconv, aconv

__version__ = "0.1.0"
__all__ = [
    "download_year", 
    "download_years", 
    "download_main_extr_years",
    "_get_url", 
    "_get_dta_path", 
    "_match_vars",
    "_optimize_dtypes",
    "minYear", 
    "maxYear", 
    "YEARS",
    "scfcombo_result", 
    "prepare_scf_data", 
    "scf_estimate", 
    "scf_mean", 
    "scf_quantile", 
    "scf_regression",
    "scf_total",
    "read_excel_from_url", 
    "cpi_adjust",
    "mconv",
    "aconv",
]
