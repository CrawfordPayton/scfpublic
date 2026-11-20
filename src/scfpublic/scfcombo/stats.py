"""
Convenience summary-statistic wrappers around scf_estimate().

Author: Payton Crawford
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import warnings

from .scfcombo import scfcombo_result, scf_estimate

def scf_mean(
    data: Dict[str, pd.DataFrame],
    variable: str,
    subset: Optional[Callable[[pd.DataFrame], pd.Series]] = None
) -> scfcombo_result:
    """
    Compute weighted mean with full variance estimation.
    
    This is a convenience wrapper around scf_estimate() for computing means.
    
    Args:
        data (dict): Dictionary from prepare_scf_data()
        variable (str): Name of variable to compute mean for
        subset (function): Optional function for subsetting data
        
    Returns:
        scfcombo_result object
        
    Example:
        >>> result = scf_mean(scf_data, 'networth')
        >>> result = scf_mean(scf_data, 'income', subset=lambda df: df['age'] > 25)
    """
    def weighted_mean(data: pd.DataFrame, weights: np.ndarray, variable: str) -> float:
        notmissing = data[variable].notna()
        if not notmissing.any():
            return np.nan
        return np.average(data.loc[notmissing, variable].values, weights=weights[notmissing])

    result = scf_estimate(data, weighted_mean, subset=subset, variable=variable)
    
    # Add display names
    result.statistic_name = 'Mean'
    result.variable_name = variable
    
    return result


def scf_quantile(
    data: Dict[str, pd.DataFrame],
    variable: str,
    quantile: float = 0.5,
    subset: Optional[Callable[[pd.DataFrame], pd.Series]] = None
) -> scfcombo_result:
    """
    Compute weighted quantile with full variance estimation.
    
    This is a convenience wrapper around scf_estimate() for computing quantiles.
    
    Args:
        data (dict): Dictionary from prepare_scf_data()
        variable (str): Name of variable to compute quantile for
        quantile (float): Quantile to compute (0.5 for median)
        subset (function): Optional function for subsetting data
        
    Returns:
       scfcombo_result object
        
    Example:
        >>> result = scf_quantile(scf_data, 'networth', quantile=0.5)  # Median
        >>> result = scf_quantile(scf_data, 'income', quantile=0.9)   # 90th percentile
    """
    try:
        from statsmodels.stats.weightstats import DescrStatsW
    except ImportError:
        raise ImportError(
            "statsmodels required for quantiles. "
            "Install with: pip install statsmodels"
        )

    def weighted_quantile(
        data: pd.DataFrame,
        weights: np.ndarray,
        variable: str,
        q: float
    ) -> float:
        notmissing = data[variable].notna()
        if not notmissing.any():
            return np.nan
        
        weighted_stats = DescrStatsW(data.loc[notmissing, variable].values, weights=weights[notmissing])
        return float(weighted_stats.quantile(probs=[q]).iloc[0])

    result = scf_estimate(
        data,
        weighted_quantile,
        subset=subset,
        variable=variable,
        q=quantile
    )
    
    # Add display names
    if quantile == 0.5:
        result.statistic_name = 'Median'
    else:
        result.statistic_name = f'{quantile*100:.0f}th Percentile'
    result.variable_name = variable
    
    return result

def scf_regression(
    data: Dict[str, pd.DataFrame],
    formula: str,
    model_type: str = 'ols',
    subset: Optional[Callable[[pd.DataFrame], pd.Series]] = None
) -> scfcombo_result:
    """
    Estimate regression model with SCF variance correction.
    
    This function properly handles survey weights for regression estimation:
    
    Weight Usage:
        
        1. Point Estimates (Coefficients):
           - For each implicate i: Run regression using main weight (wgt)
           - Pool coefficients: beta_hat = (1/5) * sum(beta_hat{i})
           
        2. Imputation Variance:
           - Variance of coefficients across 5 implicates
           
        3. Sampling Variance:
           - For each replicate r: Run regression on first implicate using 
             replicate weight r (wgt1, wgt2, ..., wgt999)
           - Variance of coefficients across replicates
    
    Args:
        data (dict): Dictionary from prepare_scf_data()
        formula (str): Regression formula in statsmodels format
        model_type (str): Type of model ('ols', 'logit', 'probit', 'poisson')
        subset (function): Optional function for subsetting data
        
    Returns:
        scfcombo_result object with coefficient estimates and standard errors
    
    Example:
        >>> result = scf_regression(
        ...     scf_data,
        ...     'networth ~ age + income + C(education)',
        ...     model_type='ols'
        ... )
        >>> result.display()
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels required for regression. "
            "Install with: pip install statsmodels"
        )
    
    # Map model types to statsmodels functions
    model_funcs = {
        'ols': smf.wls,
        'logit': smf.logit,
        'probit': smf.probit,
        'poisson': smf.poisson
    }
    
    if model_type.lower() not in model_funcs:
        raise ValueError(f"Model type must be one of: {list(model_funcs.keys())}")
    
    model_type = model_type.lower()
    valid_types = ['ols', 'logit', 'probit', 'poisson']
    if model_type not in valid_types:
        raise ValueError(f"Model type must be one of: {valid_types}")

    # GLM families for non-OLS models
    families = {
        'logit': sm.families.Binomial(),
        'probit': sm.families.Binomial(link=sm.families.links.Probit()),
        'poisson': sm.families.Poisson(),
    }

    def _build_model(data: pd.DataFrame, weights: np.ndarray, formula: str):
        """
        Build a weighted statsmodels model.

        Args:
            data (pd.DataFrame): DataFrame with variables
            weights (np.ndarray): Weight array (main weight OR replicate weight)
            formula (str): Regression formula

        Returns:
            Unfitted statsmodels model
        """
        if model_type == 'ols':
            return smf.wls(formula, data=data, weights=weights)
        return smf.glm(formula, data=data, family=families[model_type], var_weights=weights)

    def fit_regression(data: pd.DataFrame, weights: np.ndarray, formula: str) -> np.ndarray:
        """
        Fit regression and return coefficient array.

        Called by scf_estimate() with the main weight (wgt) for each implicate
        and with each replicate weight (wgt1-wgt999).

        Args:
            data (pd.DataFrame): DataFrame with variables
            weights (np.ndarray): Weight array (main weight OR replicate weight)
            formula (str): Regression formula

        Returns:
            np.ndarray: Coefficient estimates
        """
        return _build_model(data, weights, formula).fit().params.values

    # Get coefficient names and dependent variable from the first implicate,
    # using the same subset the estimation will use
    first_imp = data['implicates'][0]
    if subset is not None:
        first_imp = first_imp[subset(first_imp)].copy()
    first_wgt = first_imp[data['metadata']['weight_col']].values

    try:
        temp_model = _build_model(first_imp, first_wgt, formula)
        coef_names = temp_model.fit().params.index.tolist()
        dep_var = temp_model.endog_names
    except Exception as e:
        raise ValueError(f"Failed to extract coefficient names: {str(e)}")

    # Now use scf_estimate to compute coefficients with proper variance
    result = scf_estimate(
        data,
        fit_regression,
        subset=subset,
        coefficient_names=coef_names,
        formula=formula
    )

    # Add display names
    result.variable_name = dep_var
    result.statistic_name = {
        'ols': 'OLS Regression',
        'logit': 'Logit Regression',
        'probit': 'Probit Regression',
        'poisson': 'Poisson Regression',
    }[model_type]
    result.is_regression = True

    return result

def scf_total(
    data: Dict[str, pd.DataFrame],
    variable: str,
    subset: Optional[Callable[[pd.DataFrame], pd.Series]] = None
) -> scfcombo_result:
    """
    Compute weighted total (population-level sum) with full variance estimation.
    NOTE: This is optimized for the 'wgt' variable. If the weight specified in prepare_scf_data represents the whole population
          for each implicate (i.e. wgt0), then you should not specify {estimand_type='total'}. Otherwise, the aggregate will be
          5x its correct size.
    
    Args:
        data (dict): Dictionary from prepare_scf_data()
        variable (str): Name of variable to compute total for
        subset (function): Optional function for subsetting data
        
    Returns:
        scfcombo_result object
        
    Example:
        >>> result = scf_total(scf_data, 'networth')
        >>> result.display()  # Total household net worth in population
    """
    def weighted_total(data: pd.DataFrame, weights: np.ndarray, variable: str) -> float:
        notmissing = data[variable].notna()
        if not notmissing.any():
            return np.nan
        return np.sum(data.loc[notmissing, variable].values * weights[notmissing])

    result = scf_estimate(data, weighted_total, estimand_type='total', subset=subset, variable=variable)
    
    result.statistic_name = 'Total'
    result.variable_name = variable
    
    return result