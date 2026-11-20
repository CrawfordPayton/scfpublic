"""
SCF Variance Estimation Module for public dataset

This module provides a general-purpose framework for computing statistics with
proper variance estimation on SCF data, following the methodology in:

    - SAS macro MEANIT from https://www.federalreserve.gov/econres/files/codebk2025.txt
    - Stata module SCFCOMBO from https://ideas.repec.org/c/boc/bocode/s458017.html
    - R function scf_MIcombine from https://www.convey-r.org/1.6-survey-of-consumer-finances-scf.html#survey-of-consumer-finances-scf
    
Author: Payton Crawford
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Union, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings

@dataclass
class scfcombo_result:
    """
    Container for SCF variance estimation results with output formatting.
    
    Attributes:
        estimate (float/np.ndarray): Point estimate 
        imputation_var (float/np.ndarray): Variance due to multiple imputation
        sampling_var (float/np.ndarray): Variance due to sampling 
        total_var (float/np.ndarray): Combined variance 
        standard_error (float/np.ndarray): Standard error
        n_implicates (int): Number of implicates used
        n_replicates (int): Number of replicate weights used
        n_obs (int): Number of observations (via first implicate)
        coefficient_names (list): Names of coefficients (None for scalar results)
        statistic_name (str): Name of the statistic (e.g., 'Mean', 'Median', 'Regression') (None for scalar results)
        variable_name (str): Variable analyzed (scalar results) or dependent variable (regressions)
        is_regression (bool): True if regression, False otherwise to discriminate between vector results (such as group means) and regressions
    """
    estimate: Union[float, np.ndarray]
    imputation_var: Union[float, np.ndarray]
    sampling_var: Union[float, np.ndarray]
    total_var: Union[float, np.ndarray]
    standard_error: Union[float, np.ndarray]
    n_implicates: int
    n_replicates: int
    n_obs: int
    coefficient_names: Optional[list] = None
    statistic_name: Optional[str] = None
    variable_name: Optional[str] = None
    is_regression: bool = False
    
    def __post_init__(self):
        """Set default statistic name if not provided."""
        if self.statistic_name is None:
            if self.is_regression:
                self.statistic_name = 'Regression'
            elif self.coefficient_names is not None:
                self.statistic_name = 'Multiple Statistics'
            else:
                self.statistic_name = 'Statistic'
    
    def summary(self, alpha: float = 0.05, decimals: Optional[int] = None) -> pd.DataFrame:
        """
        Return a summary DataFrame of the results (alias for results_table).

        Args:
            alpha (float): Significance level for confidence intervals (default: 0.05)
            decimals (int): Round numeric columns if provided (default: None)

        Returns:
            pd.DataFrame: Results table
        """
        return self.results_table(alpha=alpha, decimals=decimals)
    
    def _get_p_value(self, z_score: float) -> float:
        """
        Compute two-tailed p-value from z-score.
        
        Args: 
            z_score (float): Z-score

        Returns:
            float: 2-tailed p-value
        """
        return 2 * stats.norm.sf(abs(z_score))
    
    def _get_conf_interval(self, 
        estimate: float, 
        se: float, 
        alpha: float = 0.05
    ) -> tuple:
        """
        Compute confidence interval.

        Args: 
            estimate (float): Beta of regression or statistic
            se (float): Standard error
            alpha (float): Significance level for confidence intervals (default: 0.05)

        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        lower, upper = stats.norm.interval(1 - alpha, loc = estimate, scale = se)
        return lower, upper
    
    def _ci_labels(self, alpha: float) -> tuple:
        """
        Column labels for the confidence interval bounds.

        Args:
            alpha (float): Significance level

        Returns:
            tuple: (lower label, upper label)
        """
        conf = f"{(1 - alpha) * 100:g}%"
        return f'CI Lower ({conf})', f'CI Upper ({conf})'
    
    def _variance_shares(self) -> tuple:
        """
        Percentage of total variance from imputation and sampling.

        Returns:
            tuple: (imputation %, sampling %) as arrays
        """
        _, _, _, imp_var, samp_var, tot_var = self._as_arrays()
        positive = tot_var > 0
        imp_pct = np.divide(imp_var, tot_var, out=np.zeros_like(tot_var), where=positive) * 100
        samp_pct = np.divide(samp_var, tot_var, out=np.zeros_like(tot_var), where=positive) * 100
        return imp_pct, samp_pct

    def _as_arrays(self) -> tuple:
        """
        Normalize scalar and vector results into arrays.

        Returns:
            tuple: (names, estimate, standard_error, imputation_var,
                    sampling_var, total_var) as 1-D float arrays
        """
        est = np.atleast_1d(np.asarray(self.estimate, dtype=float))
        shape = est.shape

        def _arr(x):
            return np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), shape)

        se = _arr(self.standard_error)
        imp_var = _arr(self.imputation_var)
        samp_var = _arr(self.sampling_var)
        tot_var = _arr(self.total_var)

        if self.coefficient_names is not None:
            names = list(self.coefficient_names)
        else:
            names = [self.variable_name or self.statistic_name]

        return names, est, se, imp_var, samp_var, tot_var
    
    def results_table(self, alpha: float = 0.05, decimals: Optional[int] = None) -> pd.DataFrame:
        """
        Return a coefficient table as a DataFrame.

        Columns: Variable, Coef., Std. Err., z, P>|z|, CI Lower, CI Upper,
        Imputation SE, Sampling SE. Model-level information is stored in df.attrs.

        Args:
            alpha (float): Significance level for confidence intervals (default: 0.05)
            decimals (int): Round numeric columns if provided (default: None, no rounding)

        Returns:
            pd.DataFrame: Results table, with model info in df.attrs
        """
        names, est, se, imp_var, samp_var, _ = self._as_arrays()

        with np.errstate(divide='ignore', invalid='ignore'):
            z = est / se
        p = self._get_p_value(z)
        lower, upper = self._get_conf_interval(est, se, alpha)
        lo_label, hi_label = self._ci_labels(alpha)

        df = pd.DataFrame({
            'Variable': names,
            'Coef.': est,
            'Std. Err.': se,
            'z': z,
            'P>|z|': p,
            lo_label: lower,
            hi_label: upper,
            'Imputation SE': np.sqrt(imp_var),
            'Sampling SE': np.sqrt(samp_var),
        })

        if decimals is not None:
            df = df.round(decimals)

        df.attrs.update({
            'Statistic': self.statistic_name,
            'Variable': self.variable_name,
            'No. Observations': self.n_obs,
            'No. Implicates': self.n_implicates,
            'No. Replicates': self.n_replicates,
            'Confidence Level': 1 - alpha,
        })
        return df

    def display(self, alpha: float = 0.05, decimals: int = 4) -> None:
        """
        Display formatted output.
        
        Args:
            alpha (float): Significance level for confidence intervals (default: 0.05)
            decimals (int): Number of decimal places to display (default: 4)
        """
        table = self.results_table(alpha=alpha)
        if self.is_regression:
            # Regression-style output
            self._display_regression(table, alpha, decimals)
        elif self.coefficient_names is not None:
            # Vector-style output
            self._display_vector(table, alpha, decimals)
        else:
            # Summary statistic output
            self._display_scalar(table, alpha, decimals)

    def _print_footer(self) -> None:
        """Print information footer."""
        print("\n" + "-" * 78)
        print(f"Number of observations: {self.n_obs:,}")
        print(f"Number of implicates: {self.n_implicates}")
        print(f"Number of replicates: {self.n_replicates}")
        print("=" * 78 + "\n")
    
    def _display_scalar(self, table: pd.DataFrame, alpha: float, decimals: int) -> None:
        """
        Display scalar result.

        Args:
            table (pd.DataFrame): Output of results_table()
            alpha (float): Significance level for confidence intervals
            decimals (int): Number of decimal places to display
        """
        row = table.iloc[0]
        lo_label, hi_label = self._ci_labels(alpha)
        conf_level = f"{(1 - alpha) * 100:g}"
        imp_pct, samp_pct = (x[0] for x in self._variance_shares())
        w = decimals + 10

        # Header
        print("\n" + "=" * 78)
        if self.variable_name:
            print(f"{self.statistic_name}: {self.variable_name}")
        else:
            print(f"{self.statistic_name}")
        print("=" * 78)

        # Main results
        print(f"\n{'Estimate':<20} {row['Coef.']:>{w}.{decimals}f}")
        print(f"{'Std. Error':<20} {row['Std. Err.']:>{w}.{decimals}f}")
        print(f"{'Z-Score':<20} {row['z']:>{w}.{decimals}f}")
        print(f"{'P>|z|':<20} {row['P>|z|']:>{w}.{decimals}f}")
        print(f"\n[{conf_level}% Conf. Interval]   "
              f"{row[lo_label]:>{w}.{decimals}f}    {row[hi_label]:>{w}.{decimals}f}")

        # Variance decomposition
        print("\n" + "-" * 78)
        print("Variance Decomposition:")
        print("-" * 78)
        print(f"{'Imputation SE':<30} {row['Imputation SE']:>12.{decimals}f}  ({imp_pct:>5.1f}% of variance)")
        print(f"{'Sampling SE':<30} {row['Sampling SE']:>12.{decimals}f}  ({samp_pct:>5.1f}% of variance)")
        print(f"{'Total SE':<30} {row['Std. Err.']:>12.{decimals}f}")

        self._print_footer()

    def _display_vector(self, table: pd.DataFrame, alpha: float, decimals: int) -> None:
        """
        Display vector results.

        Args:
            table (pd.DataFrame): Output of results_table()
            alpha (float): Significance level for confidence intervals
            decimals (int): Number of decimal places to display
        """
        lo_label, hi_label = self._ci_labels(alpha)
        conf_level = f"{(1 - alpha) * 100:g}"
        imp_pct, samp_pct = self._variance_shares()
        
        # Header
        print("\n" + "=" * 78)
        print(f"{self.statistic_name}")
        if self.variable_name:
            print(f"Variable: {self.variable_name}")
        print("=" * 78)
        
        # Table
        name_width = max(table['Variable'].str.len().max(), 20)
        num_width = decimals + 8
        stat_width = 8
        
        header = (f"{'Statistic':<{name_width}} "
                  f"{'Estimate':>{num_width}} "
                  f"{'Std.Err.':>{num_width}} "
                  f"{'z':>{stat_width}} "
                  f"{'P>|z|':>{stat_width}} "
                  f"{f'[{conf_level}% Conf.':>{num_width}} "
                  f"{'Interval]':>{num_width}}")
        print(f"\n{header}")
        print("-" * 78)
        
        for _, r in table.iterrows():
            print(f"{r['Variable']:<{name_width}} "
                  f"{r['Coef.']:>{num_width}.{decimals}f} "
                  f"{r['Std. Err.']:>{num_width}.{decimals}f} "
                  f"{r['z']:>{stat_width}.{decimals}f} "
                  f"{r['P>|z|']:>{stat_width}.{decimals}f} "
                  f"{r[lo_label]:>{num_width}.{decimals}f} "
                  f"{r[hi_label]:>{num_width}.{decimals}f}")
        
        print("-" * 78)
        print("\nVariance Decomposition (average):")
        print(f"  Imputation SE: {table['Imputation SE'].mean():.{decimals}f}  "
              f"({imp_pct.mean():.1f}% of variance)")
        print(f"  Sampling SE:   {table['Sampling SE'].mean():.{decimals}f}  "
              f"({samp_pct.mean():.1f}% of variance)")
        
        self._print_footer()

    def _display_regression(self, table: pd.DataFrame, alpha: float, decimals: int) -> None:
        """
        Display regression results in statsmodels style.

        Args:
            table (pd.DataFrame): Output of results_table()
            alpha (float): Significance level for confidence intervals
            decimals (int): Number of decimal places to display
        """
        lo_label, hi_label = self._ci_labels(alpha)
        conf_level = f"{(1 - alpha) * 100:g}"
        imp_pct, samp_pct = self._variance_shares()

        # Column widths
        name_width = max(table['Variable'].str.len().max(), 20)
        num_width = decimals + 8
        stat_width = 8

        # Header
        print("\n" + "=" * 78)
        print("SCF Regression Results (Multiple Imputation with Replicate Weights)")
        print("=" * 78)
        print(f"Dep. Variable: {self.variable_name or 'N/A'}")

        # Column headers
        header = (f"{'Variable':<{name_width}} "
                  f"{'Coef.':>{num_width}} "
                  f"{'Std.Err.':>{num_width}} "
                  f"{'z':>{stat_width}} "
                  f"{'P>|z|':>{stat_width}} "
                  f"{f'[{conf_level}% Conf.':>{num_width}} "
                  f"{'Interval]':>{num_width}}")
        print(f"\n{header}")
        print("-" * 78)

        # Coefficient rows
        for _, r in table.iterrows():
            print(f"{r['Variable']:<{name_width}} "
                  f"{r['Coef.']:>{num_width}.{decimals}f} "
                  f"{r['Std. Err.']:>{num_width}.{decimals}f} "
                  f"{r['z']:>{stat_width}.{decimals}f} "
                  f"{r['P>|z|']:>{stat_width}.{decimals}f} "
                  f"{r[lo_label]:>{num_width}.{decimals}f} "
                  f"{r[hi_label]:>{num_width}.{decimals}f}")
        print("-" * 78)

        # Variance decomposition summary
        print("\n" + "-" * 78)
        print("Variance Decomposition (average across coefficients):")
        print("-" * 78)
        print(f"{'Average Imputation SE:':<30} {table['Imputation SE'].mean():>12.{decimals}f}  "
              f"({imp_pct.mean():>5.1f}% of variance)")
        print(f"{'Average Sampling SE:':<30} {table['Sampling SE'].mean():>12.{decimals}f}  "
              f"({samp_pct.mean():>5.1f}% of variance)")

        self._print_footer()

    def __repr__(self) -> str:
        """
        Short string representation.
        
        Returns:
            str: String representation
        """
        if self.coefficient_names is not None:
            return (f"scfcombo_result(Regression, {len(self.coefficient_names)} coefficients, "
                   f"SE={np.mean(self.standard_error):.4f})")
        else:
            return (f"scfcombo_result({self.statistic_name}, "
                   f"estimate={self.estimate:.4f}, se={self.standard_error:.4f})")

def prepare_scf_data(
    main_df: pd.DataFrame,
    replicate_weights_df: pd.DataFrame,
    weight_col: str = 'wgt',
    implicate_id_col: str = 'y1',
    household_id_col: str = 'yy1',
    year_col: str = 'year',
    repweight_col: str = 'wt1b',
    repmultiple_col: str = 'mm',
    n_implicates: int = 5,
    n_replicates: int = 999,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare SCF data by splitting into implicates and merging with replicate weights.
    
    Args:
        main_df (pd.DataFrame): Pandas dataframe containing dataframe for calculating statistics (contains weight, implicate_id, and household_id columns)
        replicate_weights_df (pd.DataFrame): Pandas dataframe containing replicate weights for each household (contains household_id column)
        weight_col (str): Contains column name for weight (e.g. wgt)
        implicate_id_col (str): Contains column name for unique implicate id (e.g. y1)
        household_id_col (str): Contains column name for unique household id (e.g. yy1)
        year_col (str): Contains column name for year variable (e.g. year)
        repweight_col (str): Contains header for replicate weight variable name (1-999) (default is wt1b)
        repmultiple_col (str): Contains header for replicate multiplier variable name (1-999) (default is mm)
        n_implicates (int): Number of implicates in main_df (default is 5)
        n_replicates (int): Number of replicates in replicate_weights_df (default is 999)
        verbose (bool): True to output print messages, False for quietly

    Returns:
        dict: Contains the following:
            - 'implicates': List of DataFrames (one per implicate)
            - 'replicate_weights': DataFrame with replicate weights
            - 'metadata': Dict with dataset information
    """
    # Validation: Check inputs
    if n_implicates < 2:
        raise ValueError(
            f"At least 2 implicates required for variance estimation. "
            f"Got n_implicates={n_implicates}."
        )
    
    if n_replicates < 2:
        raise ValueError(
            f"At least 2 replicate weights required for sampling variance. "
            f"Got n_replicates={n_replicates}."
        )
    
    # Validate required columns in main data
    required_vars = [household_id_col, implicate_id_col, weight_col, year_col]
    missing_vars = [col for col in required_vars if col not in main_df.columns]
    if missing_vars:
        raise ValueError(f'Missing columns in main data: {missing_vars}')
    
    # Validate household ID in replicate weights
    required_vars = [household_id_col, year_col]
    missing_vars = [col for col in required_vars if col not in replicate_weights_df.columns]
    if missing_vars:
        raise ValueError(f'Missing columns in replicate weights data: {missing_vars}')
    
    # =========================================================================
    # STEP 1: Split main data into implicates
    # =========================================================================
    
    # Extract implicate number from y1
    main_df = main_df.copy()
    main_df['_implicate_num'] = (main_df[implicate_id_col].astype('int64') % 10)
    
    # Validate implicate numbers
    unique_implicates = main_df['_implicate_num'].unique()
    expected_implicates = set(range(1, n_implicates + 1))
    
    if not set(unique_implicates).issubset(expected_implicates):
        raise ValueError(
            f"Found unexpected implicate numbers: {sorted(unique_implicates)}. "
            f"Expected values 1-{n_implicates} based on last digit of '{implicate_id_col}'."
        )
    
    # Split into separate implicates
    implicates = []
    for i in range(1, n_implicates + 1):
        imp_df = main_df[main_df['_implicate_num'] == i].copy()
        
        if len(imp_df) == 0:
            raise ValueError(
                f"Implicate {i} has no observations. "
                f"Check that '{implicate_id_col}' has values ending in {i}."
            )
        
        # Drop temporary column and sort by household id
        imp_df = imp_df.drop('_implicate_num', axis=1)
        imp_df = imp_df.sort_values([year_col, household_id_col]).reset_index(drop=True)
        implicates.append(imp_df)
    
    # Validate all implicates have same number of households
    n_households_per_imp = [len(imp) for imp in implicates]
    if len(set(n_households_per_imp)) > 1:
        warnings.warn(
            f"Implicates have different numbers of households: {n_households_per_imp}. "
            f"This may indicate a problem with the data or filtering."
        )
    
    n_households = len(implicates[0])
    
    # =========================================================================
    # STEP 2: Process replicate weights
    # =========================================================================
    
    rw_df = replicate_weights_df.copy()
    
    # Check for base replicate weights (wt1b1-wt1b999) and multiplicity factors (mm1-mm999)
    # WARNING: Uses assumed replicate weight variable names (lowercase following scfpublic) from public dataset
    wt_cols = [f'{repweight_col}{i}' for i in range(1, n_replicates + 1)]
    mm_cols = [f'{repmultiple_col}{i}' for i in range(1, n_replicates + 1)]
    
    has_wt = all(col in rw_df.columns for col in wt_cols)
    has_mm = all(col in rw_df.columns for col in mm_cols)
    
    if has_wt and has_mm:
        # Compute combined replicate weights: wgt{i} = wt1b{i} * mm{i}
        # For each replicate, each weight (wt1b{i}) is a household's weight relative to total population, and each multiplicity factor (mm{i}) represents the number of times the household was sampled
        if verbose:
            print(f"Computing combined replicate weights: wgt{{i}} = {repweight_col}{{i}} * {repmultiple_col}{{i}}")
        
        # Extract base weights and multiplicity factors as arrays
        wt_array = rw_df[wt_cols].values  # Shape: (n_households, n_replicates)
        mm_array = rw_df[mm_cols].values  # Shape: (n_households, n_replicates)
        
        # Vectorized multiplication (contrary to SAS row-by-row multiplication)
        wgt_array = wt_array * mm_array
        
        # Create new column names
        wgt_col_names = [f'wgt{i}' for i in range(1, n_replicates + 1)]
        
        # Create DataFrame with computed weights
        wgt_df = pd.DataFrame(wgt_array, columns=wgt_col_names, index=rw_df.index)
        
        # Combine household ID with computed weights
        rw_df = pd.concat([rw_df[[year_col, household_id_col]], wgt_df], axis=1)
        
    else:
        # Assume weights are already combined
        wgt_cols = [f'wgt{i}' for i in range(1, n_replicates + 1)]
        missing_cols = [col for col in wgt_cols if col not in rw_df.columns]
        
        if missing_cols:
            # Show first few missing for clarity
            missing_sample = missing_cols[:5]
            raise ValueError(
                f"Replicate weights must have either:\n"
                f"  1) '{repweight_col}1'-'{repweight_col}{n_replicates}' AND '{repmultiple_col}1'-'{repmultiple_col}{n_replicates}' columns, OR\n"
                f"  2) 'wgt1'-'wgt{n_replicates}' columns (already combined)\n"
                f"Missing columns include: {missing_sample}..."
            )
        
        if verbose:
            print(f"Using pre-combined replicate weights: wgt1-wgt{n_replicates}")
        
        rw_df = rw_df[[year_col, household_id_col] + wgt_cols]
    
    # Replace missing values with 0
    with pd.option_context('future.no_silent_downcasting', True): # .infer_objects(copy=False) conforms to future behavior, so silence this warning
        rw_df = rw_df.fillna(0).infer_objects(copy=False)
    
    # Sort by household ID to ensure proper merge
    rw_df = rw_df.sort_values(household_id_col).reset_index(drop=True)
    
    # Validate replicate weights structure
    if len(rw_df) == 0:
        raise ValueError("Replicate weights DataFrame is empty.")
    
    # Check for households in implicates but not in replicate weights
    implicate_households = set(implicates[0][household_id_col])
    rw_households = set(rw_df[household_id_col])
    
    missing_rw = implicate_households - rw_households
    if missing_rw:
        n_missing = len(missing_rw)
        warnings.warn(
            f"{n_missing} households in main data not found in replicate weights. "
            f"These will have missing replicate weights after merge. "
            f"Sample missing IDs: {list(missing_rw)[:5]}"
        )
    
    # Create metadata for implicates, replicates, and column information
    metadata = {
        'n_households': n_households,
        'n_implicates': n_implicates,
        'n_replicates': n_replicates,
        'weight_col': weight_col,
        'household_id_col': household_id_col,
        'implicate_id_col': implicate_id_col,
        'year_col': year_col,
        'merge_keys': [year_col, household_id_col]
    }
    
    return {
        'implicates': implicates,
        'replicate_weights': rw_df,
        'metadata': metadata
    }

def scf_estimate(
    data: Dict[str, pd.DataFrame],
    estimator: Callable[[pd.DataFrame, np.ndarray], Union[float, np.ndarray]],
    subset: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    coefficient_names: Optional[list] = None,
    estimand_type: str = 'mean',
    **estimator_kwargs
) -> scfcombo_result:
    """
    General-purpose SCF variance estimation  implements the unified variance estimation procedure that works
    for means, medians, quantiles, regression coefficients, or user-defined statistic.
    
    Mathematical Procedure:
        1. For each implicate i (i=1 to 5), where I is the total number of implicates:
           b_hat_imp{i} = estimator(data{i}, weights{i})
           
        2. Pooled estimate:
           b_hat_imp = (1/I) * sum(b_hat_imp{i}) 
           
        3. Imputation variance:
           V_imp = (1/(I-1)) * sum(b_hat_imp{i} - b_hat_imp)**2
           
        4. For each replicate r (r=1 to 999, using first implicate (i=1) only), where R is the total number of replicates:
           b_hat_sam{r} = estimator(data{1}, weights{r})

        5. Pooled estimate:
           b_hat_sam = (1/R) * sum(b_hat_sam{r}) 
           
        6. Sampling variance:
           V_sam = (1/(R-1)) * sum(beta_hat{r} - b_hat_sam)**2
           
        7. Total variance:
           V_total = ((I+1)/I) * V_imp + V_sam
    
    Args:
        data: Dictionary from prepare_scf_data() containing:
            - 'implicates': List of DataFrames (one per implicate)
            - 'replicate_weights': DataFrame with replicate weights
            - 'metadata': Dict with dataset information
        estimator: Function that computes the statistic of interest.
            Signature: estimator(data: pd.DataFrame, weights: np.ndarray, **kwargs)
                       -> Union[float, np.ndarray]
            - data: DataFrame containing the variables needed
            - weights: Array of weights to use
            - **kwargs: Additional arguments passed through estimator_kwargs
            - Returns: Scalar estimate or array of estimates (e.g., coefficients)
        subset: Optional function that takes a DataFrame and returns a boolean Series
            for subsetting. Example: lambda df: df['age'] > 25
        coefficient_names: Optional list of names for vector results (e.g., regression)
        estimand_type (str): Type of estimand pooling:
            - 'mean': Average across implicates (default)
                     Use for: means, medians, quantiles, regression coefficients, Gini coefficients
            - 'total': Sum across implicates
                      Use for: population totals, counts, aggregates
        **estimator_kwargs: Additional keyword arguments passed to estimator function
        
    Returns:
        scfcombo_result object containing estimates and variance components
        
    Example:
        >>> # Example 1: Weighted mean
        >>> def weighted_mean(data, weights, variable):
        ...     non_missing = data[variable].notna()
        ...     return np.average(data.loc[non_missing, variable], weights=weights[non_missing])
        >>> 
        >>> result = scf_estimate(
        ...     scf_data,
        ...     weighted_mean,
        ...     variable='networth'
        ... )
        
        >>> # Example 2: Weighted median
        >>> def weighted_quantile(data, weights, variable, q):
        ...     from statsmodels.stats.weightstats import DescrStatsW
        ...     notmissing = data[variable].notna()
        ...     weighted_stats = DescrStatsW(data.loc[notmissing, variable].values, weights=weights[notmissing])
        ...     return float(weighted_stats.quantile(probs=[q]).iloc[0])
        >>> 
        >>> result = scf_estimate(
        ...     scf_data,
        ...     weighted_quantile,
        ...     variable='networth',
        ...     q=0.5
        ... )
        
        >>> # Example 3: Regression
        >>> import statsmodels.api as sm
        >>> 
        >>> def ols_coefficients(data, weights, formula):
        ...     import statsmodels.formula.api as smf
        ...     model = smf.wls(formula, data=data, weights=weights)
        ...     result = model.fit()
        ...     return result.params.values
        >>> 
        >>> result = scf_estimate(
        ...     scf_data,
        ...     ols_coefficients,
        ...     formula='networth ~ age + income + C(education)',
        ...     coefficient_names=['Intercept', 'age', 'income', 'education_1', 'education_2']
        ... )
        
    References:
        - SAS macro MEANIT from https://www.federalreserve.gov/econres/files/codebk2025.txt
        - Stata module SCFCOMBO from https://ideas.repec.org/c/boc/bocode/s458017.html
        - R function scf_MIcombine from https://www.convey-r.org/1.6-survey-of-consumer-finances-scf.html#survey-of-consumer-finances-scf
    """
    # Validate estimand_type
    if estimand_type not in ['mean', 'total']:
        raise ValueError(f"estimand_type must be 'mean' or 'total', got '{estimand_type}'")

    implicates = data['implicates']
    rw_df = data['replicate_weights']
    metadata = data['metadata']
    
    weight_col = metadata['weight_col']
    merge_keys = metadata['merge_keys']
    n_implicates = metadata['n_implicates']
    n_replicates = metadata['n_replicates']

    # Check minimum requirements for implicates and replicates
    if n_implicates < 2:
        raise ValueError(
            f"At least 2 implicates required for variance estimation. "
            f"Found {n_implicates}."
        )
    
    if n_replicates < 2:
        raise ValueError(
            f"At least 2 replicate weights required for sampling variance. "
            f"Found {n_replicates}."
        )
    
    # =========================================================================
    # STEP 1: IMPUTATION VARIANCE (using all implicates)
    # =========================================================================
    
    implicate_estimates = []
    
    for imp_df in implicates:
        # Apply subset if provided
        if subset is not None:
            imp_df = imp_df[subset(imp_df)].copy()
        
        # Get weights for this implicate
        weights = imp_df[weight_col].values
        
        # Compute estimate using the provided estimator function
        estimate_i = estimator(imp_df, weights, **estimator_kwargs)
        
        implicate_estimates.append(estimate_i)
    
    # Convert to array for faster computation
    implicate_estimates = np.array(implicate_estimates)

    # Uniformly, multiply implicate estimates by the number of implicates 
    # Assuming weights mimic the bulletin with wgt = X42001/5, such that multiplying by 5 makes each implicate representative of the entire population
    # Alternatively, call prepare_scf_data with the 'wgt0' bulletin variable and use estimand_type = 'mean' with any calculation type
    if estimand_type == 'total':
        implicate_estimates *= n_implicates
    # Average across implicates
    pooled_estimate = np.mean(implicate_estimates, axis=0)
    
    if implicate_estimates.ndim == 1:
        # Scalar results (mean, median, etc.)
        imputation_var = np.var(implicate_estimates, ddof=1) # ddof=1 divides the variance by N-1 (default is division by N)
    else:
        # Vector results (regression coefficients, etc.)
        imputation_var = np.var(implicate_estimates, axis=0, ddof=1) # ddof=1 divides the variance by N-1 (default is division by N)

    # =========================================================================
    # STEP 2: SAMPLING VARIANCE (using first implicate with replicate weights)
    # =========================================================================
    
    # Use first implicate only
    first_implicate = implicates[0].copy()
    
    # Apply subset if provided
    if subset is not None:
        first_implicate = first_implicate[subset(first_implicate)].copy()

    n_obs = len(first_implicate)
    
    # Merge with replicate weights
    first_implicate = first_implicate.merge(rw_df, on=merge_keys, how='left')
    
    replicate_estimates = []
    
    for r in range(1, n_replicates + 1):
        rep_weight_col = f'wgt{r}'
        
        # Check if replicate weight exists and is valid
        if rep_weight_col not in first_implicate.columns:
            warnings.warn(f"Replicate weight column {rep_weight_col} not found. Stopping at {r-1} replicates.")
            break
        
        # Get replicate weights
        rep_weights = first_implicate[rep_weight_col].values
        
        # Skip if all weights are zero (no valid observations)
        if np.sum(rep_weights > 0) == 0:
            warnings.warn(f"Replicate {r} has no valid observations. Skipping.")
            continue
        
        try:
            # Compute estimate with replicate weights
            estimate_r = estimator(first_implicate, rep_weights, **estimator_kwargs)
            replicate_estimates.append(estimate_r)
        except Exception as e:
            warnings.warn(f"Replicate {r} failed with error: {str(e)}. Skipping.")
            continue
    
    # Convert to array
    replicate_estimates = np.array(replicate_estimates)
    n_successful_replicates = len(replicate_estimates)
    
    if n_successful_replicates == 0:
        raise ValueError("No successful replicate estimates. Cannot compute sampling variance.")
    
    # Sampling variance (variance across replicates)
    if replicate_estimates.ndim == 1:
        # Scalar results
        sampling_var = np.var(replicate_estimates, ddof=1) # ddof=1 divides the variance by N-1 (default is division by N)
    else:
        # Vector results
        sampling_var = np.var(replicate_estimates, axis=0, ddof=1) # ddof=1 divides the variance by N-1 (default is division by N)
    
    # =========================================================================
    # STEP 3: COMBINE VARIANCES
    # =========================================================================
    
    # Apply Rubin's finite imputation correction: (I+1)/I
    adjusted_imputation_var = ((n_implicates + 1) / n_implicates) * imputation_var
    
    # Total variance (assume independence)
    total_var = adjusted_imputation_var + sampling_var
    
    # Standard errors
    standard_error = np.sqrt(total_var)
    
    return scfcombo_result(
        estimate=pooled_estimate,
        imputation_var=adjusted_imputation_var,
        sampling_var=sampling_var,
        total_var=total_var,
        standard_error=standard_error,
        n_implicates=n_implicates,
        n_replicates=n_successful_replicates,
        n_obs=n_obs,
        coefficient_names=coefficient_names
    )
