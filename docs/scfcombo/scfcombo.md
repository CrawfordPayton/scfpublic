# scfpublic.scfcombo.scfcombo


SCF Variance Estimation Module for public dataset

This module provides a general-purpose framework for computing statistics with
proper variance estimation on SCF data, following the methodology in:

    - SAS macro MEANIT from https://www.federalreserve.gov/econres/files/codebk2022.txt
    - Stata module SCFCOMBO from https://ideas.repec.org/c/boc/bocode/s458017.html
    - R function scf_MIcombine from https://www.convey-r.org/1.6-survey-of-consumer-finances-scf.html#survey-of-consumer-finances-scf
    
Author: Payton Crawford


## `prepare_scf_data`

```python
prepare_scf_data(main_df: pandas.core.frame.DataFrame, replicate_weights_df: pandas.core.frame.DataFrame, weight_col: str = 'wgt', implicate_id_col: str = 'y1', household_id_col: str = 'yy1', year_col: str = 'year', repweight_col: str = 'wt1b', repmultiple_col: str = 'mm', n_implicates: int = 5, n_replicates: int = 999, verbose: bool = False) -> Dict[str, pandas.core.frame.DataFrame]
```

Prepare SCF data by splitting into implicates and merging with replicate weights.

**Parameters:**

- `main_df (pd.DataFrame)`: Pandas dataframe containing dataframe for calculating statistics (contains weight, implicate_id, and household_id columns)
- `replicate_weights_df (pd.DataFrame)`: Pandas dataframe containing replicate weights for each household (contains household_id column)
- `weight_col (str)`: Contains column name for weight (e.g. wgt)
- `implicate_id_col (str)`: Contains column name for unique implicate id (e.g. y1)
- `household_id_col (str)`: Contains column name for unique household id (e.g. yy1)
- `year_col (str)`: Contains column name for year variable (e.g. year)
- `repweight_col (str)`: Contains header for replicate weight variable name (1-999) (default is wt1b)
- `repmultiple_col (str)`: Contains header for replicate multiplier variable name (1-999) (default is mm)
- `n_implicates (int)`: Number of implicates in main_df (default is 5)
- `n_replicates (int)`: Number of replicates in replicate_weights_df (default is 999)
- `verbose (bool)`: True to output print messages, False for quietly

**Returns:**

dict: Contains the following:

    - 'implicates': List of DataFrames (one per implicate)
    - 'replicate_weights': DataFrame with replicate weights
    - 'metadata': Dict with dataset information

## `scfcombo_result` (class)

Container for SCF variance estimation results with output formatting.

**Attributes:**

- `estimate` (*float/np.ndarray*): Point estimate
- `imputation_var` (*float/np.ndarray*): Variance due to multiple imputation
- `sampling_var` (*float/np.ndarray*): Variance due to sampling
- `total_var` (*float/np.ndarray*): Combined variance
- `standard_error` (*float/np.ndarray*): Standard error
- `n_implicates` (*int*): Number of implicates used
- `n_replicates` (*int*): Number of replicate weights used
- `n_obs` (*int*): Number of observations (via first implicate)
- `coefficient_names` (*list*): Names of coefficients (None for scalar results)
- `statistic_name` (*str*): Name of the statistic (e.g., 'Mean', 'Median', 'Regression') (None for scalar results)
- `variable_name` (*str*): Variable analyzed (scalar results) or dependent variable (regressions)
- `is_regression` (*bool*): True if regression, False otherwise to discriminate between vector results (such as group means) and regressions

### `scfcombo_result.__init__`

### `scfcombo_result.display`

```python
scfcombo_result.display(self, alpha: float = 0.05, decimals: int = 4) -> None
```

Display formatted output.

**Parameters:**

- `alpha (float)`: Significance level for confidence intervals (default: 0.05)
- `decimals (int)`: Number of decimal places to display (default: 4)

### `scfcombo_result.results_table`

```python
scfcombo_result.results_table(self, alpha: float = 0.05, decimals: Optional[int] = None) -> pandas.core.frame.DataFrame
```

Return a coefficient table as a DataFrame.

Columns: Variable, Coef., Std. Err., z, P>|z|, CI Lower, CI Upper,
Imputation SE, Sampling SE. Model-level information is stored in df.attrs.

**Parameters:**

- `alpha (float)`: Significance level for confidence intervals (default: 0.05)
- `decimals (int)`: Round numeric columns if provided (default: None, no rounding)

**Returns:**

pd.DataFrame: Results table, with model info in df.attrs

### `scfcombo_result.summary`

```python
scfcombo_result.summary(self, alpha: float = 0.05, decimals: Optional[int] = None) -> pandas.core.frame.DataFrame
```

Return a summary DataFrame of the results (alias for results_table).

**Parameters:**

- `alpha (float)`: Significance level for confidence intervals (default: 0.05)
- `decimals (int)`: Round numeric columns if provided (default: None)

**Returns:**

pd.DataFrame: Results table

## `scf_estimate`

```python
scf_estimate(data: Dict[str, pandas.core.frame.DataFrame], estimator: Callable[[pandas.core.frame.DataFrame, numpy.ndarray], Union[float, numpy.ndarray]], subset: Optional[Callable[[pandas.core.frame.DataFrame], pandas.core.series.Series]] = None, coefficient_names: Optional[list] = None, estimand_type: str = 'mean', **estimator_kwargs) -> scfpublic.scfcombo.scfcombo.scfcombo_result
```

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

**Parameters:**

- `data`: Dictionary from prepare_scf_data() containing:

    - 'implicates': List of DataFrames (one per implicate)
    - 'replicate_weights': DataFrame with replicate weights
    - 'metadata': Dict with dataset information
- `estimator`: Function that computes the statistic of interest.
    Signature: estimator(data: pd.DataFrame, weights: np.ndarray, **kwargs)
               -> Union[float, np.ndarray]

    - data: DataFrame containing the variables needed
    - weights: Array of weights to use
    - **kwargs: Additional arguments passed through estimator_kwargs
    - Returns: Scalar estimate or array of estimates (e.g., coefficients)
- `subset`: Optional function that takes a DataFrame and returns a boolean Series
    for subsetting. Example: lambda df: df['age'] > 25
- `coefficient_names`: Optional list of names for vector results (e.g., regression)
- `estimand_type (str)`: Type of estimand pooling:

    - 'mean': Average across implicates (default)
             Use for: means, medians, quantiles, regression coefficients, Gini coefficients
    - 'total': Sum across implicates
              Use for: population totals, counts, aggregates
- `**estimator_kwargs`: Additional keyword arguments passed to estimator function

**Returns:**

scfcombo_result object containing estimates and variance components

**Examples:**

```python
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
    - SAS macro MEANIT from https://www.federalreserve.gov/econres/files/codebk2022.txt
    - Stata module SCFCOMBO from https://ideas.repec.org/c/boc/bocode/s458017.html
    - R function scf_MIcombine from https://www.convey-r.org/1.6-survey-of-consumer-finances-scf.html#survey-of-consumer-finances-scf
```

