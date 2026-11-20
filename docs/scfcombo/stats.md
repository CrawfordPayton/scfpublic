# scfpublic.scfcombo.stats


Convenience summary-statistic wrappers around scf_estimate().

Author: Payton Crawford


## `scf_mean`

```python
scf_mean(data: Dict[str, pandas.core.frame.DataFrame], variable: str, subset: Optional[Callable[[pandas.core.frame.DataFrame], pandas.core.series.Series]] = None) -> scfpublic.scfcombo.scfcombo.scfcombo_result
```

Compute weighted mean with full variance estimation.

This is a convenience wrapper around scf_estimate() for computing means.

**Parameters:**

- `data (dict)`: Dictionary from prepare_scf_data()
- `variable (str)`: Name of variable to compute mean for
- `subset (function)`: Optional function for subsetting data

**Returns:**

scfcombo_result object

**Examples:**

```python
>>> result = scf_mean(scf_data, 'networth')
>>> result = scf_mean(scf_data, 'income', subset=lambda df: df['age'] > 25)
```

## `scf_quantile`

```python
scf_quantile(data: Dict[str, pandas.core.frame.DataFrame], variable: str, quantile: float = 0.5, subset: Optional[Callable[[pandas.core.frame.DataFrame], pandas.core.series.Series]] = None) -> scfpublic.scfcombo.scfcombo.scfcombo_result
```

Compute weighted quantile with full variance estimation.

This is a convenience wrapper around scf_estimate() for computing quantiles.

**Parameters:**

- `data (dict)`: Dictionary from prepare_scf_data()
- `variable (str)`: Name of variable to compute quantile for
- `quantile (float)`: Quantile to compute (0.5 for median)
- `subset (function)`: Optional function for subsetting data

**Returns:**

scfcombo_result object

**Examples:**

```python
>>> result = scf_quantile(scf_data, 'networth', quantile=0.5)  # Median
>>> result = scf_quantile(scf_data, 'income', quantile=0.9)   # 90th percentile
```

## `scf_regression`

```python
scf_regression(data: Dict[str, pandas.core.frame.DataFrame], formula: str, model_type: str = 'ols', subset: Optional[Callable[[pandas.core.frame.DataFrame], pandas.core.series.Series]] = None) -> scfpublic.scfcombo.scfcombo.scfcombo_result
```

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

**Parameters:**

- `data (dict)`: Dictionary from prepare_scf_data()
- `formula (str)`: Regression formula in statsmodels format
- `model_type (str)`: Type of model ('ols', 'logit', 'probit', 'poisson')
- `subset (function)`: Optional function for subsetting data

**Returns:**

scfcombo_result object with coefficient estimates and standard errors

References:

    - SAS macro MEANIT from https://www.federalreserve.gov/econres/files/codebk2022.txt
    - Stata module SCFCOMBO from https://ideas.repec.org/c/boc/bocode/s458017.html

**Examples:**

```python
>>> result = scf_regression(
...     scf_data,
...     'networth ~ age + income + C(education)',
...     model_type='ols'
... )
>>> result.display()
```

## `scf_total`

```python
scf_total(data: Dict[str, pandas.core.frame.DataFrame], variable: str, subset: Optional[Callable[[pandas.core.frame.DataFrame], pandas.core.series.Series]] = None) -> scfpublic.scfcombo.scfcombo.scfcombo_result
```

Compute weighted total (population-level sum) with full variance estimation.
NOTE: This is optimized for the 'wgt' variable. If the weight specified in prepare_scf_data represents the whole population
      for each implicate (i.e. wgt0), then you should not specify {estimand_type='total'}. Otherwise, the aggregate will be
      5x its correct size.

**Parameters:**

- `data (dict)`: Dictionary from prepare_scf_data()
- `variable (str)`: Name of variable to compute total for
- `subset (function)`: Optional function for subsetting data

**Returns:**

scfcombo_result object

**Examples:**

```python
>>> result = scf_total(scf_data, 'networth')
>>> result.display()  # Total household net worth in population
```

