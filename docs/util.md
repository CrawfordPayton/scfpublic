# scfpublic.util


Utility functions for SCF

    - mconv: Monthly conversion function
    - aconv: Annual conversion function
    
Author: Payton Crawford


## `aconv`

```python
aconv(df: pandas.core.frame.DataFrame, value_col: str, period_col: str)
```

Convert SCF values to annual equivalents based on time period codes following SAS Bulletin mconv function (https://www.federalreserve.gov/econres/files/bulletin.macro.txt).

**Parameters:**

- `df (pandas.DataFrame)`: The dataframe containing the SCF data
- `value_col (str)`: Name of the column containing the values to convert
- `period_col (str)`: Name of the column containing the period/frequency codes

**Returns:**

pandas.Series: Series with values converted to annual amounts

**Examples:**

```python
>>> df['foodhome'] = aconv(df, 'x3024', 'x3025')
```

## `mconv`

```python
mconv(df: pandas.core.frame.DataFrame, value_col: str, period_col: str)
```

Convert SCF values to monthly equivalents based on time period codes following SAS Bulletin (https://www.federalreserve.gov/econres/files/bulletin.macro.txt).

**Parameters:**

- `df (pandas.DataFrame)`: The dataframe containing the SCF data
- `value_col (str)`: Name of the column containing the values to convert
- `period_col (str)`: Name of the column containing the period/frequency codes

**Returns:**

pandas.Series: Series with values converted to monthly amounts

**Examples:**

```python
>>> df['payedu1'] = mconv(df, 'x7815', 'x7816')
```

