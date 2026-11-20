# scfpublic.cpi


Stores cpi adjustment formulas using data from https://www.bls.gov/cpi/research-series/r-cpi-u-rs-home.htm

Author: Payton Crawford


## `cpi_adjust`

```python
cpi_adjust(df: pandas.core.frame.DataFrame, base_year: int = 2025)
```

Appends cpiadj and cpilag variables based on 'year' variable in dataset.

**Parameters:**

- `df (pd.DataFrame)`: Pandas dataframe to have CPI variables added to
- `base_year (int)`: Year to use as base for for CPI base value (default: 2022)

**Returns:**

pd.DataFrame: Original dataframe with new cpiadj and cpilag variables merged on year

**Raises:**

- `ValueError`: Provided invalid year, DataFrame lacks year variable, or DataFrame lacks correct years

## `read_excel_from_url`

```python
read_excel_from_url(url: str, sheet_name: int = 0, **kwargs)
```

Download Excel file from BLS URL and read into DataFrame.

**Parameters:**

- `url (str)`: URL of excel file
- `sheet_name (int)`: Sheet name or index (default: 0 = first sheet)
- `**kwargs`: Additional arguments for pd.read_excel()

**Returns:**

pd.DataFrame: Pandas dataframe containing Excel file from URL

**Raises:**

- `IOError`: Failure to download or process BLS file

