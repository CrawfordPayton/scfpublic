# scfpublic.scfpublic


Downloads SCF files from https://www.federalreserve.gov/econres/scfindex.htm and formats them into a Pandas dataframe

Author: Payton Crawford


## `download_main_extr_years`

```python
download_main_extr_years(years: Union[int, List[int]], columns: Union[str, List[str], NoneType] = None, exclude: bool = True, verbose: bool = False) -> pandas.core.frame.DataFrame
```

Load and merge x-data and bulletin data incrementally.

**Parameters:**

- `years (int/list)`: A single survey year or a list of survey years (e.g. 2019 or [2010, 2013, 2019])
- `columns (list/str)`: Either a string or a list containing all the variable names (as strings) to keep (returns all vars if None)
- `exclude (bool)`: If True and columns is None, then excludes all j-code variables
- `verbose (bool)`: True to output print messages, False for quietly

**Returns:**

pd.DataFrame: Pandas df containing the requested information

## `download_year`

```python
download_year(year: int, file_type: fileType = 'main_data', columns: Union[str, List[str], NoneType] = None, exclude: bool = True, timeout: int = 60, verbose: bool = False) -> pandas.core.frame.DataFrame
```

Download .dta from zip.

**Parameters:**

- `year (int)`: year to be imported
- `file_type (str)`: Either 'main_data' for x-variables, 'extr_data' for extract data, or 'rep_wgts' for replicate weights
- `columns (list/str)`: Either a string or a list containing all the variable names (as strings) to keep (returns all vars if None)
- `exclude (bool)`: If True and columns is None, then excludes all j-code variables
- `timeout (int)`: The amount of time the server has to connect to the SCF website and download the zip file
- `verbose (bool)`: True to output print messages, False for quietly

**Returns:**

pd.DataFrame: Pandas df containing the requested information

**Raises:**

- `ValueError`: Raises an exception for incorrect columns format
- `IOError`: Raises an exception if the zip file or the dta cannot be downloaded or read

## `download_years`

```python
download_years(years: Union[int, List[int]], file_type: fileType = 'main_data', columns: Union[str, List[str], NoneType] = None, exclude: bool = True, verbose: bool = False) -> pandas.core.frame.DataFrame
```

Download .dta from zip with robust error handling.

**Parameters:**

- `years (int/list)`: A single survey year or a list of survey years (e.g. 2019 or [2010, 2013, 2019])
- `file_type (str)`: Either 'main_data' for x-variables, 'extr_data' for bulletin data
- `columns (list/str)`: Either a string or a list containing all the variable names (as strings) to keep (returns all vars if None)
- `exclude (bool)`: If True and columns is None, then excludes all j-code variables
- `verbose (bool)`: True to output print messages, False for quietly

**Returns:**

pd.DataFrame: Pandas df containing the requested information

**Raises:**

- `ValueError`: Raises an exception for years not in the dataset or if concatenated dataframe is empty

