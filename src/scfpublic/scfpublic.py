'''
Downloads SCF files from https://www.federalreserve.gov/econres/scfindex.htm and formats them into a Pandas dataframe

Author: Payton Crawford
'''
# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import fnmatch
import gc
import requests
import zipfile
from io import BytesIO
import psutil
import os
from pydantic import validate_call
from typing import Optional, Union, List, Dict, Literal

# Years for datasets
minYear = 1989
maxYear = 2025
YEARS = list(range(minYear,maxYear+1,3))

# Base URL for Federal Reserve files
BASE_URL = "https://www.federalreserve.gov/econres/files/"

type fileType = Literal['main_data', 'rep_wgts', 'extr_data']
        
    
'''
Internal functions for creating URLs and data pathnames
''' 
def _generate_urls(years: List[int], base_url: str) -> Dict[int, Dict[str, str]]:
    """
    Generate all URLs in a nested dictionary structure.
    
    Returns:
        dict: {year: {'main_data': url, 'rep_wgts': url, 'extr_data': url}}
    """
    urls = {}
    for year in years:
        formatted_year = str(year)[-2:] if year < 2004 else year # Years pre-2004 have a two-digit extension 
        urls[year] = {
            'main_data': f"{base_url}scf{formatted_year}s.zip",
            'rep_wgts': f"{base_url}scf{formatted_year}rw1s.zip",
            'extr_data': f"{base_url}scfp{year}s.zip"
        }
    urls[2001]['rep_wgts'] = f"{base_url}scf2001rw1s.zip" # 2001 has differing formats between main data and replication weights
    return urls

def _generate_dta_pathnames(years: List[int]) -> Dict[int, Dict[str, str]]:
    """
    Generate all DTA pathnames in a nested dictionary structure.
    
    Returns:
        dict: {year: {'main_data': url, 'rep_wgts': url, 'extr_data': url}}
    """
    dta_pathnames = {}
    for year in years:
        formatted_year = str(year)[-2:] # Last two digits of each year
        dta_pathnames[year] = {
            'main_data': f"p{formatted_year}i6.dta",
            'rep_wgts': f"p{formatted_year}_rw1.dta",
            'extr_data': f"rscfp{year}.dta"
        }
    # Handle unique file names
    dta_pathnames[1992]['main_data'] = f"p92i4.dta"
    dta_pathnames[2001]['rep_wgts'] = f"scf2001rw1s.dta"
    return dta_pathnames

_URLS = _generate_urls(YEARS, BASE_URL)

_DTA_PATHNAMES = _generate_dta_pathnames(YEARS)

"""
Gets URLs and dta paths
"""
def _get_url(year: int, file_type: fileType = 'main_data') -> str:
    """
    Get the URL for a specific year and file type.
    
    Args:
        year (int): Year
        file_type (str): Type of file - 'main_data', 'rep_wgts', or 'extr_data'
    
    Returns:
        str: URL

    Raises:
        ValueError: Raises an exception for incorrect years and file types
    """
    # Error if year does not exist
    if year not in _URLS:
        raise ValueError(f"Year {year} not found in URLs")
    # Error if file type does not exist
    if file_type not in _DTA_PATHNAMES[year]:
        raise ValueError(f"file_type '{file_type}' not found for year {year}")
    
    return _URLS[year][file_type]

def _get_dta_path( 
    year: int, 
    file_type: fileType = 'main_data'
) -> str:
    """
    Get the expected path for a .dta file.
    
    Args:
        year (int): Year
        file_type (str): Type of file - 'main_data', 'rep_wgts', or 'extr_data'
    
    Returns:
        str: Expected file extension in zip file

    Raises:
        ValueError: Raises an exception for incorrect years and file types
    """
    # Error if year does not exist
    if year not in _DTA_PATHNAMES:
        raise ValueError(f"Year {year} not found in URLs")
    # Error if file type does not exist
    if file_type not in _DTA_PATHNAMES[year]:
        raise ValueError(f"file_type '{file_type}' not found for year {year}")
    
    return _DTA_PATHNAMES[year][file_type]

'''
String Matching Functions
'''
def _match_vars(
    all_vars: List[str], 
    patterns: Union[str, List[str]], 
    verbose: bool = False
) -> List[str]:
    """
    Match variable names against wildcard patterns and upper-case variables
    
    Args:
        all_vars (list): List of all variable names
        patterns (list/str): String or list of patterns with wildcards ('*')
        verbose (bool): True to output print messages, False for quietly
    
    Returns:
        list: List of matching variable names
    """
    # Create lowercase variable names for matching
    vars_lower = [col.lower() for col in all_vars]
    var_mapping = dict(zip(vars_lower, all_vars))
    
    # Match against lowercase
    matched_lower = set()
    for pattern in patterns:
        pattern_lower = pattern.lower()
        matches = fnmatch.filter(vars_lower, pattern_lower)
        matched_lower.update(matches)
    
    # Return original case names
    matched_original = [var_mapping[col] for col in sorted(matched_lower)]
    
    return matched_original

'''
Optimization Functions
'''
def _optimize_dtypes(
    df: pd.DataFrame, 
    verbose: bool = False
) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.

    Args:
        df (pd.DataFrame): Pandas dataframe with variables to be downcasted
        verbose (bool): True to output print messages, False for quietly
    
    Returns:
        pd.DataFrame: Pandas df with downcasted variables
    """
    start_mem = df.memory_usage(deep=True).sum() / 1e9
    
    if verbose:
        print(f"Initial memory: {start_mem:.2f} GB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Keep datetime columns as datetime64
        if pd.api.types.is_datetime64_any_dtype(col_type):
            continue
        
        # Handle object columns
        if col_type == 'object':
            # Try to convert to datetime first
            try:
                # Check if first non-null value looks like a datetime
                first_valid = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if first_valid is not None:
                    # Check if it's already a datetime object or looks like a date string
                    if hasattr(first_valid, 'strftime') or (isinstance(first_valid, str) and 
                        any(char in str(first_valid) for char in ['-', '/', ':'])):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        #if verbose:
                            #print(f"  Converted '{col}' to datetime64")
                        continue
            except (ValueError, TypeError):
                pass

             # Check if object column is actually boolean with NaN
            try:
                non_null = df[col].dropna()
                if len(non_null) > 0 and all(isinstance(v, (bool, np.bool_)) for v in non_null):
                    df[col] = df[col].astype('boolean')
                    #if verbose:
                       #print(f"  Converted '{col}' to boolean")
                    continue
            except (ValueError, TypeError):
                pass
            
            # If not datetime, keep as string (don't use category for parquet compatibility)
            df[col] = df[col].astype(str)
            continue
        
        # Downcast float columns
        if col_type.name.startswith('float'):
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0 and (non_null_values % 1 == 0).all():
                # These are integers with NaN, convert to nullable int
                min_val, max_val = non_null_values.min(), non_null_values.max()
                if min_val >= -128 and max_val <= 127:
                    df[col] = df[col].astype('Int8')
                elif min_val >= -32768 and max_val <= 32767:
                    df[col] = df[col].astype('Int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df[col] = df[col].astype('Int32')
                else:
                    df[col] = df[col].astype('Int64')
                #if verbose:
                    #print(f"  Converted '{col}' to nullable integer")
            continue
        
        # Handle integer columns with nullable types
        if col_type.name.startswith('int') or col_type.name.startswith('Int'):
            if df[col].isna().any():
                # Determine appropriate nullable int size based on range
                min_val, max_val = df[col].min(), df[col].max()
                if pd.isna(min_val) or pd.isna(max_val):
                    df[col] = df[col].astype('Int32')
                elif min_val >= -128 and max_val <= 127:
                    df[col] = df[col].astype('Int8')
                elif min_val >= -32768 and max_val <= 32767:
                    df[col] = df[col].astype('Int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df[col] = df[col].astype('Int32')
                else:
                    df[col] = df[col].astype('Int64')
            else:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            continue
        
        # Handle boolean columns with nulls
        if col_type == 'bool' and df[col].isna().any():
            df[col] = df[col].astype('boolean')
    
    end_mem = df.memory_usage(deep=True).sum() / 1e9
    
    if verbose:
        print(f"Final memory: {end_mem:.2f} GB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df

'''
Extracting dta files for zip files at https://www.federalreserve.gov/econres/scfindex.htm
'''
@validate_call
def download_year( 
    year: int, 
    file_type: fileType = 'main_data', 
    columns: Optional[Union[str, List[str]]] = None, 
    exclude: bool = True, 
    timeout: int = 60, 
    verbose: bool = False
) -> pd.DataFrame:
    """
    Download .dta from zip.

    Args:
        year (int): year to be imported
        file_type (str): Either 'main_data' for x-variables, 'extr_data' for extract data, or 'rep_wgts' for replicate weights
        columns (list/str): Either a string or a list containing all the variable names (as strings) to keep (returns all vars if None)
        exclude (bool): If True and columns is None, then excludes all j-code variables
        timeout (int): The amount of time the server has to connect to the SCF website and download the zip file
        verbose (bool): True to output print messages, False for quietly
    
    Returns:
        pd.DataFrame: Pandas df containing the requested information
    
    Raises:
        ValueError: Raises an exception for incorrect columns format
        IOError: Raises an exception if the zip file or the dta cannot be downloaded or read
    """

    # If you provide a string of words (i.e. yy1 y1 x7607...), this converts the string to a list of variables
    if isinstance(columns, str):
        columns = columns.split()
    # If a string, list, or no option is specified, return an error message
    elif (not (isinstance(columns, list))) & (not columns is None):
        raise ValueError(f"Variables specified must be in string or list format")

    # Ensure error check and add yy1/y1 to list if necessary
    if (not columns is None):
        if year == 1989:
            if 'xx1' not in columns:
                columns.append('xx1')
            if 'x1' not in columns:
                columns.append('x1')
        elif year > 1989:
            if 'yy1' not in columns:
                columns.append('yy1')
            if 'y1' not in columns:
                columns.append('y1')

    # Try to download zip file from FRB website, throw error otherwise
    try:
        url = _get_url(year, file_type)
        if verbose:
            print(f"Downloading from: {url}")
        # Read in zip file from URL
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Read dta file from zip
        zip_buffer = BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer) as zip_file:
            
            # Grab stored dta name from internal function
            dta_filename = _get_dta_path(year, file_type)

            if verbose:
                print(f"Reading: {dta_filename}")

            # Read dta file
            dta_content = zip_file.read(dta_filename)
            dta_buffer = BytesIO(dta_content)

            # Only read in first row (variable names)
            chunk_sample = pd.read_stata(dta_buffer, chunksize=1)
            df_sample = next(chunk_sample)
            all_cols = [word for word in df_sample.columns]

            # Determine which vars to keep based on columns specification and exclude boolean
            if exclude and columns is None and file_type == 'main_data':
                # Drop all j-codes
                read_vars = _match_vars(all_cols, ['x*', 'y*'], verbose=verbose)
            elif (
                (not exclude and columns is None and file_type == 'main_data') or # Exclude is false for main dataset 
                (columns is None and file_type in ['extr_data', 'rep_wgts']) # Keep all bulletin variables unless otherwise specified
            ):
                read_vars = all_cols
            else:
                # Match based on provided columns to keep
                read_vars = _match_vars(all_cols, columns, verbose=verbose)
                filter_vars = [item for item in columns if '*' not in item]

            if not read_vars:
                raise ValueError(f"Please provide at least one valid variable (e.g. x1) or specify columns=None") 
            
            # Reset buffer
            dta_buffer.seek(0)

            # Only read in specified variables
            df = pd.read_stata(dta_buffer, columns=read_vars)

            # Convert all columns to lowercase for ease of concatenation/merge
            df.rename(columns=str.lower, inplace=True)

            # 1989 has x1 and xx1 variables in place of y1 and yy1 variables for later years
            if year == 1989:
                df = df.copy()
                if 'xx1' in df.columns:
                    df['yy1'] = df['xx1']
                    df.drop(columns=['xx1'], inplace=True)
                if 'x1' in df.columns:
                    df['y1'] = df['x1']
                    df.drop(columns=['x1'], inplace=True)

        # If all variables are selected, optimize dtypes by downcasting
        if columns is None:
            df = _optimize_dtypes(df, verbose=verbose)
            df = df.copy()

        df['year'] = year

        if verbose:
            print(f"Loaded: {len(df):,} rows | {len(df.columns)} columns")
        
        return df
        
    # Failure to download file
    except requests.exceptions.RequestException as e:
        raise IOError(f"Failed to download file: {e}")
    # Failure to unzip file
    except zipfile.BadZipFile:
        raise IOError("Downloaded file is not a valid zip file")
    # Error reading in zip file
    except Exception as e:
        raise IOError(f"Error processing zip file: {e}")

@validate_call
def download_years( 
    years: Union[int, List[int]],
    file_type: fileType = 'main_data', 
    columns: Optional[Union[str, List[str]]] = None, 
    exclude: bool = True, 
    verbose: bool = False
) -> pd.DataFrame:
    """
    Download .dta from zip with robust error handling.
    
    Args:
        years (int/list): A single survey year or a list of survey years (e.g. 2019 or [2010, 2013, 2019])
        file_type (str): Either 'main_data' for x-variables, 'extr_data' for bulletin data
        columns (list/str): Either a string or a list containing all the variable names (as strings) to keep (returns all vars if None)
        exclude (bool): If True and columns is None, then excludes all j-code variables
        verbose (bool): True to output print messages, False for quietly
    
    Returns:
        pd.DataFrame: Pandas df containing the requested information

    Raises:
        ValueError: Raises an exception for years not in the dataset or if concatenated dataframe is empty
    """
    years = sorted(set([years] if isinstance(years, int) else years))
    if not years:
        raise ValueError("Please specify at least one year")

    # Raise an error if there are invalid years specified
    invalid_years = [y for y in years if y not in YEARS]
    if invalid_years:
        raise ValueError(f"Years {invalid_years} not found. Valid survey years: {YEARS}")

    # Concatenates together dataframes, stores total memory used, and any failures
    result_df = None
    total_memory = 0
    failed_years = []

    for year in years:
        try:
            # Download dataframe
            df = download_year(year, file_type=file_type, columns=columns, exclude=exclude, verbose=verbose)

            # Drop any all NA columns
            df = df.dropna(axis=1, how='all').copy()

            if verbose:
                # Track memory
                mem = df.memory_usage(deep=True).sum()
                total_memory += mem
                print(f"Loaded {year}: {mem/1e9:.2f} GB (Total: {total_memory/1e9:.2f} GB)")
            
            if result_df is None:
                result_df = df
            else:
                result_df = pd.concat([result_df, df], ignore_index=True, copy=False)
                del df 
                gc.collect()

        # Throw error if year isn't loaded correctly
        except Exception as e:
            print(f"Error loading {year}: {e}")
            failed_years.append(year)
            continue


    # If dataframe is empty, throw error
    if (result_df is None) or (result_df.empty):
        raise ValueError(f"Dataframe is empty")

    if (len(failed_years) != 0) and (verbose):
        print(f'Warning: Failed to load years {failed_years}')

    # Downcast variables
    result_df = _optimize_dtypes(result_df, verbose=verbose)

    return result_df

@validate_call
def download_main_extr_years( 
    years: Union[int, List[int]],
    columns: Optional[Union[str, List[str]]] = None, 
    exclude: bool = True, 
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load and merge x-data and bulletin data incrementally.
    Args:
        years (int/list): A single survey year or a list of survey years (e.g. 2019 or [2010, 2013, 2019])
        columns (list/str): Either a string or a list containing all the variable names (as strings) to keep (returns all vars if None)
        exclude (bool): If True and columns is None, then excludes all j-code variables
        verbose (bool): True to output print messages, False for quietly
    
    Returns:
        pd.DataFrame: Pandas df containing the requested information
    """
    years = sorted(set([years] if isinstance(years, int) else years))

    if not years:
        raise ValueError("Please specify at least one year")

    invalid_years = [y for y in years if y not in YEARS]
    if invalid_years:
        raise ValueError(f"Years {invalid_years} not found. Valid survey years: {YEARS}")

    single_year = len(years) == 1

    merge_keys = ['y1', 'year']
    result_df = None
    failed_years = []

    for year in years:
        try:
            # Load both file types for this year
            main_df = download_year(year, file_type='main_data',
                                columns=columns, exclude=exclude, verbose=verbose)
            extract_df = download_year(year, file_type='extr_data',
                                   columns=columns, verbose=verbose)

            # Confirm merge keys exist in both files
            missing = [k for k in merge_keys if k not in main_df.columns or k not in extract_df.columns]
            if missing:
                raise ValueError(f"Merge keys {missing} missing from main or extract data for {year}")

            # xx1 is implicate-level, never merge on it from the bulletin data
            extract_df = extract_df.drop(columns=['yy1'], errors='ignore')

            # Keep overlapping columns from the main data
            overlap = (set(main_df.columns) & set(extract_df.columns)) - set(merge_keys)
            if overlap:
                if verbose:
                    print(f"\t{year}: dropping duplicate bulletin columns {sorted(overlap)}")
                extract_df = extract_df.drop(columns=list(overlap))

            # Bulletin data must be household-level (one row per y1)
            if extract_df.duplicated(subset=merge_keys).any():
                raise ValueError(f"Bulletin data for {year} has duplicate y1 rows; expected one per household")

            # Many x-data rows (implicates) to one bulletin row (household)
            year_merged = main_df.merge(
                extract_df, on=merge_keys, how='left',
                validate='many_to_one', indicator=True
            )

            year_merged = year_merged.drop(columns='_merge')

            # Concatenate by year pair
            if result_df is None:
                result_df = year_merged
            else:
                result_df = pd.concat([result_df, year_merged], ignore_index=True, copy=False)

            # Clean up unused-files
            del main_df, extract_df, year_merged
            gc.collect()

        except Exception as e:
            # For a single year, surface the real error instead of a generic empty-frame error
            if single_year:
                raise
            print(f"Error loading {year}: {e}")
            failed_years.append(year)
            continue

    if result_df is None or result_df.empty:
        raise ValueError("Merged dataframe is empty")

    # Final optimization
    if verbose:
        print("Optimizing final merged dataframe...")
    result_df = _optimize_dtypes(result_df, verbose=verbose)

    if failed_years and verbose:
        print(f"Warning: Failed to load years: {failed_years}")

    return result_df
