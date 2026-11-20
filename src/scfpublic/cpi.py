'''
Stores cpi adjustment formulas using data from https://www.bls.gov/cpi/research-series/r-cpi-u-rs-home.htm

Author: Payton Crawford
'''
import pandas as pd
import requests
from io import BytesIO

# URL for BLS data
URL = 'https://www.bls.gov/cpi/research-series/r-cpi-u-rs-allitems.xlsx'

# Current years of dataset
minYear = 1989
maxYear = 2025
allYears = list(range(minYear,maxYear+1,3))

def read_excel_from_url(url: str, sheet_name: int = 0, **kwargs):
    """
    Download Excel file from BLS URL and read into DataFrame.
        
    Args:
        url (str): URL of excel file
        sheet_name (int): Sheet name or index (default: 0 = first sheet)
        **kwargs: Additional arguments for pd.read_excel()
    
    Returns:
        pd.DataFrame: Pandas dataframe containing Excel file from URL

    Raises:
        IOError: Failure to download or process BLS file
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Read Excel from memory
        excel_buffer = BytesIO(response.content)
        df = pd.read_excel(excel_buffer, sheet_name=sheet_name, **kwargs)

        return df
        
    # Failure to download file
    except requests.exceptions.RequestException as e:
        raise IOError(f"Failed to download file: {e}")
    # Error reading in file
    except Exception as e:
        raise IOError(f"Error processing BLS file: {e}")

def cpi_adjust(df: pd.DataFrame, base_year: int = maxYear):
    '''
    Appends cpiadj and cpilag variables based on 'year' variable in dataset.

    Args:
        df (pd.DataFrame): Pandas dataframe to have CPI variables added to
        base_year (int): Year to use as base for for CPI base value (default: 2022)

    Returns:
        pd.DataFrame: Original dataframe with new cpiadj and cpilag variables merged on year

    Raises:
        ValueError: Provided invalid year, DataFrame lacks year variable, or DataFrame lacks correct years
    '''
    # Grab years from dataframe
    if 'year' not in df.columns:
        raise ValueError('DataFrame must contain a \'year\' column')
    df_years = df['year'].unique()

    # Validate base year is in range
    if base_year < minYear or base_year > maxYear:
        raise ValueError(f"base_year must be between {minYear} and {maxYear}, got {base_year}")

    # Only read in valid years (triennial survey)
    years_in_range = [y for y in df_years if y in allYears]
    if len(years_in_range) == 0:
        raise ValueError(f"No valid years in range {minYear}-{maxYear}")

    cpi_df =  read_excel_from_url(URL, skiprows=5, header=0)

    cpibase = cpi_df.loc[cpi_df['YEAR'] == base_year, 'SEP'].values[0]
    df['cpiadj'] = None
    df['cpilag'] = None
    for year in years_in_range:
        # CPI Adjustments based on release of survey
        df.loc[df['year']==year, 'cpiadj'] = cpibase  / cpi_df.loc[cpi_df['YEAR'] == year, 'SEP'].values[0]
        # CPI Lags from previous year to year of release
        df.loc[df['year']==year, 'cpilag'] = cpi_df.loc[cpi_df['YEAR'] == year, 'AVG'].values[0] / cpi_df.loc[cpi_df['YEAR'] == (year-1), 'AVG'].values[0]
    
    return df