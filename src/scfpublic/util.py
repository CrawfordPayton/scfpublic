"""
Utility functions for SCF

    - mconv: Monthly conversion function
    - aconv: Annual conversion function
    
Author: Payton Crawford
"""

import pandas as pd

ConversionFactors = {
    -7: 0, # Other/Not specified
    -1: 0, # Nothing
    0: 0, # Inap.
    1: 260, # Daily
    2: 52, # Weekly
    3: 26, # Every two weeks
    4: 12, # Monthly
    5: 4, # Quarterly
    6: 1, # Yearly
    8: 0, # Lump sum/one payment only; in total
    11: 2, # Twice per year
    12: 6, # Every two months
    13: 1/3, # Every three years
    14: 0, # By the piece/job
    18: 2080, # Hourly
    22: 0, # Varies
    23: 13, # 13 times a year/every 4 weeks
    24: 52/6, # Every 6 weeks
    31: 24 # Twice a month
}

def mconv(df: pd.DataFrame, value_col: str, period_col: str):
    """
    Convert SCF values to monthly equivalents based on time period codes following SAS Bulletin (https://www.federalreserve.gov/econres/files/bulletin.macro.txt).
    
    Args:
        df (pandas.DataFrame): The dataframe containing the SCF data
        value_col (str): Name of the column containing the values to convert
        period_col (str): Name of the column containing the period/frequency codes
    
    Returns:
        pandas.Series: Series with values converted to monthly amounts
        
    Examples:
    >>> df['payedu1'] = mconv(df, 'x7815', 'x7816')
    """
    
    # Convert to monthly factors by dividing by 12
    monthly_factors = {k: v/12 for k, v in ConversionFactors.items()}
    
    # Map period codes to factors and multiply by values
    return df[value_col] * df[period_col].map(monthly_factors)

def aconv(df: pd.DataFrame, value_col: str, period_col: str):
    """
    Convert SCF values to annual equivalents based on time period codes following SAS Bulletin mconv function (https://www.federalreserve.gov/econres/files/bulletin.macro.txt).
    
    Args:
        df (pandas.DataFrame): The dataframe containing the SCF data
        value_col (str): Name of the column containing the values to convert
        period_col (str): Name of the column containing the period/frequency codes
    
    Returns:
        pandas.Series: Series with values converted to annual amounts
        
    Examples:
    >>> df['foodhome'] = aconv(df, 'x3024', 'x3025')
    """
    
    # Map period codes to factors and multiply by values
    return df[value_col] * df[period_col].map(ConversionFactors)