"""
Historical Table 5 recreation using SCFPublic package
Author: Payton Crawford
"""
import scfpublic
import numpy as np
import pandas as pd

vars = (
    "y1 yy1 wgt "
    "fin liq cds nmmf stocks bond retqliq savbnd cashli othma othfin nfin "
    "hfin hliq hcds hnmmf hstocks hbond hretqliq hsavbnd hcashli hothma hothfin hnfin"
)

print(vars)

df = scfpublic.download_main_extr_years(range(1989, 2026, 3), columns=vars)

vars = "liq cds nmmf stocks bond retqliq savbnd cashli othma fin othfin nfin"
vars_list = vars.split()

for var in vars_list:
    if f'h{var}' in df:
        df.loc[df[f'h{var}'] == 0, f'{var}'] = np.nan
        df.loc[df[f'{var}'] == 0, f'{var}'] = np.nan
        df.loc[pd.isna(df[f'{var}']), f'h{var}'] = 0

    else:
        print(f'h{var}')

def weighted_sum(df, col, weight_col):
    """Calculate weighted sum for a single column, ignoring NaNs"""
    mask = ~df[col].isna()
    if mask.sum() > 0:
        return (df.loc[mask, col] * df.loc[mask, weight_col]).sum()
    else:
        return np.nan

def weighted_percentages(df, numerator_vars, denominator_var, weight_col='weight'):
    """
    Calculate weighted percentage: weighted sum of each numerator variable 
    divided by weighted sum of denominator variable
    Returns array of length = len(numerator_vars)
    """
    # Calculate weighted sum for denominator (var8)
    denom_sum = weighted_sum(df, denominator_var, weight_col)
    
    # Calculate weighted percentage for each numerator variable (var1-7)
    percentages = []
    for col in numerator_vars:
        num_sum = weighted_sum(df, col, weight_col)
        percentage = (num_sum / denom_sum) * 100
        percentages.append(percentage)
    
    return np.array(percentages)

percentages_index = [
                    'Transaction accounts',
                    'Certificates of deposit',
                    'Savings bonds',
                    'Bonds',
                    'Stocks',
                    'Pooled investment funds (excluding money market funds)',
                    'Retirement accounts',
                    'Cash value life insurance',
                    'Other managed assets',
                    'Other',
                    'Total',
                    'Financial assets as a share of total assets'
                ]

percentages = pd.DataFrame()
percentages.index = percentages_index

vars = "liq cds savbnd bond stocks nmmf retqliq cashli othma othfin fin"
vars_list = vars.split()

for year in range(1989,2026,3):

    df_cut = df[df['year'] == year]
    # Calculate percentages
    year_df = weighted_percentages(df_cut, vars_list, 'fin', 'wgt')

    fin_percent_assets = weighted_sum(df_cut, 'fin', 'wgt') / (weighted_sum(df_cut, 'fin', 'wgt') + weighted_sum(df_cut, 'nfin', 'wgt'))
    year_df = np.append(year_df, fin_percent_assets * 100)
    percentages[f'{year}'] = year_df


print(percentages.round(1)) 