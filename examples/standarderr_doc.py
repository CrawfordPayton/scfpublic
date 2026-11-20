"""
Recreating Standard Error documentation example using SCFPublic package
Author: Payton Crawford
"""
from scfpublic import download_year, download_years, download_main_extr_years
from scfpublic.scfcombo import scfcombo_result, prepare_scf_data, scf_regression
import numpy as np
import statsmodels.formula.api as smf

#####
# Run regressions using loaded data
#####
vars = 'yy1 y1 income hhouses hdebt hmrthel edcl age wgt'

df = download_years(2025, columns= vars, file_type='extr_data')
rep_wgts = download_years(2025, file_type='rep_wgts')

print(df)

df['lninc'] = np.log(df['income'] + 1)
# Recreating the bulletin variable wgt0 also works for scf_estimate and does not require "estimand_type = 'total'" for aggregate estimations
#df['wgt0'] = df['wgt'] * 5

cut_df = df[df['age'].between(25,40)]
print(cut_df)

formula = 'lninc ~ C(edcl) + age + hhouses + hmrthel'

model = smf.wls(formula=formula, weights=cut_df['wgt'], data=cut_df).fit()
print(model.summary2(float_format="%.4f"))


scf_data = prepare_scf_data(df, rep_wgts, weight_col='wgt')

result = scf_regression(scf_data, formula = formula, subset=lambda df:  df['age'].between(25, 40))

result.display()

results_table = result.results_table()

print(results_table.loc[results_table['Variable'] == 'hhouses'])
