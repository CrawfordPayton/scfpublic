# Survey of Consumer Finances Replicate Weights Documentation
#### Author: Payton Crawford
#### Last Updated: November 18th, 2026

## Table of Contents
* [1. Introduction](#1-introduction)
* [2. Replicate Weights](#2-replicate-weights)
* [3. Variances](#3-variances)
    * [3.1 Multiple Imputation Variance](#31-multiple-imputation-variance)
    * [3.2 Sampling Variance](#32-sampling-variance)
    * [3.3 Combined Variance](#33-combined-variance)
* [4. Combined Standard Error Programs](#4-programs-for-computing-combined-standard-error)
    * [4.1 Python Package scfcombo](#41-python-scfcombo)
    * [4.2 Stata Module SCFCOMBO](#42-stata-module-scfcombo)
    * [4.3 SAS Macro MEANIT](#43-sas-macro-meanit)
    * [4.4 R Function scf_MIcombine](#44-r-function-scf_micombine)
* [5. Multiple Imputation Resources](#5-further-documentation-on-multiple-imputations)
    * [5.1 Papers](#51-multiple-imputation-papers)
    * [5.2 Programs](#52-multiple-imputation-programs)

## 1. Introduction
When using the Survey of Consumer Finances (SCF) data, standard error calculations can over-estimate the reliability of regressions and other statistics unless imputation error and sample variability error are accounted for. The SCF uses multiple imputations to handle missing data and uses replicate weights to estimate sampling variability error. Documentation for both of these resources can be found below. As an example of what effect the failure to account for these errors can have, here is an example of standard errors and p-values from different regressions:

Regression estimates of correlation between income and home-ownership among families with head aged 25-40
| Variable | (1) | (2) | (3) |
| :--- | :--- | :--- | :--- |
| **I(Homeowner)** | $0.150$ | $0.150$ | $0.150$ |
| | (0.046) | (0.105) | (0.097) |
| | p=0.001 | p=0.152 | p=0.122 |
| **$N$** | 4,995 | 999 | 999 |

*Note: OLS regression estimates of log of family income on left-hand side and homeowner
indicator variable on right hand side using the 2025 Survey of Consumer Finances. Includes controls for education, age, and mortgage indicator
variable. Families with head of household aged 25-40 only. Estimated standard errors are in
parentheses. Column (1) shows standard error without correction for multiple imputation or
sampling variance. Column (2) shows an estimated standard error with a correction for multiple
imputation [from Stata ado micombine], but still no correction for sampling variance, and column
(3) shows an estimated standard error with a correction for multiple imputation and the sampling
variance [from Python function scf_regression].

Adjusting for both types of error caused the p-value to increase from 0.001 to over 0.1, which substantially alters the statistical significance of the result. Additionally, note that accounting for imputation error will always increase the standard error while adjusting for sampling error may increase or decrease the standard error. The adjustment in number of observations occurs as imputation error accounts for the pooled estimate across implicates, so the sample size will be divided by 5.

## 2. Replicate Weights
Replicate weights are constructed in the following process:
1. Following bootstrapping theory, we take 999 samples from the SCF data in a way that mirrors the original sampling process. This means that in any given bootstrap replicate sample, an SCF family could be sampled zero times or multiple times (represented by `mm{i}`, where `i` represents the sample, in the replicate weights dataset).
2. Each time the data is resampled, a new version of the data is created (i), which provides a sample that could have been drawn originally.
    - Households may get a different weight each time a sample is drawn (represented by `wt1b{i}` in the replicate weights dataset).
    - The weights sum to the total US population, similar to the base SCF weights.
3. Because each bootstrap replicate sample will have a slightly different composition, they will provide a distribution for each variable when take together. This distribution is used to estimate sampling variability.
4. We estimate sampling variability only on the first implicate.

## 3. Variances

### 3.1 Multiple Imputation Variance
1. Missing data in the SCF are imputed 5 times, which means that each SCF family has 5 separate observations (called "implicates") in the final data (represented by the final digit in `y1` in the public data).
2. The estimated standard error of a statistic is smaller than the true standard error when estimated without correcting for multiple imputation.
3. The multiple imputation variability for a weighted estimator function $F(x, w)$, where $x$ is the dataset and $w$ is the weights, is calculated as follows:<br>

For each implicate $i$, where $I$ is the total number of implicates,

$$
\hat{\beta}_{imp, i} = F(x_i, w_i)
$$

The pooled estimate is then calculated as the mean of estimates across implicate,

$$
\hat{\beta}_{imp} = (1/I) * \sum_{i=1}^I(\hat{\beta}_{imp, i})
$$

Thus, the imputation variance is,

$$
V_{imp} = (1/(I-1)) * \sum_{i=1}^I(\hat{\beta}_{imp, i} - \hat{\beta}_{imp})^2
$$

### 3.2 Sampling Variance
1. Sampling weights are provided due to using a sample rather than the entire population.
2. Estimated standard errors can be further corrected to include an estimate of sampling variability. Correcting for sampling variability can increase or decrease the standard errors.
3. Sampling variability is estimated by the set of bootstrapped replicate weights described above for only the first implicate.
4. The sampling variability for a weighted estimator function $F(x, w)$, where $x$ is the dataset and $w$ is the weights, is calculated as follows:<br>

For each replicate $r$ (using only the first implicate, $i$=1), where $R$ is the total number of replicates,

$$
\hat{\beta}_{sam, r} = F(x_1, w_r)
$$

The pooled estimate is then calculated as the mean of estimates across replicates,

$$
\hat{\beta}_{sam} = (1/R) * \sum_{r=1}^R(\hat{\beta}_{sam, r})
$$

Thus, the sampling variance is,

$$
V_{sam} = (1/(R-1)) * \sum_{r=1}^R(\hat{\beta}_{sam, r} - \hat{\beta}_{sam})^2
$$

### 3.3 Combined Variance
Then, the variances are combined using Rubin's finite imputation correction, $(I + 1) / I$,

$$
V_{adj} = ((I+1)/I) * V_{imp} + V_{sam}
$$

Thus, the adjusted standard error is,

$$
SE_{adj} = \sqrt{V_{adj}}
$$

## 4. Programs for Computing Combined Standard Error
### 4.1 Python Package scfpublic
The Python package `scfpublic` allows the user to download and use SCF data. An additional subpackage, `scfcombo`, formats the SCF data and replicate weights with the function `scf_prepare_data()`, and standard errors are calculated with the function `scf_estimate()`, which takes the prepared data and a user-defined function. For convenience, wrapper functions for means, quantiles, and regressions are available as examples. Full documentation is available [here](https://github.com/CrawfordPayton/scfpublic/blob/main/variances/variances.md).<br>
&emsp;Citation: Payton Crawford (2026). scfpublic. Python package. ver. 1.0.5. https://github.com/CrawfordPayton/scfpublic.

### 4.2 Stata Module SCFCOMBO
The user-written .ado file, SCFCOMBO, for Stata calculates and combines the imputation uncertainty and bootstrapped standard errors for estimation commands run on the SCF. It can be installed in Stata using `ssc install scfcombo`. Full documentation is available [here](https://ideas.repec.org/c/boc/bocode/s458017.html).<br>
&emsp;Citation: Karen Pence, 2015. "SCFCOMBO: Stata module to estimate errors using the Survey of Consumer Finances," Statistical Software Components S458017, Boston College Department of Economics, revised 04 Oct 2015. 
 
### 4.3 SAS Macro MEANIT
The SAS Macro **MEANIT** in the Sampling Error section of the [SCF Bulletin code](https://www.federalreserve.gov/econres/files/codebk2022.txt) details how multiple imputation variance, sampling variance, and combined variance are computed.<br>
&emsp;Citation: "Codebook for the 2025 Survey of Consumer Finances." Board of Governors of the Federal Reserve System, 2026. https://www.federalreserve.gov/econres/files/codebk2025.txt. 

### 4.4 R Function scf_MIcombine
Another programmer has created scripts for converting and working with SCF data in R. These scripts are available to download from [here](https://guilhermejacob.github.io/context/1.6-survey-of-consumer-finances-scf.html#survey-of-consumer-finances-scf).<br>
&emsp;Citation: Jacob, Guilherme, Anthony Damico, and Djalma Pessoa. "1.6 Survey of Consumer Finances." In *Poverty and Inequality with Complex Survey Data*. https://www.convey-r.org/index.html. 


## 5. Further Documentation on Multiple Imputations
### 5.1 Multiple Imputation Papers
- Kennickell, Arthur B. "Imputation of the 1989 Survey of Consumer Finances: Stochastic relaxation and multiple imputation." In Proceedings of the survey research methods section of the American Statistical Association, vol. 1, no. 10, p. 41. 1991.
    - [Link](https://www.federalreserve.gov/econresdata/scf/files/imp89.pdf)
- Kennickell, Arthur B. "Multiple imputation in the Survey of Consumer Finances." Statistical Journal of the IAOS 33, no. 1 (2017): 143-151.
    - [Link](https://www.federalreserve.gov/Pubs/OSS/oss2/papers/impute98.pdf)
- Kennickell, Arthur B. "Multiple imputation and disclosure protection: The case of the 1995 Survey of Consumer Finances." Record Linkage Techniques 1997 (1997): 248-267.
    - [Link](https://www.federalreserve.gov/pubs/OSS/oss2/papers/expers95.pdf)
- Kennickell, Arthur B., Douglas A. McManus, and Freddie Mac. "Multiple Imputation of the 1983 and 1989 Waves of the SCF." In Proceedings of the Survey Research Methods Section, American Statistical Association, pp. 523-528. 1994.
    - [Link](https://www.federalreserve.gov/econresdata/scf/files/panelimputationasa94.pdf)

### 5.2 Multiple Imputation Programs
- SAS Macro MISECOMP in the [SCF Codebook](https://www.federalreserve.gov/econres/files/codebk2022.txt).<br>
    Citation: "Codebook for the 2025 Survey of Consumer Finances." Board of Governors of the Federal Reserve System, 2026. https://www.federalreserve.gov/econres/files/codebk2025.txt. 
- Stata .ado file MICOMBINE. Full documentation found [here](https://ideas.repec.org/c/boc/bocode/s446602.html).<br>
    Citation: Patrick Royston, 2006. "ICE: Stata module for multiple imputation of missing values," Statistical Software Components S446602, Boston College Department of Economics, revised 25 Oct 2014. 
