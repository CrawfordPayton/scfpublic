# Python Compiler for Public SCF Files (scfpublic)
#### Author: Payton Crawford
#### Last Updated: November 19th, 2026

## Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Directory Structure](#directory-structure)
    * [Docs](#docs)
    * [Examples](#examples)
    * [Src](#srcscfpublic)
    * [Tests](#tests)
* [Packages and Functions](#packages-and-functions)

### Introduction
The purpose of these Python package is to provide a centralized package for downloading the Survey of Consumer Finances files and computing standard errors that are corrected for multiple imputation and the sample design. By doing so, I aim to make it easier and clearer how to properly use SCF data. Please refer to the official Federal Reserve Board website for the [Survey of Consumer Finances](https://www.federalreserve.gov/econres/scfindex.htm). Their website includes the datasets, codebooks (for example, [2025](https://www.federalreserve.gov/econres/files/codebk2025.txt)) containing information by each year for each of the x-variables, and an updated [SAS bulletin macro](https://www.federalreserve.gov/econres/files/bulletin.macro.txt) to define each of the bulletin variables. Additionally, they have a [codebook search](https://www.federalreserve.gov/apps/scfcb), a [net worth flowchat](https://www.federalreserve.gov/econres/files/Networth%20Flowchart.pdf), materials from [previous surveys](https://www.federalreserve.gov/econres/scf-previous-surveys.htm), and other relevant materials.

### Installation
From anywhere in your terminal, run `pip install scfpublic`. This will run commands from `pyproject.toml` to install the Python package and its minimum dependencies locally in that environment. From there, you will be able to access any of the Python files in `src/scfpublic` by running the command `import scfpublic` at the top of a Python script. <br>
- The Python dependencies that will be installed are listed in `requirements.txt`.
- For further in-depth details on how to utilize the package, please refer to the Jupyter Notebook in examples.

### Directory Structure
- #### examples
    - Stores Jupyter notebooks detailing how to use each respective package with examples and base Python files showing examples of usage.
- #### src/scfpublic
    - Stores the Python files that are used each time the package is imported.
        - scfpublic has the scfpublic package, which is used to load x-data, bulletin, and replicate weights files from their current storages into Python as a Pandas dataframe.
        - cpi.py has the cpi package, which is used for creating the cpiadj and cpilag variables that mimics the [SCF bulletin](https://www.federalreserve.gov/econres/files/bulletin.macro.txt).
        - util.py has the util package, which contains annual and monthly conversion functions following the [SCF bulletin](https://www.federalreserve.gov/econres/files/bulletin.macro.txt).
        - #### /scfcombo
            - scfcombo.py has the scfcombo package, which is used to convert the SCF data and replicate weights files to compute imputation error, sampling error, and combined errors for user-defined analysis.
            - stats.py has the scfcombo.stats package, which contains convenience wrapper functions around scf_estimate.
- #### tests
    - Stores Python test files for each of the above packages.
    - To run the tests for each program, run the command `python3 -m pytest {tests directory}` in your terminal.
- #### variances
    - Stores `variances.md` details how the combined standard errors are computed and how to use the respective functions for your programming language of choice.

### Packages and Functions
- `scfpublic`
    - Functions
        - `download_year` - Loads year, variables, and file type requested for public SCF data
        - `download_years` - Loads multiple years, variables, and file type requested for public SCF data
        - `download_main_extr_years` - Loads merged main and extract data using years and variables for public SCF data
    - `cpi` - Package for adding CPI adjustments by year
        - Functions
            - `read_excel_from_url` - Reads CPI-U-RS-ALLITEMS Excel file
            - `cpi_adjust` - Creates cpiadj and cpilag variables by year
    - `scfcombo` - Package for imputation and sampling error corrections
        - DataClass
            - `scfcombo_result` - Data class that stores statistic and regression information
                - Functions
                    - `display` - Outputs formatted results to terminal
                    - `summary` - Outputs data class information
        - Functions
            - `prepare_scf_data` - Prepares SCF data and replicate weights file into a dataset usable for scf_estimate
            - `scf_estimate` - Takes user functions and outputs pooled estimate, imputation variance, sampling variance, and combined variance into a scfcombo_result object
        - `stats` - Convenience functions for basic statistics applications using `scf_estimate`
            - Functions
                - `scf_mean` - Convenience wrapper for mean estimation around scf_estimate
                - `scf_quantile` - Convenience wrapper for median estimation around scf_estimate
                - `scf_regression` - Convenience wrapper for regression estimation around scf_estimate
                - `scf_total` - Convenience wrapper for aggregate estimation around scf_estimate
    - `util` - Package for basic SCF functions
        - Functions
            - `aconv` - Annual conversion function
            - `mconv` - Monthly conversion function

Citation: Payton Crawford (2026). scfpublic. Python package. ver. 1.0.1. https://github.com/CrawfordPayton/scfpublic.

Bibliography:

&emsp;Board of Governors of the Federal Reserve System. (2026). 2025 Survey of Consumer Finances [Dataset]. https://doi.org/10.17016/8799.1.
