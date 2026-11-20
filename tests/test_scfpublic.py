"""
Unit tests for SCFpublic module.

This module tests loading and processing of internal SCF datasets including
path generation, variable matching, and data optimization.

Author: Payton Crawford
"""


import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import zipfile
from typing import List
from datetime import datetime
from pydantic import ValidationError

# Import functions to test
import scfpublic
from scfpublic import (
    _match_vars,
    _optimize_dtypes,
    _get_url,
    _get_dta_path,
    minYear,
    maxYear,
    YEARS
)

class TestUrlGeneration:
    """Tests for URL and path generation functions"""
    
    def test_get_url_valid_year(self):
        """Test URL generation for valid years"""
        url = _get_url(2025, 'main_data')
        assert 'scf2025s.zip' in url
        assert url.startswith('https://www.federalreserve.gov')
    
    def test_get_url_pre_2004(self):
        """Test URL generation for years before 2004"""
        url = _get_url(1989, 'main_data')
        assert 'scf89s.zip' in url
    
    def test_get_url_invalid_year(self):
        """Test error for invalid year"""
        with pytest.raises(ValueError, match="Year .* not found"):
            _get_url(2023, 'main_data')
    
    def test_get_url_invalid_file_type(self):
        """Test error for invalid file type"""
        with pytest.raises(ValueError, match="file_type .* not found"):
            _get_url(2025, 'invalid_type')
    
    def test_get_dta_path_valid(self):
        """Test DTA path generation"""
        path = _get_dta_path(2025, 'main_data')
        assert path.endswith('.dta')
        assert 'p25i6' in path
    
    def test_get_dta_path_special_cases(self):
        """Test special case paths"""
        path_1992 = _get_dta_path(1992, 'main_data')
        assert 'p92i4' in path_1992
        
        path_2001 = _get_dta_path(2001, 'rep_wgts')
        assert 'scf2001rw1s' in path_2001

class TestMatchVars:
    """Tests for _match_vars() function"""
    
    @pytest.fixture
    def sample_vars(self):
        """Create sample variable list"""
        return ['X1', 'X2', 'X7607', 'NETWORTH', 'AGE', 'INCOME', 'Y1']
    
    def test_single_pattern_exact(self, sample_vars):
        """Test exact pattern matching"""
        result = _match_vars(sample_vars, ['X1'])
        assert result == ['X1']
    
    def test_single_pattern_wildcard(self, sample_vars):
        """Test wildcard pattern matching"""
        result = _match_vars(sample_vars, ['X*'])
        assert 'X1' in result
        assert 'X2' in result
        assert 'X7607' in result
        assert 'NETWORTH' not in result
    
    def test_multiple_patterns(self, sample_vars):
        """Test multiple pattern matching"""
        result = _match_vars(sample_vars, ['X*', 'AGE'])
        assert 'X1' in result
        assert 'AGE' in result
        assert 'INCOME' not in result
    
    def test_case_insensitive(self, sample_vars):
        """Test case-insensitive matching"""
        result = _match_vars(sample_vars, ['x1', 'networth'])
        assert 'X1' in result
        assert 'NETWORTH' in result
    
    def test_no_matches(self, sample_vars):
        """Test when no variables match"""
        result = _match_vars(sample_vars, ['Z*'])
        assert len(result) == 0

class TestOptimizeDtypes:
    """Tests for _optimize_dtypes function."""
    
    def test_optimize_float_columns(self):
        """Test downcasting of float columns."""
        df = pd.DataFrame({
            'float_col': [1.1, 2.2, 3.3],
        })
        df['float_col'] = df['float_col'].astype('float64')
        initial_dtype = df['float_col'].dtype
        
        result = _optimize_dtypes(df, verbose=False)
        
        # Should be downcasted to smaller float type
        assert result['float_col'].dtype.itemsize <= initial_dtype.itemsize
    
    def test_optimize_integer_columns(self):
        """Test downcasting of integer columns."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
        })
        df['int_col'] = df['int_col'].astype('int64')
        initial_dtype = df['int_col'].dtype
        
        result = _optimize_dtypes(df, verbose=False)
        
        # Should be downcasted to smaller int type
        assert result['int_col'].dtype.itemsize <= initial_dtype.itemsize
    
    def test_handle_null_integers(self):
        """Test handling of nullable integers."""
        df = pd.DataFrame({
            'int_with_na': [1, 2, np.nan, 4],
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        # Should use nullable integer type (Int8, Int16, Int32, or Int64)
        assert str(result['int_with_na'].dtype).startswith('Int')
        # Verify it can still hold NaN
        assert result['int_with_na'].isna().any()
    
    def test_nullable_integer_size_optimization(self):
        """Test that nullable integers are sized appropriately."""
        df = pd.DataFrame({
            'small_int': [1, 2, np.nan, 100],  # Should be Int8
            'medium_int': [1, 2, np.nan, 30000],  # Should be Int16
            'large_int': [1, 2, np.nan, 1000000000],  # Should be Int32
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert result['small_int'].dtype == 'Int8'
        assert result['medium_int'].dtype == 'Int16'
        assert result['large_int'].dtype == 'Int32'
    
    def test_optimize_boolean_columns(self):
        """Test optimization of boolean columns."""
        df = pd.DataFrame({
            'bool_col': [True, False, True],
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert result['bool_col'].dtype == 'bool'
    
    def test_nullable_boolean_columns(self):
        """Test handling of nullable boolean columns."""
        df = pd.DataFrame({
            'bool_with_na': [True, False, np.nan],
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert result['bool_with_na'].dtype == 'boolean'
    
    def test_datetime64_columns(self):
        """Test that datetime64 columns are preserved."""
        df = pd.DataFrame({
            'date_col': pd.date_range('2020-01-01', periods=3)
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date_col'])
    
    def test_datetime_object_conversion(self):
        """Test conversion of datetime objects within object columns."""
        df = pd.DataFrame({
            'date_obj': [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date_obj'])
    
    def test_datetime_string_conversion(self):
        """Test conversion of datetime strings within object columns."""
        df = pd.DataFrame({
            'date_str': ['2020-01-01', '2020-01-02', '2020-01-03']
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date_str'])
    
    def test_stata_datetime_conversion(self):
        """Test conversion of Stata-style datetime strings."""
        df = pd.DataFrame({
            'stata_date': ['1960-01-01 00:00:00', '1960-01-02 00:00:00', '1960-01-03 00:00:00']
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        assert pd.api.types.is_datetime64_any_dtype(result['stata_date'])
    
    def test_memory_reduction(self):
        """Test that memory usage is reduced."""
        df = pd.DataFrame({
            'float_col': np.random.randn(1000).astype('float64'),
            'int_col': np.random.randint(0, 100, 1000).astype('int64')
        })
        
        initial_memory = df.memory_usage(deep=True).sum()
        result = _optimize_dtypes(df, verbose=False)
        final_memory = result.memory_usage(deep=True).sum()
        
        assert final_memory <= initial_memory
    
    def test_mixed_column_types(self):
        """Test optimization with mixed column types."""
        df = pd.DataFrame({
            'float_col': [1.1, 2.2, 3.3],
            'int_col': [1, 2, 3],
            'int_with_na': [1, np.nan, 3],
            'date_col': pd.date_range('2020-01-01', periods=3),
            'text_col': ['a', 'b', 'c']
        })
        
        result = _optimize_dtypes(df, verbose=False)
        
        # Verify all columns are properly typed
        assert result['float_col'].dtype.name.startswith('float')
        assert result['int_col'].dtype.name.startswith('int')
        assert str(result['int_with_na'].dtype).startswith('Int')
        assert pd.api.types.is_datetime64_any_dtype(result['date_col'])
        assert result['text_col'].dtype == 'object'

class TestDownloadYear:
    """Tests for download_year() function"""
    
    @patch('requests.get')
    @patch('zipfile.ZipFile')
    @patch('pandas.read_stata')
    def test_successful_download(
        self, 
        mock_read_stata, 
        mock_zipfile, 
        mock_requests,
        sample_main_data,
        mock_read_stata_handler,
        mock_http_response,
        mock_zip_file
    ):
        """Test successful download and processing"""
        mock_requests.return_value = mock_http_response
        mock_zipfile.return_value = mock_zip_file
        mock_read_stata.side_effect = mock_read_stata_handler(sample_main_data)
        
        result = scfpublic.download_year(2025, columns=['x1'])
        assert isinstance(result, pd.DataFrame)
        assert 'year' in result.columns
    
    @patch('requests.get')
    @patch('zipfile.ZipFile')
    @patch('pandas.read_stata')
    def test_load_with_exclude_jvars(
        self,
        mock_read_stata,
        mock_zipfile,
        mock_requests,
        sample_main_data,
        mock_read_stata_handler,
        mock_http_response,
        mock_zip_file
    ):
        """Test loading with j-variables excluded"""
        mock_requests.return_value = mock_http_response
        mock_zipfile.return_value = mock_zip_file
        mock_read_stata.side_effect = mock_read_stata_handler(sample_main_data)
        
        result = scfpublic.download_year(1992, file_type='main_data', 
                              columns=None, exclude=True, verbose=False)
        
        assert 'year' in result.columns
        # Should have x variables but not j variables
        assert any(col.startswith('x') for col in result.columns)
        assert not any(col.startswith('j') for col in result.columns)
    
    @patch('requests.get')
    @patch('zipfile.ZipFile')
    @patch('pandas.read_stata')
    def test_load_with_column_filter_list(
        self,
        mock_read_stata,
        mock_zipfile,
        mock_requests,
        sample_main_data,
        mock_read_stata_handler,
        mock_http_response,
        mock_zip_file
    ):
        """Test loading with column filter as list"""
        mock_requests.return_value = mock_http_response
        mock_zipfile.return_value = mock_zip_file
        mock_read_stata.side_effect = mock_read_stata_handler(sample_main_data)
        
        result = scfpublic.download_year(1992, file_type='main_data',
                              columns=['x1', 'x7607'], exclude=False, verbose=False)
        
        assert 'year' in result.columns
        assert 'x1' in result.columns
        assert 'j1000' not in result.columns
    
    @patch('requests.get')
    @patch('zipfile.ZipFile')
    @patch('pandas.read_stata')
    def test_load_with_column_filter_str(
        self,
        mock_read_stata,
        mock_zipfile,
        mock_requests,
        sample_main_data,
        mock_read_stata_handler,
        mock_http_response,
        mock_zip_file
    ):
        """Test loading with column filter as list"""
        mock_requests.return_value = mock_http_response
        mock_zipfile.return_value = mock_zip_file
        mock_read_stata.side_effect = mock_read_stata_handler(sample_main_data)
        
        result = scfpublic.download_year(1992, file_type='main_data',
                              columns='x1 x7607', exclude=False, verbose=False)
        
        assert 'year' in result.columns
        assert 'x1' in result.columns
        assert 'j1000' not in result.columns

    def test_invalid_year(self):
        """Test with invalid year"""
        with pytest.raises(OSError):
            scfpublic.download_year(1985)  # Before 1989
    
    def test_columns_invalid_type(self):
        """Test invalid columns type"""
        with pytest.raises((ValidationError, TypeError)):
            scfpublic.download_year(2025, columns=123)
    
    @patch('requests.get')
    def test_network_error(self, mock_requests):
        """Test handling of network errors"""
        mock_requests.side_effect = Exception("Network error")
        with pytest.raises(IOError):
            scfpublic.download_year(2025)

class TestDownloadYears:
    """Tests for download_years() function"""
    
    @patch('scfpublic.download_year')
    def test_single_year(self, mock_download, sample_main_data):
        """Test downloading single year"""
        df = sample_main_data.copy()
        df['year'] = 2025
        mock_download.return_value = df
        
        result = scfpublic.download_years(2025)
        assert isinstance(result, pd.DataFrame)
    
    @patch('scfpublic.download_year')
    def test_multiple_years(self, mock_download, sample_main_data):
        """Test downloading multiple years"""
        def mock_year_func(year, **kwargs):
            df = sample_main_data.copy()
            df['year'] = year
            return df

        mock_download.side_effect = mock_year_func
        
        result = scfpublic.download_years([2022, 2025])
        assert isinstance(result, pd.DataFrame)
        assert 2022 in result['year'].values
        assert 2025 in result['year'].values
    
    @patch('scfpublic.download_year')
    def test_partial_failure(self, mock_download, sample_main_data, capsys):
        """Test handling when some years fail"""
        def side_effect(year, **kwargs):
            if year == 2022:
                raise Exception("Failed to download")
            df = sample_main_data.copy()
            df['year'] = year
            return df
        
        mock_download.side_effect = side_effect
        
        result = scfpublic.download_years([2022, 2025], verbose=True)
        captured = capsys.readouterr()
        assert isinstance(result, pd.DataFrame)

class TestDownloadMainExtrYears:
    """Tests for download_main_extr_years() function"""
    
    @patch('scfpublic.download_year')
    def test_single_year_merge(
        self, 
        mock_download, 
        sample_main_data, 
        sample_extr_data
    ):
        """Test merging main and extract data for single year"""
        def mock_func(year, file_type='main_data', **kwargs):
            if file_type == 'main_data':
                df = sample_main_data.copy()
            else:
                df = sample_extr_data.copy()
            df.rename(columns=str.lower, inplace=True)
            df['year'] = year
            return df
        
        mock_download.side_effect = mock_func
        
        result = scfpublic.download_main_extr_years(2025)
        assert isinstance(result, pd.DataFrame)

        result_cols_lower = [c.lower() for c in result.columns]
        assert 'y1' in result_cols_lower
        assert 'networth' in result_cols_lower
    
    @patch('scfpublic.download_year')
    def test_multiple_years_merge(
        self, 
        mock_download, 
        sample_main_data,
        sample_extr_data
    ):
        """Test merging multiple years"""
        def mock_func(year, file_type='main_data', **kwargs):
            if file_type == 'main_data':
                df = sample_main_data.copy()
            else:  # extr_data
                df = sample_extr_data.copy()

            df.rename(columns=str.lower, inplace=True)
            df['year'] = year
            return df
        
        mock_download.side_effect = mock_func
        
        result = scfpublic.download_main_extr_years([2022, 2025])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Should have both years
        assert len(result['year'].unique()) == 2
        assert 2022 in result['year'].values
        assert 2025 in result['year'].values
        
        # Should have columns from both main and extract
        result_cols_lower = [c.lower() for c in result.columns]
        assert 'y1' in result_cols_lower
        assert 'networth' in result_cols_lower