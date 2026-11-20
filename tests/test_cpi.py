"""
Unit tests for CPI adjustment module.

This module tests CPI data retrieval and adjustment calculations including
error handling for network failures and invalid inputs.

Author: Payton Crawford
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import requests

# Import functions to test
from scfpublic.cpi import read_excel_from_url, cpi_adjust, URL, minYear, maxYear, allYears

class TestReadExcelFromUrl:
    """Tests for read_excel_from_url function."""
    
    def test_successful_download(self):
        """Test successful download and parsing of Excel file."""
        # Create mock Excel file
        mock_df = pd.DataFrame({'YEAR': [2020, 2021], 'SEP': [100, 105]})
        mock_excel = BytesIO()
        mock_df.to_excel(mock_excel, index=False)
        mock_excel.seek(0)
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = mock_excel.getvalue()
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = read_excel_from_url(URL)
            
            assert isinstance(result, pd.DataFrame)
            mock_get.assert_called_once_with(URL, timeout=30)
    
    def test_network_failure(self):
        """Test IOError raised on network failure."""
        with patch('requests.get', side_effect=requests.exceptions.ConnectionError("Network error")):
            with pytest.raises(IOError, match="Failed to download file"):
                read_excel_from_url(URL)
    
    def test_timeout_failure(self):
        """Test IOError raised on timeout."""
        with patch('requests.get', side_effect=requests.exceptions.Timeout("Timeout")):
            with pytest.raises(IOError, match="Failed to download file"):
                read_excel_from_url(URL)
    
    def test_invalid_excel_content(self):
        """Test IOError raised when content is not valid Excel."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = b"Not an Excel file"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with pytest.raises(IOError, match="Error processing BLS file"):
                read_excel_from_url(URL)
    
    def test_custom_sheet_name(self):
        """Test reading from specific sheet."""
        mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_excel = BytesIO()
        with pd.ExcelWriter(mock_excel, engine='openpyxl') as writer:
            mock_df.to_excel(writer, sheet_name='Sheet2', index=False)
        mock_excel.seek(0)
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = mock_excel.getvalue()
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = read_excel_from_url(URL, sheet_name='Sheet2')
            assert isinstance(result, pd.DataFrame)

class TestCpiAdjust:
    """Tests for cpi_adjust function."""
    
    def test_successful_cpi_adjustment(self, sample_years_df, mock_cpi_data):
        """Test successful CPI adjustment calculation."""
        with patch('scfpublic.cpi.read_excel_from_url', return_value=mock_cpi_data):
            result = cpi_adjust(sample_years_df, base_year=2025)
            
            assert 'cpiadj' in result.columns
            assert 'cpilag' in result.columns
            assert result['cpiadj'].notna().all()
            assert result['cpilag'].notna().all()
    
    def test_missing_year_column(self):
        """Test ValueError raised when 'year' column is missing."""
        df = pd.DataFrame({'value': [100, 200]})
        
        with pytest.raises(ValueError, match="DataFrame must contain a 'year' column"):
            cpi_adjust(df)
    
    def test_invalid_base_year_too_low(self, sample_years_df):
        """Test ValueError raised when base_year is too low."""
        with pytest.raises(ValueError, match=f"base_year must be between {minYear} and {maxYear}"):
            cpi_adjust(sample_years_df, base_year=1950)
    
    def test_invalid_base_year_too_high(self, sample_years_df):
        """Test ValueError raised when base_year is too high."""
        with pytest.raises(ValueError, match=f"base_year must be between {minYear} and {maxYear}"):
            cpi_adjust(sample_years_df, base_year=2050)
    
    def test_no_valid_years_in_range(self):
        """Test ValueError raised when no valid years in DataFrame."""
        df = pd.DataFrame({'year': [1980, 1985, 1987]})  # None are in allYears
        
        with pytest.raises(ValueError, match=f"No valid years in range {minYear}-{maxYear}"):
            cpi_adjust(df)
    
    def test_mixed_valid_invalid_years(self, mock_cpi_data):
        """Test handling of mixed valid and invalid years."""
        df = pd.DataFrame({
            'year': [1989, 1990, 1992, 1993],  # 1990, 1993 are invalid
            'value': [100, 200, 300, 400]
        })
        
        with patch('scfpublic.cpi.read_excel_from_url', return_value=mock_cpi_data):
            result = cpi_adjust(df, base_year=2025)
            
            # Only valid years (1989, 1992) should have CPI values
            valid_mask = df['year'].isin(allYears)
            assert result.loc[valid_mask, 'cpiadj'].notna().all()
            assert result.loc[valid_mask, 'cpilag'].notna().all()
    
    def test_custom_base_year(self, sample_years_df, mock_cpi_data):
        """Test CPI adjustment with custom base year."""
        with patch('scfpublic.cpi.read_excel_from_url', return_value=mock_cpi_data):
            result = cpi_adjust(sample_years_df, base_year=2019)
            
            assert result['cpiadj'].notna().all()
            # Verify calculations use correct base year
            cpibase = mock_cpi_data.loc[mock_cpi_data['YEAR'] == 2019, 'SEP'].values[0]
            expected_cpiadj = cpibase / mock_cpi_data.loc[mock_cpi_data['YEAR'] == 1989, 'SEP'].values[0]
            assert np.isclose(result.loc[result['year'] == 1989, 'cpiadj'].values[0], expected_cpiadj)