"""
Unit tests for utility conversion functions.

This module tests the mconv and aconv functions for SCF data processing.

Author: Payton Crawford
"""

import pytest
import pandas as pd
import numpy as np

from scfpublic.util import mconv, aconv

class TestMconv:
    """Tests for mconv (monthly conversion) function."""
    
    def test_conversion_positive_values(self):
        """Test conversion with positive values."""
        df = pd.DataFrame({
            'amount': [52000, 12000, 100000],  # weekly, monthly, annual
            'period': [2, 4, 6]
        })
        
        result = mconv(df, 'amount', 'period')
        
        # Weekly: 52000 * (52/12) = 225333.33
        # Monthly: 12000 * (12/12) = 12000
        # Annual: 100000 * (1/12) = 8333.33
        assert np.isclose(result.iloc[0], 52000 * 52/12)
        assert np.isclose(result.iloc[1], 12000)
        assert np.isclose(result.iloc[2], 100000 / 12)
    
    def test_conversion_negative_values(self):
        """Test conversion with negative values (debts)."""
        df = pd.DataFrame({
            'amount': [-52000, -12000, -100000],
            'period': [2, 4, 6]
        })
        
        result = mconv(df, 'amount', 'period')
        
        # Should preserve negative sign
        assert result.iloc[0] < 0
        assert result.iloc[1] < 0
        assert result.iloc[2] < 0
        assert np.isclose(result.iloc[0], -52000 * 52/12)
        assert np.isclose(result.iloc[1], -12000)
        assert np.isclose(result.iloc[2], -100000 / 12)
    
    def test_conversion_with_na_values(self):
        """Test conversion with NA/missing values."""
        df = pd.DataFrame({
            'amount': [52000, np.nan, 100000],
            'period': [2, 4, 6]
        })
        
        result = mconv(df, 'amount', 'period')
        
        # NA should remain NA
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])
    
    def test_conversion_zero_factors(self):
        """Test conversion with zero factors (one-time payments)."""
        df = pd.DataFrame({
            'amount': [1000, 2000, 3000],
            'period': [-1, 0, 8]  # All have 0 conversion factor
        })
        
        result = mconv(df, 'amount', 'period')
        
        # All should be zero
        assert result.iloc[0] == 0
        assert result.iloc[1] == 0
        assert result.iloc[2] == 0
    
    def test_conversion_hourly_rate(self):
        """Test conversion for hourly wages."""
        df = pd.DataFrame({
            'amount': [25.0],  # $25/hour
            'period': [18]      # Hourly (2080 hours/year)
        })
        
        result = mconv(df, 'amount', 'period')
        
        # 25 * (2080/12) = 4333.33 per month
        assert np.isclose(result.iloc[0], 25 * 2080/12)
    
    def test_conversion_all_period_codes(self):
        """Test that all period codes work correctly."""
        df = pd.DataFrame({
            'amount': [1000] * 13,
            'period': [1, 2, 3, 4, 5, 6, 8, 11, 12, 14, 18, 23, 24]
        })
        
        result = mconv(df, 'amount', 'period')
        
        # All should be numeric (not NA)
        assert result.notna().all()
        # All should be non-negative
        assert (result >= 0).all()

class TestAconv:
    """Tests for aconv (adjustment conversion) function."""
    
    def test_adjustment_positive_values(self):
        """Test adjustment with positive values."""
        df = pd.DataFrame({
            'amount': [1000, 2000, 3000],
            'period': [2, 4, 6]
        })
        
        result = aconv(df, 'amount', 'period')
        
        assert np.isclose(result.iloc[0], 1000*52)
        assert np.isclose(result.iloc[1], 2000*12)
        assert np.isclose(result.iloc[2], 3000)
    
    def test_adjustment_negative_values(self):
        """Test adjustment with negative values."""
        df = pd.DataFrame({
            'amount': [-1000, -2000, -3000],
            'period': [2, 4, 6]
        })
        
        result = aconv(df, 'amount', 'period')
        
        # Should preserve sign
        assert np.isclose(result.iloc[0], -1000*52)
        assert np.isclose(result.iloc[1], -2000*12)
        assert np.isclose(result.iloc[2], -3000)
    
    def test_adjustment_with_na_values(self):
        """Test adjustment with NA/missing values."""
        df = pd.DataFrame({
            'amount': [1000, np.nan, 3000],
            'period': [2, 4, 6]
        })
        
        result = aconv(df, 'amount', 'period')
        
        # NA should remain NA
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])
    
    def test_adjustment_zero_factor(self):
        """Test adjustment with factor of 0.0."""
        df = pd.DataFrame({
            'amount': [1000, 2000, 3000],
            'period': [-1, 0, 8]
        })
        
        result = aconv(df, 'amount', 'period')
        
        assert result.iloc[0] == 0
        assert result.iloc[1] == 0
        assert result.iloc[2] == 0