"""
Unit tests for SCF variance estimation module.

This module tests convenience wrapper functions for scf_estimate.

Author: Payton Crawford
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from scfpublic.scfcombo import (
    scfcombo_result
)
from scfpublic.scfcombo.stats import (
    scf_mean,
    scf_quantile,
    scf_regression,
    scf_total
)

class TestConvenienceFunctions:
    """Tests for convenience wrapper functions"""
    
    def test_scf_mean(self, prepared_data_full):
        """Test scf_mean() convenience function"""
        result = scf_mean(prepared_data_full, 'networth')
        assert isinstance(result, scfcombo_result)
        assert result.statistic_name == 'Mean'
        assert result.variable_name == 'networth'
    
    def test_scf_quantile_median(self, prepared_data_full):
        """Test scf_quantile() for median"""
        result = scf_quantile(prepared_data_full, 'networth', quantile=0.5)
        assert isinstance(result, scfcombo_result)
        assert result.statistic_name == 'Median'
        assert result.variable_name == 'networth'
    
    def test_scf_quantile_percentile(self, prepared_data_full):
        """Test scf_quantile() for other percentiles"""
        result = scf_quantile(prepared_data_full, 'income', quantile=0.9)
        assert isinstance(result, scfcombo_result)
        assert '90th Percentile' in result.statistic_name
        assert result.variable_name == 'income'
    
    @patch('statsmodels.formula.api.wls')
    def test_scf_regression_ols(self, mock_smf, prepared_data_full, mock_statsmodels_result):
        """Test scf_regression() with OLS"""
        mock_model = Mock()
        mock_model.fit.return_value = mock_statsmodels_result
        mock_smf.wls.return_value = mock_model
        
        result = scf_regression(
            prepared_data_full,
            'networth ~ age',
            model_type='ols'
        )
        assert isinstance(result, scfcombo_result)
    
    def test_scf_regression_invalid_model(self, prepared_data_full):
        """Test scf_regression() with invalid model type"""
        with pytest.raises(ValueError, match="Model type must be one of"):
            scf_regression(prepared_data_full, 'networth ~ age', model_type='invalid')

    def test_scf_total(self, prepared_data_full):
        """Test scf_total() convenience function"""
        result = scf_total(prepared_data_full, 'networth')
        assert isinstance(result, scfcombo_result)
        assert result.statistic_name == 'Total'
        assert result.variable_name == 'networth'
