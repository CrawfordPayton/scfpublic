"""
Unit tests for SCF variance estimation module.

This module tests variance estimation with multiple imputation and
replicate weights for various statistics and regression models.

Author: Payton Crawford
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pydantic import ValidationError

from scfpublic.scfcombo import (
    scfcombo_result,
    prepare_scf_data,
    scf_estimate,
)

class TestScfComboResultDataclass:
    """Tests for scfcombo dataclass"""
    
    def test_scalar_result_initialization(self):
        """Test creating scfcombo with scalar results"""
        result = scfcombo_result(
            estimate=100.0,
            imputation_var=10.0,
            sampling_var=20.0,
            total_var=30.0,
            standard_error=5.48,
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
        )
        assert result.estimate == 100.0
        assert result.statistic_name == 'Statistic'
        assert result.coefficient_names is None
    
    def test_vector_result_initialization(self):
        """Test creating scfcombo_result with vector results"""
        result = scfcombo_result(
            estimate=np.array([1.0, 2.0, 3.0]),
            imputation_var=np.array([0.1, 0.2, 0.3]),
            sampling_var=np.array([0.2, 0.3, 0.4]),
            total_var=np.array([0.3, 0.5, 0.7]),
            standard_error=np.array([0.55, 0.71, 0.84]),
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            coefficient_names=['edcl', 'age', 'income']
        )
        assert len(result.estimate) == 3
        assert result.statistic_name == 'Multiple Statistics'
        assert len(result.coefficient_names) == 3

    def test_regression_result_initialization(self):
        """Test creating scfcombo_result with regression results"""
        result = scfcombo_result(
            estimate=np.array([1.0, 2.0, 3.0]),
            imputation_var=np.array([0.1, 0.2, 0.3]),
            sampling_var=np.array([0.2, 0.3, 0.4]),
            total_var=np.array([0.3, 0.5, 0.7]),
            standard_error=np.array([0.55, 0.71, 0.84]),
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            coefficient_names=['Intercept', 'age', 'income'],
            is_regression = True
        )
        assert len(result.estimate) == 3
        assert result.statistic_name == 'Regression'
        assert len(result.coefficient_names) == 3
    
    def test_summary_scalar(self):
        """Test summary() method for scalar results"""
        result = scfcombo_result(
            estimate=100.0,
            imputation_var=10.0,
            sampling_var=20.0,
            total_var=30.0,
            standard_error=5.48,
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
        )
        summary_df = result.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert 'Coef.' in summary_df.columns
        assert 'Std. Err.' in summary_df.columns
        assert len(summary_df) == 1
    
    def test_summary_vector(self):
        """Test summary() method for vector results"""
        result = scfcombo_result(
            estimate=np.array([1.0, 2.0]),
            imputation_var=np.array([0.1, 0.2]),
            sampling_var=np.array([0.2, 0.3]),
            total_var=np.array([0.3, 0.5]),
            standard_error=np.array([0.55, 0.71]),
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            coefficient_names=['Intercept', 'age']
        )
        summary_df = result.summary()
        assert len(summary_df) == 2
        assert 'Coef.' in summary_df.columns
    
    def test_display_scalar(self, capsys):
        """Test display() method for scalar results"""
        result = scfcombo_result(
            estimate=100.0,
            imputation_var=10.0,
            sampling_var=20.0,
            total_var=30.0,
            standard_error=5.48,
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            statistic_name='Mean',
            variable_name='networth'
        )
        result.display()
        captured = capsys.readouterr()
        assert 'Mean: networth' in captured.out
        assert '100.0000' in captured.out
    
    def test_display_regression(self, capsys):
        """Test display() method for regression results"""
        result = scfcombo_result(
            estimate=np.array([1.0, 2.0]),
            imputation_var=np.array([0.1, 0.2]),
            sampling_var=np.array([0.2, 0.3]),
            total_var=np.array([0.3, 0.5]),
            standard_error=np.array([0.55, 0.71]),
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            coefficient_names=['Intercept', 'age'],
            is_regression=True
        )
        result.display()
        captured = capsys.readouterr()
        assert 'SCF Regression Results' in captured.out
        assert 'Intercept' in captured.out

    def test_display_vector(self, capsys):
        """Test display() method for vector results"""
        result = scfcombo_result(
            estimate=np.array([1.0, 2.0]),
            imputation_var=np.array([0.1, 0.2]),
            sampling_var=np.array([0.2, 0.3]),
            total_var=np.array([0.3, 0.5]),
            standard_error=np.array([0.55, 0.71]),
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            coefficient_names=['edcl', 'age']
        )
        result.display()
        captured = capsys.readouterr()
        assert 'Multiple Statistics' in captured.out
        assert 'edcl' in captured.out
    
    def test_repr_scalar(self):
        """Test __repr__ for scalar results"""
        result = scfcombo_result(
            estimate=100.0,
            imputation_var=10.0,
            sampling_var=20.0,
            total_var=30.0,
            standard_error=5.48,
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
        )
        repr_str = repr(result)
        assert 'Statistic' in repr_str
        assert 'estimate=100.0000' in repr_str
    
    def test_repr_vector(self):
        """Test __repr__ for vector results"""
        result = scfcombo_result(
            estimate=np.array([1.0, 2.0]),
            imputation_var=np.array([0.1, 0.2]),
            sampling_var=np.array([0.2, 0.3]),
            total_var=np.array([0.3, 0.5]),
            standard_error=np.array([0.55, 0.71]),
            n_implicates=5,
            n_replicates=999,
            n_obs=1000,
            coefficient_names=['Intercept', 'age']
        )
        repr_str = repr(result)
        assert 'Regression' in repr_str
        assert '2 coefficients' in repr_str

class TestPrepareScfData:
    """Tests for prepare_scf_data() function"""
    
    def test_basic_preparation(
        self,
        sample_main_data_for_combo,
        sample_replicate_weights_for_combo
    ):
        """Test basic data preparation"""
        result = prepare_scf_data(
            sample_main_data_for_combo,
            sample_replicate_weights_for_combo,
            n_implicates=5,
            n_replicates=999
        )
        assert 'implicates' in result
        assert 'replicate_weights' in result
        assert 'metadata' in result
        assert len(result['implicates']) == 5
    
    def test_implicate_splitting(
        self,
        sample_main_data_for_combo,
        sample_replicate_weights_for_combo
    ):
        """Test that implicates are properly split"""
        result = prepare_scf_data(
            sample_main_data_for_combo,
            sample_replicate_weights_for_combo,
            n_implicates=5,
            n_replicates=999
        )
        implicates = result['implicates']
        assert all(len(imp) == 100 for imp in implicates)
        assert len(set(implicates[0]['y1'])) == 100
    
    def test_base_weight_computation(
        self,
        sample_main_data_for_combo,
        sample_replicate_weights_base_format
    ):
        """Test computation of wgt from wt1b * mm"""
        result = prepare_scf_data(
            sample_main_data_for_combo,
            sample_replicate_weights_base_format,
            n_implicates=5,
            n_replicates=999,
            verbose=True
        )
        
        rw_df = result['replicate_weights']
        # Check that wgt columns exist
        assert 'wgt1' in rw_df.columns
        assert 'wgt999' in rw_df.columns
        # Original wt1b and mm columns should not be in final result
        assert 'wt1b1' not in rw_df.columns
    
    def test_invalid_n_implicates(
        self,
        sample_main_data_for_combo,
        sample_replicate_weights_for_combo
    ):
        """Test error when n_implicates < 2"""
        with pytest.raises(ValueError, match="At least 2 implicates required"):
            prepare_scf_data(
                sample_main_data_for_combo,
                sample_replicate_weights_for_combo,
                n_implicates=1
            )
    
    def test_invalid_n_replicates(
        self,
        sample_main_data_for_combo,
        sample_replicate_weights_for_combo
    ):
        """Test error when n_replicates < 2"""
        with pytest.raises(ValueError, match="At least 2 replicate weights required"):
            prepare_scf_data(
                sample_main_data_for_combo,
                sample_replicate_weights_for_combo,
                n_replicates=1
            )
    
    def test_missing_columns_main_data(
        self,
        sample_replicate_weights_for_combo
    ):
        """Test error when main data missing required columns"""
        bad_data = pd.DataFrame({'wrong_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns in main data"):
            prepare_scf_data(bad_data, sample_replicate_weights_for_combo)
    
    def test_missing_columns_replicate_weights(
        self,
        sample_main_data_for_combo
    ):
        """Test error when replicate weights missing required columns"""
        bad_weights = pd.DataFrame({'wrong_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns in replicate weights"):
            prepare_scf_data(sample_main_data_for_combo, bad_weights)
    
    def test_metadata_content(
        self,
        sample_main_data_for_combo,
        sample_replicate_weights_for_combo
    ):
        """Test metadata contains correct information"""
        result = prepare_scf_data(
            sample_main_data_for_combo,
            sample_replicate_weights_for_combo,
            n_implicates=5,
            n_replicates=999
        )
        metadata = result['metadata']
        assert metadata['n_implicates'] == 5
        assert metadata['n_replicates'] == 999
        assert metadata['weight_col'] == 'wgt'
        assert 'merge_keys' in metadata

class TestScfEstimate:
    """Tests for scf_estimate() function"""
    
    def test_weighted_mean_estimator(
        self,
        prepared_data_full,
        weighted_mean_estimator
    ):
        """Test scf_estimate with weighted mean"""
        result = scf_estimate(
            prepared_data_full,
            weighted_mean_estimator,
            variable='networth'
        )
        assert isinstance(result, scfcombo_result)
        assert isinstance(result.estimate, (int, float))
        assert result.standard_error > 0
    
    def test_with_subset(
        self,
        prepared_data_full,
        weighted_mean_estimator
    ):
        """Test scf_estimate with subset filter"""
        result = scf_estimate(
            prepared_data_full,
            weighted_mean_estimator,
            subset=lambda df: df['age'] > 50,
            variable='networth'
        )
        assert isinstance(result, scfcombo_result)
        assert result.n_obs < 100  # Should be fewer than full sample
    
    def test_invalid_n_implicates(self):
        """Test error when fewer than 2 implicates"""
        bad_data = {
            'implicates': [pd.DataFrame()],
            'replicate_weights': pd.DataFrame(),
            'metadata': {'n_implicates': 1, 'n_replicates': 999}
        }
        with pytest.raises(KeyError):
            scf_estimate(bad_data, lambda d, w: 0)
    
    def test_invalid_n_replicates(self):
        """Test error when fewer than 2 replicates"""
        bad_data = {
            'implicates': [pd.DataFrame(), pd.DataFrame()],
            'replicate_weights': pd.DataFrame(),
            'metadata': {'n_implicates': 2, 'n_replicates': 1}
        }
        with pytest.raises(KeyError):
            scf_estimate(bad_data, lambda d, w: 0)
    
    def test_vector_results(self, prepared_data_full):
        """Test scf_estimate with vector results (e.g., regression)"""
        def simple_estimator(data, weights):
            return np.array([1.0, 2.0, 3.0])
        
        result = scf_estimate(
            prepared_data_full,
            simple_estimator,
            coefficient_names=['a', 'b', 'c']
        )
        assert len(result.estimate) == 3
        assert len(result.standard_error) == 3
        assert len(result.coefficient_names) == 3
