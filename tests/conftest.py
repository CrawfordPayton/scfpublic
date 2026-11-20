"""
Shared pytest fixtures for SCF test suite.

This module provides common fixtures for testing all packages and subpackages
Author: Payton Crawford
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Callable

# ============================================================================
# SAMPLE DATA FIXTURES - scfpublic
# ============================================================================

@pytest.fixture
def sample_main_data():
    """
    Fixture providing sample main data (X-variables) for testing scfpublic.
    
    Simulates SCF main dataset structure with:
    - Household identifiers (YY1, Y1)
    - Main weight (WGT)
    - X-variables (survey responses)
    - J-codes (imputation flags)
    """
    return pd.DataFrame({
        'YY1': [1, 1, 1],
        'Y1': [11, 12, 13],
        'WGT': [1000, 2000, 3000],
        'X1': [100, 200, 300],
        'X2': [10, 20, 30],
        'X7607': [1000, 2000, 3000],
        'J1000': [5, 10, 15]
    })

@pytest.fixture
def sample_extr_data():
    """
    Fixture providing sample extract/bulletin data for testing scfpublic.
    
    Contains computed/derived variables like:
    - NETWORTH, INCOME, AGE (common analysis variables)
    """
    return pd.DataFrame({
        'YY1': [1, 1, 1],
        'Y1': [11, 12, 13],
        'WGT': [1000, 2000, 3000],
        'NETWORTH': [100000, 200000, 300000],
        'INCOME': [50000, 60000, 70000],
        'AGE': [35, 45, 55]
    })

@pytest.fixture
def sample_rep_wgts():
    """
    Fixture providing sample replicate weights for testing scfpublic.
    
    Contains:
    - Household identifiers
    - Multiple replicate weights (wgt1, wgt2, ...)
    """
    np.random.seed(42)
    data = {
        'YY1': [1, 1, 1],
        'Y1': [11, 12, 13]
    }
    # Add 10 replicate weights for testing
    for i in range(1, 11):
        data[f'wgt{i}'] = np.random.uniform(1000, 5000, 3)
    return pd.DataFrame(data)

# ============================================================================
# SAMPLE DATA FIXTURES - SCF Combo (Multiple Implicates)
# ============================================================================

@pytest.fixture
def sample_main_data_for_combo():
    """
    Fixture for testing prepare_scf_data with 5 implicates combined.
    
    Creates data with implicate_id (y1) that ends in 1-5 for splitting.
    """
    np.random.seed(42)
    n_households = 100
    data = []
    
    for imp in range(1, 6):
        for hh in range(1, n_households + 1):
            data.append({
                'yy1': hh,
                'y1': hh * 10 + imp,  
                'wgt': np.random.uniform(1000, 5000),
                'year': 2025,
                'networth': np.random.normal(100000, 50000),
                'income': np.random.normal(50000, 20000),
                'age': np.random.randint(25, 75)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_replicate_weights_for_combo():
    """
    Fixture for testing prepare_scf_data with replicate weights.
    
    Creates replicate weights in format expected by prepare_scf_data.
    Can have either:
    - wt1b1-wt1b999 + mm1-mm999 (base weights + multiplicity)
    - wgt1-wgt999 (already combined)
    """
    np.random.seed(42)
    n_households = 100
    n_replicates = 999
    
    data = {
        'yy1': range(1, n_households + 1),
        'year': [2025] * n_households
    }
    
    # Add combined replicate weights
    for r in range(1, n_replicates + 1):
        data[f'wgt{r}'] = np.random.uniform(1000, 5000, n_households)
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_replicate_weights_base_format():
    """
    Fixture providing replicate weights in base format (wt1b + mm).
    
    This tests the case where weights need to be computed:
    wgt{i} = wt1b{i} * mm{i}
    """
    np.random.seed(42)
    n_households = 100
    n_replicates = 999
    
    data = {
        'yy1': range(1, n_households + 1),
        'year': [2025] * n_households
    }
    
    # Add base weights
    for r in range(1, n_replicates + 1):
        data[f'wt1b{r}'] = np.random.uniform(0.5, 2.0, n_households)
    
    # Add multiplicity factors
    for r in range(1, n_replicates + 1):
        data[f'mm{r}'] = np.random.uniform(1000, 5000, n_households)
    
    return pd.DataFrame(data)

@pytest.fixture
def prepared_data_full():
    """
    Fixture providing fully prepared SCF data with realistic structure.
    
    This is a complete, ready-to-use dataset for testing scf_estimate
    and convenience functions. Uses reduced replicate count for speed.
    """
    np.random.seed(42)
    n_households = 100
    n_replicates = 999
    
    # Create implicates
    implicates = []
    for imp in range(1, 6):
        data = {
            'yy1': range(1, n_households + 1),
            'y1': (np.asarray(range(1, n_households + 1)) * 10) + imp,
            'wgt': np.random.uniform(1000, 5000, n_households),
            'year': [2025] * n_households,
            'networth': np.random.normal(100000, 50000, n_households),
            'age': np.random.randint(25, 75, n_households),
            'income': np.random.normal(50000, 20000, n_households),
            'education': np.random.choice([1, 2, 3], n_households)
        }
        implicates.append(pd.DataFrame(data))
    
    # Create replicate weights
    rw_data = {
        'yy1': range(1, n_households + 1),
        'year': [2025] * n_households
    }
    for r in range(1, n_replicates + 1):
        rw_data[f'wgt{r}'] = np.random.uniform(1000, 5000, n_households)
    replicate_weights = pd.DataFrame(rw_data)
    
    # Create metadata
    metadata = {
        'n_households': n_households,
        'n_implicates': 5,
        'n_replicates': n_replicates,
        'weight_col': 'wgt',
        'household_id_col': 'yy1',
        'implicate_id_col': 'y1',
        'year_col': 'year',
        'merge_keys': ['year', 'yy1']
    }
    
    return {
        'implicates': implicates,
        'replicate_weights': replicate_weights,
        'metadata': metadata
    }

@pytest.fixture
def mock_statsmodels_result():
    """
    Fixture providing a mock statsmodels regression result.
    
    Used for testing regression functionality without actually
    running regressions.
    """
    mock_result = Mock()
    mock_result.params = pd.Series(
        [1.0, 2.0, 3.0],
        index=['Intercept', 'age', 'income']
    )
    
    return mock_result

# ============================================================================
# SAMPLE DATA FIXTURES - For Testing cpi.py
# ============================================================================
@pytest.fixture
def sample_years_df():
    """Create sample DataFrame with valid years."""
    return pd.DataFrame({
        'year': [1989, 1992, 1995],
        'value': [100, 200, 300]
    })

@pytest.fixture
def mock_cpi_data():
    """Create mock CPI data."""
    return pd.DataFrame({
        'YEAR': list(range(1988, 2026)),
        'SEP': np.linspace(50, 150, 38),
        'AVG': np.linspace(48, 148, 38)
    })

# ============================================================================
# SAMPLE DATA FIXTURES - For Testing util.py
# ============================================================================
@pytest.fixture
def sample_conversion_data():
    """Create sample data for testing conversion functions."""
    return pd.DataFrame({
        'income': [52000, 26000, 12000, 4000, 100000],
        'period': [2, 3, 4, 5, 6],  # weekly, bi-weekly, monthly, quarterly, annual
        'weight': [1000, 2000, 3000, 4000, 5000],
        'adj_factor': [1.0, 1.1, 1.2, 1.3, 1.4]
    })

# ============================================================================
# MOCK FACTORIES - HTTP/Network Mocking
# ============================================================================

@pytest.fixture
def mock_http_response():
    """
    Fixture providing a mock HTTP response for testing downloads.
    
    Returns Mock object simulating requests.Response with:
    - content: Fake binary data
    - raise_for_status: No-op method
    """
    mock_response = Mock()
    mock_response.content = b'fake zip content'
    mock_response.raise_for_status = Mock()
    return mock_response

@pytest.fixture
def mock_zip_file():
    """
    Fixture providing a mock ZIP file context manager.
    
    Returns MagicMock simulating zipfile.ZipFile with:
    - Context manager support (__enter__/__exit__)
    - read() method returning fake .dta content
    """
    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.__exit__.return_value = False
    mock_zip.read.return_value = b'fake dta content'
    return mock_zip

# ============================================================================
# MOCK FACTORIES - Data Reading
# ============================================================================

@pytest.fixture
def mock_read_stata_handler():
    """
    Fixture providing a factory for creating pd.read_stata mocks.
    
    Returns function that takes a DataFrame and creates a proper mock
    handling the chunking pattern used in download_year():
    
    Usage:
        mock_handler = mock_read_stata_handler(sample_df)
        mock_read_stata.side_effect = mock_handler
    
    The handler properly simulates:
    1. First call with chunksize=1 (returns iterator for column names)
    2. Second call with columns= (returns filtered DataFrame)
    """
    def _create_handler(mock_df: pd.DataFrame) -> Callable:
        """
        Create a mock handler for pd.read_stata.
        
        Args:
            mock_df: DataFrame to return from mock
            
        Returns:
            Callable that handles chunksize and columns parameters
        """
        def _mock(buffer, chunksize=None, columns=None):
            if chunksize:
                # First call: return iterator with sample
                return iter([mock_df])
            else:
                # Second call: return filtered data
                if columns:
                    # Case-insensitive column filtering
                    cols_upper = [c.upper() for c in mock_df.columns]
                    available = [
                        mock_df.columns[i] 
                        for i, cu in enumerate(cols_upper) 
                        if cu in [c.upper() for c in columns]
                    ]
                    return mock_df[available].copy()
                return mock_df.copy()
        return _mock
    return _create_handler

@pytest.fixture
def mock_download_year_factory(sample_main_data, sample_extr_data):
    """
    Fixture providing a factory for creating mock download_year functions.
    
    Returns function that creates customizable mocks for download_year()
    that can return different data based on year and file_type.
    
    Usage:
        # Simple case - use defaults
        mock_func = mock_download_year_factory()
        mock_download.side_effect = mock_func
        
        # Custom case - provide specific mappings
        year_map = {(2025, 'main_data'): custom_df}
        mock_func = mock_download_year_factory(year_map)
        mock_download.side_effect = mock_func
    """
    def _create_mock(year_data_map: Dict = None) -> Callable:
        """
        Create a mock download_year function.
        
        Args:
            year_data_map: Optional dict mapping (year, file_type) -> DataFrame
                          If None, uses default sample data
                          
        Returns:
            Callable that simulates download_year()
        """
        def _mock_download(year: int, file_type: str = 'main_data', **kwargs) -> pd.DataFrame:
            # Check custom mapping first
            if year_data_map and (year, file_type) in year_data_map:
                df = year_data_map[(year, file_type)].copy()
            else:
                # Default behavior based on file_type
                if file_type == 'main_data':
                    df = sample_main_data.copy()
                elif file_type == 'extr_data':
                    df = sample_extr_data.copy()
                else:
                    # Minimal DataFrame for rep_wgts
                    df = pd.DataFrame({
                        'yy1': [1, 2, 3],
                        'y1': [11, 12, 13]
                    })
            
            # Add year column if not present
            if 'year' not in df.columns:
                df['year'] = year
            
            return df
        
        return _mock_download
    return _create_mock

# ============================================================================
# ESTIMATOR FIXTURES - For Testing scf_estimate()
# ============================================================================

@pytest.fixture
def weighted_mean_estimator():
    """
    Fixture providing a weighted mean estimator function.
    
    Returns function suitable for use with scf_estimate().
    """
    def estimator(data: pd.DataFrame, weights: np.ndarray, variable: str) -> float:
        notmissing = data[variable].notna()
        if not notmissing.any():
            return np.nan
        return np.average(data.loc[notmissing, variable].values, weights=weights[notmissing])
    return estimator

@pytest.fixture
def weighted_quantile_estimator():
    """
    Fixture providing a weighted quantile estimator function.
    
    Returns function suitable for use with scf_estimate().
    Requires statsmodels.
    """
    def estimator(data: pd.DataFrame, weights: np.ndarray, variable: str, q: float) -> float:
        try:
            from statsmodels.stats.weightstats import DescrStatsW
        except ImportError:
            pytest.skip("statsmodels not available")
        
        notmissing = data[variable].notna()
        if not notmissing.any():
            return np.nan
        
        weighted_stats = DescrStatsW(
            data.loc[notmissing, variable].values, 
            weights=weights[notmissing]
        )
        return float(weighted_stats.quantile(probs=[q]).iloc[0])
    return estimator

@pytest.fixture
def ols_coefficients_estimator():
    """
    Fixture providing an OLS regression estimator function.
    
    Returns function that computes regression coefficients,
    suitable for use with scf_estimate().
    Requires statsmodels.
    """
    def estimator(data: pd.DataFrame, weights: np.ndarray, formula: str) -> np.ndarray:
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            pytest.skip("statsmodels not available")
        
        model = smf.wls(formula, data=data, weights=weights)
        result = model.fit(disp=False)
        return result.params.values
    return estimator

# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Fixture providing a temporary directory for test data.
    
    Returns Path object to temporary directory that's automatically
    cleaned up after test completion.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def sample_vars():
    """
    Fixture providing sample variable list for testing match_vars().
    """
    return ['X1', 'X2', 'X7607', 'NETWORTH', 'AGE', 'INCOME', 'Y1', 'J1000']

@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Fixture that resets random seed before each test for reproducibility.
    
    autouse=True means this runs automatically for every test.
    """
    np.random.seed(42)
    yield

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """
    Configure custom pytest markers.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, mocked)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (real external calls)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (may take several seconds)"
    )

def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location/name.
    """
    for item in items:
        # Mark all tests as unit by default
        if "integration" not in item.keywords and "e2e" not in item.keywords:
            item.add_marker(pytest.mark.unit)
