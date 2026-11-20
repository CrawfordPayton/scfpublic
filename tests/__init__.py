"""
Test suite initialization and runner for SCF data processing modules.

This module provides a centralized test runner for all test files with
organized execution and reporting.

Author: Payton Crawford
"""

import pytest
import sys
from pathlib import Path

def run_all_tests(verbose: bool = True, coverage: bool = False) -> int:
    """
    Run all test files in the test suite.
    
    Args:
        verbose (bool): Enable verbose output (default: True)
        coverage (bool): Enable coverage reporting (default: False)
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Get the directory containing this file
    test_dir = Path(__file__).parent
    
    # Build pytest arguments
    args = [str(test_dir)]
    
    if verbose:
        args.append('-v')
    
    
    if coverage:
        args.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])
    
    # Add flags for output
    args.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Strict marker validation
        '-ra',  # Show summary of all test outcomes
    ])
    
    print("=" * 70)
    print("Running SCF Data Processing Test Suite")
    print("=" * 70)
    
    # Run tests
    exit_code = pytest.main(args)
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Review output above.")
    print("=" * 70)
    
    return exit_code

def run_test_file(test_file: str, verbose: bool = True) -> int:
    """
    Run a specific test file.
    
    Args:
        test_file (str): Name of test file (e.g., 'test_scfcombo.py')
        verbose (bool): Enable verbose output (default: True)
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    test_dir = Path(__file__).parent
    test_path = test_dir / test_file
    
    if not test_path.exists():
        print(f"Error: Test file '{test_file}' not found in {test_dir}")
        return 1
    
    args = [str(test_path)]
    
    if verbose:
        args.append('-v')
    
    args.extend(['--tb=short', '-ra'])
    
    print(f"Running tests from: {test_file}")
    print("=" * 70)
    
    return pytest.main(args)

def run_tests_by_module(module_name: str, verbose: bool = True) -> int:
    """
    Run tests for a specific module.
    
    Args:
        module_name (str): Module name ('scfcombo', 'scfpublic', 'cpi')
        verbose (bool): Enable verbose output (default: True)
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    module_to_test_file = {
        'scfpublic': 'test_scfpublic.py',
        'scfpublic.cpi': 'test_cpi.py',
        'scfpublic.scfcombo': 'test_scfcombo.py',
        'scfpublic.util': 'test_util.py',
        'scfpublic.scfcombo.stats': 'test_scfcombo_stats.py',
    }
    
    if module_name not in module_to_test_file:
        print(f"Error: Unknown module '{module_name}'")
        print(f"Available modules: {', '.join(module_to_test_file.keys())}")
        return 1
    
    test_file = module_to_test_file[module_name]
    return run_test_file(test_file, verbose=verbose)

if __name__ == '__main__':
    # Allow running specific tests from command line
    if len(sys.argv) > 1:
        module = sys.argv[1]
        exit_code = run_tests_by_module(module)
    else:
        # Run all tests
        exit_code = run_all_tests(verbose=True, coverage=False)
    
    sys.exit(exit_code)
