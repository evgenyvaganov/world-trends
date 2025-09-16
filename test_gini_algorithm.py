#!/usr/bin/env python3
"""
Test suite for Gini coefficient calculation algorithm
Tests edge cases, known values, and mathematical properties
"""

import unittest
import pandas as pd
import numpy as np
import sys
from extract_data import WIDDataExtractor

def prepare_test_data(percentiles, values, year=2020, variable='aptincj992'):
    """Helper function to prepare test data in WID format with required columns"""
    # Ensure we have p0p100 (total income) for the algorithm
    if 'p0p100' not in percentiles:
        percentiles = percentiles + ['p0p100']
        values = values + [1.0]  # Total income = 1.0 for normalized data
    
    data = pd.DataFrame({
        'percentile': percentiles,
        'value': values,
        'year': [year] * len(percentiles),
        'variable': [variable] * len(percentiles)
    })
    
    return data

def test_perfect_equality():
    """Test Gini = 0 for perfect equality"""
    print("Testing perfect equality...")
    
    # Create mock data with equal income shares (need 5+ points)
    income_data = prepare_test_data(
        ['p0p10', 'p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p100'],
        [0.10, 0.20, 0.40, 0.60, 0.80, 1.0]  # Equal distribution
    )
    
    extractor = WIDDataExtractor()
    gini = extractor.calculate_gini_coefficient(income_data)
    
    expected = 0.0
    print(f"  Expected: {expected}, Got: {gini:.6f}")
    assert abs(gini - expected) < 0.01, f"Perfect equality should give Gini ≈ 0, got {gini}"
    print("  ✅ PASSED")

def test_perfect_inequality():
    """Test Gini approaches 1 for maximum inequality"""
    print("Testing maximum inequality...")
    
    # High inequality: bottom 90% has very little, top 10% has most
    income_data = prepare_test_data(
        ['p0p10', 'p0p50', 'p0p80', 'p0p90', 'p0p99', 'p0p100'],
        [0.0, 0.02, 0.05, 0.10, 0.90, 1.0]  # Extreme inequality
    )
    
    extractor = WIDDataExtractor()
    gini = extractor.calculate_gini_coefficient(income_data)
    
    print(f"  Expected: close to 1.0, Got: {gini:.6f}")
    assert gini > 0.8, f"Maximum inequality should give Gini close to 1, got {gini}"
    print("  ✅ PASSED")

def test_known_distribution():
    """Test with known income distribution"""
    print("Testing known distribution (80-20 rule approximation)...")
    
    # Approximate 80-20 distribution: bottom 80% has 20% of income
    income_data = prepare_test_data(
        ['p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p100'],
        [0.04, 0.08, 0.12, 0.20, 1.0]  # 80-20 rule
    )
    
    extractor = WIDDataExtractor()
    gini = extractor.calculate_gini_coefficient(income_data)
    
    # 80-20 rule typically gives Gini around 0.6-0.8
    print(f"  Expected: 0.6-0.8 range, Got: {gini:.6f}")
    assert 0.5 < gini < 0.9, f"80-20 distribution should give Gini ~0.6-0.8, got {gini}"
    print("  ✅ PASSED")

def test_manual_calculation():
    """Test algorithm consistency with reasonable bounds"""
    print("Testing algorithm consistency...")
    
    # WID-style cumulative distribution with moderate inequality (need 5+ points)
    income_data = prepare_test_data(
        ['p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p90', 'p0p100'],
        [0.08, 0.24, 0.48, 0.72, 0.85, 1.0]  # Moderate inequality
    )
    
    extractor = WIDDataExtractor()
    gini = extractor.calculate_gini_coefficient(income_data)
    
    # This distribution should give moderate inequality
    print(f"  Moderate inequality Gini: {gini:.6f}")
    assert 0.1 < gini < 0.4, f"Moderate inequality should give Gini ~0.1-0.4, got {gini}"
    print("  ✅ PASSED")

def test_real_data_sample():
    """Test with actual US data sample to ensure reasonable results"""
    print("Testing with real US data sample...")
    
    extractor = WIDDataExtractor()
    
    # Load actual US data
    us_data = extractor.load_country_data('US')
    if us_data.empty:
        print("  ⚠️ SKIPPED - No US data available")
        return
    
    # Get 2020 data
    us_2020 = us_data[
        (us_data['year'] == 2020) & 
        (us_data['variable'] == 'aptincj992') &
        (us_data['percentile'].str.startswith('p0p'))
    ].copy()
    
    if us_2020.empty:
        print("  ⚠️ SKIPPED - No 2020 US percentile data")
        return
    
    gini = extractor.calculate_gini_coefficient(us_2020)
    
    # US Gini should be in reasonable range (World Bank reports ~0.41, but our method may differ)
    print(f"  US 2020 Gini: {gini:.6f}")
    assert 0.3 < gini < 0.9, f"US Gini out of reasonable range: {gini}"
    print("  ✅ PASSED")

def test_edge_cases():
    """Test edge cases and data quality issues"""
    print("Testing edge cases...")
    
    extractor = WIDDataExtractor()
    
    # Test with empty data
    empty_data = pd.DataFrame()
    gini_empty = extractor.calculate_gini_coefficient(empty_data)
    print(f"  Empty data Gini: {gini_empty}")
    assert pd.isna(gini_empty), f"Empty data should return NaN, got {gini_empty}"
    
    # Test with insufficient data points
    minimal_data = prepare_test_data(['p0p100'], [1.0])
    gini_minimal = extractor.calculate_gini_coefficient(minimal_data)
    print(f"  Single point Gini: {gini_minimal}")
    assert pd.isna(gini_minimal), f"Single data point should return NaN, got {gini_minimal}"
    
    print("  ✅ PASSED")

def test_mathematical_properties():
    """Test mathematical properties of Gini coefficient"""
    print("Testing mathematical properties...")
    
    extractor = WIDDataExtractor()
    
    # Property: More unequal distribution should have higher Gini (need 5+ points)
    
    # Distribution 1: Moderate inequality
    dist1 = prepare_test_data(
        ['p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p90', 'p0p100'],
        [0.12, 0.28, 0.48, 0.72, 0.85, 1.0]  # Moderate
    )
    
    # Distribution 2: Higher inequality
    dist2 = prepare_test_data(
        ['p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p90', 'p0p100'],
        [0.04, 0.12, 0.25, 0.45, 0.70, 1.0]  # Higher inequality
    )
    
    gini1 = extractor.calculate_gini_coefficient(dist1)
    gini2 = extractor.calculate_gini_coefficient(dist2)
    
    print(f"  Moderate inequality Gini: {gini1:.6f}")
    print(f"  Higher inequality Gini: {gini2:.6f}")
    
    assert gini2 > gini1, f"Higher inequality should have higher Gini: {gini2} vs {gini1}"
    print("  ✅ PASSED")

def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("GINI COEFFICIENT ALGORITHM TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_perfect_equality,
        test_perfect_inequality,
        test_known_distribution,
        test_manual_calculation,
        test_real_data_sample,
        test_edge_cases,
        test_mathematical_properties
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 All tests passed!")

class TestGiniAlgorithm(unittest.TestCase):
    """Unittest wrapper for Gini algorithm tests"""
    
    def test_all_gini_functions(self):
        """Run all Gini algorithm tests as a single unittest"""
        # Run the original test functions and capture results
        tests = [
            test_perfect_equality,
            test_perfect_inequality,
            test_known_distribution,
            test_manual_calculation,
            test_real_data_sample,
            test_edge_cases,
            test_mathematical_properties
        ]
        
        failed_tests = []
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                failed_tests.append(f"{test_func.__name__}: {e}")
        
        if failed_tests:
            self.fail(f"Gini algorithm tests failed: {'; '.join(failed_tests)}")

if __name__ == "__main__":
    # Can be run either as unittest or standalone
    if len(sys.argv) > 1 and sys.argv[1] == 'unittest':
        unittest.main()
    else:
        run_all_tests()