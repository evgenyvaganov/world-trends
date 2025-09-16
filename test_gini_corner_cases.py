#!/usr/bin/env python3
"""
Comprehensive corner case testing for Gini coefficient algorithm
Tests edge cases, mathematical properties, and potential failure modes
"""

import unittest
import pandas as pd
import numpy as np
from extract_data import WIDDataExtractor

class TestGiniCornerCases(unittest.TestCase):
    """Test suite for Gini algorithm corner cases and edge conditions"""
    
    def setUp(self):
        self.extractor = WIDDataExtractor()
    
    def prepare_test_data(self, percentiles, values):
        """Helper to prepare test data in WID format"""
        return pd.DataFrame({
            'percentile': percentiles,
            'value': values,
            'year': [2020] * len(percentiles),
            'variable': ['aptincj992'] * len(percentiles)
        })
    
    def test_missing_p0p100_total(self):
        """Test when p0p100 (total income) is missing"""
        data = self.prepare_test_data(
            ['p0p10', 'p0p50', 'p0p90'],
            [10, 50, 90]
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertTrue(pd.isna(gini), "Should return NaN when p0p100 missing")
    
    def test_negative_total_income(self):
        """Test when total income is negative (economic crisis scenario)"""
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [10, 20, 30, -100]  # Negative total
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertTrue(pd.isna(gini), "Should return NaN for negative total income")
    
    def test_zero_total_income(self):
        """Test when total income is zero"""
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [0, 0, 0, 0]  # All zeros
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertTrue(pd.isna(gini), "Should return NaN for zero total income")
    
    def test_decreasing_cumulative_values(self):
        """Test when cumulative values decrease (data error)"""
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [50, 40, 60, 100]  # p0p50 < p0p25 (impossible)
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        # Algorithm should handle this - check if result is reasonable
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Gini should be in valid range despite bad data")
    
    def test_extremely_sparse_data(self):
        """Test with only 2 data points"""
        data = self.prepare_test_data(
            ['p0p50', 'p0p100'],
            [10, 100]
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertTrue(pd.isna(gini), "Should return NaN for insufficient data points")
    
    def test_duplicate_percentiles(self):
        """Test with duplicate percentile entries"""
        data = self.prepare_test_data(
            ['p0p50', 'p0p50', 'p0p100'],  # Duplicate p0p50
            [30, 35, 100]  # Different values for same percentile
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        # Should handle duplicates gracefully
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Should handle duplicate percentiles")
    
    def test_very_high_inequality(self):
        """Test extreme inequality (99.9% to top 0.1%)"""
        data = self.prepare_test_data(
            ['p0p10', 'p0p50', 'p0p90', 'p0p99.9', 'p0p100'],
            [0.01, 0.05, 0.1, 1, 100]  # 99% of income to top 0.1%
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertFalse(pd.isna(gini), "Should handle extreme inequality")
        self.assertGreater(gini, 0.9, "Should show very high inequality")
        self.assertLessEqual(gini, 1.0, "Gini should not exceed 1.0")
    
    def test_negative_cumulative_values(self):
        """Test with negative cumulative values (debt economy)"""
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [-10, -5, 20, 100]  # Bottom groups have negative wealth
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        # Should handle negative values
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Should handle negative cumulative values")
    
    def test_non_monotonic_data(self):
        """Test non-monotonic cumulative data"""
        data = self.prepare_test_data(
            ['p0p10', 'p0p30', 'p0p20', 'p0p100'],  # p0p20 after p0p30
            [5, 15, 10, 100]
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        # Algorithm sorts by percentile_value, should handle this
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Should sort and handle non-monotonic input")
    
    def test_malformed_percentile_strings(self):
        """Test with malformed percentile strings"""
        data = pd.DataFrame({
            'percentile': ['p0p50', 'bad_format', 'p0p100', ''],
            'value': [25, 50, 100, 75],
            'year': [2020] * 4,
            'variable': ['aptincj992'] * 4
        })
        gini = self.extractor.calculate_gini_coefficient(data)
        # Should filter out malformed percentiles
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Should filter malformed percentiles")
    
    def test_very_large_numbers(self):
        """Test with very large income values"""
        large_val = 1e15  # Quadrillion
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [large_val * 0.1, large_val * 0.3, large_val * 0.6, large_val]
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertFalse(pd.isna(gini), "Should handle very large numbers")
        self.assertTrue(0 <= gini <= 1, "Gini should be valid for large numbers")
    
    def test_very_small_numbers(self):
        """Test with very small income values"""
        small_val = 1e-10  # Very small
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [small_val * 0.1, small_val * 0.3, small_val * 0.6, small_val]
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertFalse(pd.isna(gini), "Should handle very small numbers")
        self.assertTrue(0 <= gini <= 1, "Gini should be valid for small numbers")
    
    def test_zero_filtering_edge_case(self):
        """Test the smart zero filtering logic"""
        # Case where zeros might be filtered incorrectly
        data = self.prepare_test_data(
            ['p0p5', 'p0p10', 'p0p15', 'p0p90', 'p0p95', 'p0p100'],
            [0, 0, 1, 50, 80, 100]  # Multiple zeros at bottom
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Should handle zero filtering correctly")
    
    def test_sudden_jump_detection(self):
        """Test sudden jump detection in zero filtering"""
        data = self.prepare_test_data(
            ['p0p10', 'p0p20', 'p0p30', 'p0p100'],
            [0, 0, 90, 100]  # Sudden jump from 0 to 90% of total income
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        # Should detect and possibly filter the sudden jump
        if not pd.isna(gini):
            self.assertTrue(0 <= gini <= 1, "Should handle sudden jumps")
    
    def test_numerical_precision(self):
        """Test numerical precision with close values"""
        data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [24.999999999, 49.999999999, 74.999999999, 100.0]
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        self.assertFalse(pd.isna(gini), "Should handle precision issues")
        # This is near-equality, so Gini should be very small but not exactly 0.25
        self.assertLess(gini, 0.001, "Near-equal distribution should give very small Gini")
        self.assertGreaterEqual(gini, 0.0, "Gini should be non-negative")
    
    def test_lorenz_curve_boundary_conditions(self):
        """Test Lorenz curve calculation at boundary conditions"""
        # Perfect equality should give area = 0.5, Gini = 0
        equal_data = self.prepare_test_data(
            ['p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p100'],
            [20, 40, 60, 80, 100]
        )
        gini_equal = self.extractor.calculate_gini_coefficient(equal_data)
        self.assertAlmostEqual(gini_equal, 0.0, places=3, msg="Perfect equality should give Gini ≈ 0")
        
        # Maximum inequality should give area ≈ 0, Gini ≈ 1
        max_ineq_data = self.prepare_test_data(
            ['p0p20', 'p0p40', 'p0p60', 'p0p80', 'p0p99', 'p0p100'],
            [0, 0, 0, 0, 0, 100]
        )
        gini_max = self.extractor.calculate_gini_coefficient(max_ineq_data)
        self.assertGreater(gini_max, 0.9, "Maximum inequality should give high Gini")
    
    def test_trapezoidal_integration_accuracy(self):
        """Test accuracy of trapezoidal rule integration"""
        # Create data where we know the analytical answer
        # Linear Lorenz curve: y = x^2 gives Gini = 1/3
        data = self.prepare_test_data(
            ['p0p10', 'p0p20', 'p0p30', 'p0p40', 'p0p50', 'p0p60', 'p0p70', 'p0p80', 'p0p90', 'p0p100'],
            [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]  # x^2 distribution
        )
        gini = self.extractor.calculate_gini_coefficient(data)
        expected_gini = 1/3  # Analytical result for y = x^2
        self.assertAlmostEqual(gini, expected_gini, places=2, 
                             msg=f"Should approximate analytical result: expected {expected_gini:.3f}, got {gini:.3f}")

class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties that Gini coefficient must satisfy"""
    
    def setUp(self):
        self.extractor = WIDDataExtractor()
    
    def prepare_test_data(self, percentiles, values):
        """Helper to prepare test data"""
        return pd.DataFrame({
            'percentile': percentiles,
            'value': values,
            'year': [2020] * len(percentiles),
            'variable': ['aptincj992'] * len(percentiles)
        })
    
    def test_scale_invariance(self):
        """Test that Gini is invariant to scaling all incomes by same factor"""
        base_data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [10, 30, 60, 100]
        )
        scaled_data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [100, 300, 600, 1000]  # 10x scaling
        )
        
        gini_base = self.extractor.calculate_gini_coefficient(base_data)
        gini_scaled = self.extractor.calculate_gini_coefficient(scaled_data)
        
        self.assertAlmostEqual(gini_base, gini_scaled, places=5, 
                             msg="Gini should be scale-invariant")
    
    def test_transfer_principle(self):
        """Test Pigou-Dalton transfer principle: transfer from rich to poor should decrease Gini"""
        before_data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [5, 25, 60, 100]
        )
        # Transfer 5 units from top group to bottom group
        after_data = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [10, 30, 60, 100]  # More equal distribution
        )
        
        gini_before = self.extractor.calculate_gini_coefficient(before_data)
        gini_after = self.extractor.calculate_gini_coefficient(after_data)
        
        self.assertLess(gini_after, gini_before, 
                       "Transfer from rich to poor should decrease Gini")
    
    def test_anonymity_principle(self):
        """Test that order of income receivers doesn't matter"""
        # This is implicitly satisfied by cumulative percentile approach
        # But test that reordering data doesn't affect result
        data1 = self.prepare_test_data(
            ['p0p25', 'p0p50', 'p0p75', 'p0p100'],
            [10, 30, 60, 100]
        )
        data2 = self.prepare_test_data(
            ['p0p50', 'p0p100', 'p0p25', 'p0p75'],  # Different order
            [30, 100, 10, 60]
        )
        
        gini1 = self.extractor.calculate_gini_coefficient(data1)
        gini2 = self.extractor.calculate_gini_coefficient(data2)
        
        self.assertAlmostEqual(gini1, gini2, places=5,
                             msg="Order of data should not affect Gini")

if __name__ == '__main__':
    unittest.main(verbosity=2)