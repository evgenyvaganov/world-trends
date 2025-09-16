import unittest
import pandas as pd
import numpy as np
import os


class TestMMRAlgorithm(unittest.TestCase):
    def setUp(self):
        self.data_file = "/workspaces/ai/world-trends/datasets/income_mean_median_mmr_g20.csv"
    
    def test_mmr_calculation_mathematical_accuracy(self):
        """Test that MMR values match manual calculation of mean/median ratio"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
        
        df = pd.read_csv(self.data_file)
        
        # Calculate MMR manually and compare with stored values
        calculated_mmr = df['mean_ppp_income'] / df['median_ppp_income']
        
        np.testing.assert_array_almost_equal(
            df['mmr'].values, 
            calculated_mmr.values, 
            decimal=10,
            err_msg="MMR values don't match manual calculation"
        )
    
    def test_mmr_perfect_equality_theoretical(self):
        """Test theoretical MMR calculation for perfect equality"""
        mean_income = 40000
        median_income = 40000
        expected_mmr = mean_income / median_income
        
        self.assertAlmostEqual(expected_mmr, 1.0, places=6,
                              msg="Perfect equality should yield MMR = 1.0")
    
    def test_mmr_high_inequality_theoretical(self):
        """Test theoretical MMR calculation for high inequality"""
        mean_income = 100000  # High mean due to wealthy outliers
        median_income = 25000  # Lower median (typical middle-class income)
        expected_mmr = mean_income / median_income
        
        self.assertAlmostEqual(expected_mmr, 4.0, places=4,
                              msg="High inequality case should yield MMR = 4.0")
    
    def test_mmr_data_structure(self):
        """Test that MMR data has correct structure and columns"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
            
        df = pd.read_csv(self.data_file)
        
        # Check required columns exist
        required_cols = ['country_code', 'country_name', 'year', 'mean_ppp_income', 'median_ppp_income', 'mmr']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['mmr']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['mean_ppp_income']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['median_ppp_income']))
    
    def test_mmr_no_division_by_zero(self):
        """Test that MMR handles edge cases without division by zero"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
            
        df = pd.read_csv(self.data_file)
        
        # Check no infinite or NaN values from division by zero
        self.assertFalse(np.isinf(df['mmr']).any(), "Found infinite MMR values")
        self.assertFalse(df['mmr'].isna().any(), "Found NaN MMR values")
    
    def test_mmr_positive_values(self):
        """Test that MMR values are positive (both mean and median should be positive)"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
            
        df = pd.read_csv(self.data_file)
        
        self.assertTrue((df['mmr'] > 0).all(), "Found non-positive MMR values")
        self.assertTrue((df['mean_ppp_income'] > 0).all(), "Found non-positive mean income")
        self.assertTrue((df['median_ppp_income'] > 0).all(), "Found non-positive median income")
    
    def test_mmr_g20_coverage(self):
        """Test that MMR data covers all G20 countries"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
            
        df = pd.read_csv(self.data_file)
        
        g20_countries = {
            'AR', 'AU', 'BR', 'CA', 'CN', 'DE', 'FR', 'GB', 
            'ID', 'IN', 'IT', 'JP', 'KR', 'MX', 'RU', 'SA', 'TR', 'US', 'ZA'
        }
        
        countries_in_data = set(df['country_code'].unique())
        
        # Check that we have data for most G20 countries (allowing for some missing data)
        coverage = len(countries_in_data.intersection(g20_countries)) / len(g20_countries)
        self.assertGreater(coverage, 0.8, f"Low G20 coverage: {coverage:.2%}")
    
    def test_mmr_realistic_ranges(self):
        """Test that MMR values fall within realistic economic ranges"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
            
        df = pd.read_csv(self.data_file)
        
        # Check for and report unusual MMR < 1 cases (median > mean)
        low_mmr = df[df['mmr'] < 1.0]
        if not low_mmr.empty:
            print(f"Found {len(low_mmr)} cases where median > mean (MMR < 1.0):")
            print(low_mmr[['country_name', 'year', 'mmr']].to_string())
            
        # Most MMR values should be >= 0.8 (allowing for some unusual cases)
        self.assertTrue((df['mmr'] >= 0.8).all(), 
                       f"Found extremely low MMR < 0.8, min value: {df['mmr'].min()}")
        
        # MMR should be reasonable (< 10 for most countries)
        extreme_mmr = df[df['mmr'] > 10]
        if not extreme_mmr.empty:
            print(f"Warning: Found {len(extreme_mmr)} extreme MMR values > 10:")
            print(extreme_mmr[['country_name', 'year', 'mmr']].head())
        
        # Most values should be reasonable (between 0.9 and 5.0)
        reasonable_mmr = df[(df['mmr'] >= 0.9) & (df['mmr'] <= 5.0)]
        reasonable_ratio = len(reasonable_mmr) / len(df)
        self.assertGreater(reasonable_ratio, 0.9, 
                          f"Only {reasonable_ratio:.1%} of MMR values are reasonable (0.9-5 range)")
    
    def test_mmr_sample_validation(self):
        """Test MMR calculation on specific sample data points"""
        if not os.path.exists(self.data_file):
            self.skipTest("MMR data file not found")
            
        df = pd.read_csv(self.data_file)
        
        # Test a few sample points manually
        sample = df.head(5)
        for _, row in sample.iterrows():
            expected_mmr = row['mean_ppp_income'] / row['median_ppp_income']
            self.assertAlmostEqual(row['mmr'], expected_mmr, places=10,
                                 msg=f"MMR mismatch for {row['country_name']} {row['year']}")


if __name__ == '__main__':
    unittest.main(verbosity=1)