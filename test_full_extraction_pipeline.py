#!/usr/bin/env python3
"""
Integration tests for the full median income extraction pipeline
Tests end-to-end extraction, processing, and output validation
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os

class TestFullExtractionPipeline(unittest.TestCase):
    """Integration tests for the complete data extraction pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Load the actual extracted data
        self.df = pd.read_csv('datasets/median_ppp_income_g20.csv')
        
        # Load metadata
        with open('datasets/extraction_metadata.json', 'r') as f:
            self.metadata = json.load(f)
    
    def test_data_completeness(self):
        """Test that all G20 countries have data for expected years"""
        expected_countries = self.metadata['country_codes'].keys()
        
        for country in expected_countries:
            country_data = self.df[self.df['country_code'] == country]
            self.assertGreater(len(country_data), 0, f"No data found for {country}")
            
            # Should have data from 1995-2024 range (at minimum 20+ years)
            years_available = len(country_data['year'].unique())
            self.assertGreater(years_available, 20, f"{country} has insufficient years of data")
    
    def test_data_structure_integrity(self):
        """Test that the output CSV has correct structure and types"""
        # Check required columns
        required_cols = ['country_code', 'country_name', 'year', 'median_ppp_income']
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")
        
        # Check data types
        self.assertTrue(pd.api.types.is_object_dtype(self.df['country_code']))
        self.assertTrue(pd.api.types.is_object_dtype(self.df['country_name']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df['year']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df['median_ppp_income']))
        
        # Check for missing values in critical columns
        self.assertFalse(self.df['country_code'].isnull().any())
        self.assertFalse(self.df['year'].isnull().any())
        self.assertFalse(self.df['median_ppp_income'].isnull().any())
    
    def test_country_code_consistency(self):
        """Test that country codes match metadata"""
        expected_codes = set(self.metadata['country_codes'].keys())
        actual_codes = set(self.df['country_code'].unique())
        
        # All countries in data should be in metadata
        self.assertTrue(actual_codes.issubset(expected_codes))
        
        # Check country names match
        for code in actual_codes:
            expected_name = self.metadata['country_codes'][code]
            actual_names = self.df[self.df['country_code'] == code]['country_name'].unique()
            self.assertEqual(len(actual_names), 1, f"Multiple country names for {code}")
            self.assertEqual(actual_names[0], expected_name)
    
    def test_year_range_validity(self):
        """Test that years are within expected range"""
        min_year = self.df['year'].min()
        max_year = self.df['year'].max()
        
        # Should span roughly 1995-2024
        self.assertGreaterEqual(min_year, 1990)
        self.assertLessEqual(max_year, 2025)
        self.assertGreater(max_year - min_year, 25, "Insufficient time span")
    
    def test_income_value_reasonability(self):
        """Test that income values are reasonable"""
        # Check for obviously wrong values
        self.assertFalse((self.df['median_ppp_income'] < 0).any(), "Negative incomes found")
        self.assertFalse((self.df['median_ppp_income'] > 1000000).any(), "Unrealistically high incomes")
        
        # Check minimum reasonable income (should be > $1000 PPP)
        self.assertTrue((self.df['median_ppp_income'] > 1000).all(), "Unrealistically low incomes")
    
    def test_known_relationships(self):
        """Test known economic relationships between countries"""
        latest_year = self.df['year'].max()
        latest_data = self.df[self.df['year'] == latest_year]
        
        # Create income lookup
        income_by_country = {}
        for _, row in latest_data.iterrows():
            income_by_country[row['country_code']] = row['median_ppp_income']
        
        # Test expected relationships for latest year
        if 'US' in income_by_country and 'IN' in income_by_country:
            self.assertGreater(income_by_country['US'], income_by_country['IN'],
                             "US should have higher median income than India")
        
        if 'DE' in income_by_country and 'BR' in income_by_country:
            self.assertGreater(income_by_country['DE'], income_by_country['BR'],
                             "Germany should have higher median income than Brazil")
        
        if 'AU' in income_by_country and 'MX' in income_by_country:
            self.assertGreater(income_by_country['AU'], income_by_country['MX'],
                             "Australia should have higher median income than Mexico")
    
    def test_growth_trends_consistency(self):
        """Test that growth trends are historically consistent"""
        # Test China's rapid growth
        cn_data = self.df[self.df['country_code'] == 'CN'].sort_values('year')
        if len(cn_data) >= 2:
            first_val = cn_data.iloc[0]['median_ppp_income']
            last_val = cn_data.iloc[-1]['median_ppp_income']
            cn_growth = (last_val / first_val) - 1
            self.assertGreater(cn_growth, 1.0, "China should show substantial growth")
        
        # Test Japan's stagnation
        jp_data = self.df[self.df['country_code'] == 'JP'].sort_values('year')
        if len(jp_data) >= 10:  # At least 10 years of data
            jp_data['pct_change'] = jp_data['median_ppp_income'].pct_change()
            avg_change = jp_data['pct_change'].mean()
            self.assertLess(avg_change, 0.02, "Japan should show low/stagnant growth")
    
    def test_metadata_accuracy(self):
        """Test that metadata matches actual data"""
        # Check country count
        actual_countries = len(self.df['country_code'].unique())
        expected_countries = self.metadata['extraction_info']['total_countries']
        self.assertEqual(actual_countries, expected_countries)
        
        # Check data period
        actual_min_year = self.df['year'].min()
        actual_max_year = self.df['year'].max()
        
        # Should roughly match target period (allowing some variance)
        target_period = self.metadata['extraction_info']['target_period']
        self.assertIn(str(actual_min_year)[:3], target_period)  # 199x in 1995-2024
        self.assertIn(str(actual_max_year)[:3], target_period)  # 202x in 1995-2024
    
    def test_ppp_conversion_applied(self):
        """Test that PPP conversion was applied correctly"""
        # Values should be in reasonable PPP USD range, not raw local currency
        # For example, China values should be ~20k USD, not ~200k yuan
        
        cn_2021 = self.df[(self.df['country_code'] == 'CN') & (self.df['year'] == 2021)]
        if not cn_2021.empty:
            cn_income = cn_2021['median_ppp_income'].iloc[0]
            # Should be ~20k USD, not ~200k yuan or ~3k USD
            self.assertGreater(cn_income, 15000)
            self.assertLess(cn_income, 30000)
        
        # Indonesia should be reasonable PPP, not millions of rupiah
        id_2021 = self.df[(self.df['country_code'] == 'ID') & (self.df['year'] == 2021)]
        if not id_2021.empty:
            id_income = id_2021['median_ppp_income'].iloc[0]
            # Should be ~15k USD, not millions of rupiah
            self.assertGreater(id_income, 10000)
            self.assertLess(id_income, 25000)
    
    def test_inflation_adjustment_applied(self):
        """Test that inflation adjustment to 2021 current prices was applied"""
        # US 1995 value should be lower than 2021 value
        us_data = self.df[self.df['country_code'] == 'US'].sort_values('year')
        
        if len(us_data) >= 2:
            us_1995 = us_data[us_data['year'] == 1995]['median_ppp_income']
            us_2021 = us_data[us_data['year'] == 2021]['median_ppp_income']
            
            if not us_1995.empty and not us_2021.empty:
                # Should show some real growth + inflation adjustment
                # Even with wage stagnation, should show some increase
                self.assertGreater(us_2021.iloc[0], us_1995.iloc[0])
    
    def test_no_duplicate_records(self):
        """Test that there are no duplicate country-year records"""
        # Each country-year combination should appear only once
        duplicates = self.df.groupby(['country_code', 'year']).size()
        max_duplicates = duplicates.max()
        self.assertEqual(max_duplicates, 1, "Found duplicate country-year records")
    
    def test_data_export_format(self):
        """Test that exported CSV can be properly read and parsed"""
        # Try reading the CSV with different pandas settings
        try:
            # Test with different encodings and separators
            df_test = pd.read_csv('datasets/median_ppp_income_g20.csv', encoding='utf-8')
            self.assertGreater(len(df_test), 0)
            
            # Test that numeric columns parse correctly
            self.assertTrue(pd.api.types.is_numeric_dtype(df_test['year']))
            self.assertTrue(pd.api.types.is_numeric_dtype(df_test['median_ppp_income']))
            
        except Exception as e:
            self.fail(f"Failed to read exported CSV: {e}")
    
    def test_extreme_values_flagged(self):
        """Test identification of extreme values that need investigation"""
        # Calculate z-scores for latest year to identify outliers
        latest_year = self.df['year'].max()
        latest_data = self.df[self.df['year'] == latest_year]
        
        if len(latest_data) > 3:  # Need sufficient data for z-score
            mean_income = latest_data['median_ppp_income'].mean()
            std_income = latest_data['median_ppp_income'].std()
            latest_data['z_score'] = (latest_data['median_ppp_income'] - mean_income) / std_income
            
            # Flag extreme values (|z| > 2.5)
            extreme_values = latest_data[abs(latest_data['z_score']) > 2.5]
            
            # Log extreme values for manual review
            if not extreme_values.empty:
                print("\nExtreme values flagged for review:")
                for _, row in extreme_values.iterrows():
                    print(f"{row['country_code']}: ${row['median_ppp_income']:,.0f} (z={row['z_score']:.2f})")

class TestCornerCases(unittest.TestCase):
    """Test corner cases and edge conditions"""
    
    def setUp(self):
        self.df = pd.read_csv('datasets/median_ppp_income_g20.csv')
    
    def test_countries_with_economic_crises(self):
        """Test countries that experienced major economic crises"""
        # Argentina should show volatility due to economic crises
        ar_data = self.df[self.df['country_code'] == 'AR'].sort_values('year')
        if len(ar_data) > 5:
            ar_data['pct_change'] = ar_data['median_ppp_income'].pct_change()
            # Should have some years with significant changes
            large_changes = abs(ar_data['pct_change']) > 0.10
            self.assertTrue(large_changes.any(), "Argentina should show crisis-related volatility")
    
    def test_oil_dependent_economies(self):
        """Test oil-dependent economies during price volatility periods"""
        oil_countries = ['SA', 'RU']
        
        for country in oil_countries:
            country_data = self.df[self.df['country_code'] == country]
            if not country_data.empty:
                # Should show some correlation with oil price cycles
                # (This is a basic test - could be expanded)
                income_range = country_data['median_ppp_income'].max() - country_data['median_ppp_income'].min()
                mean_income = country_data['median_ppp_income'].mean()
                volatility = income_range / mean_income
                
                # Oil economies should show higher volatility than average
                self.assertGreater(volatility, 0.1, f"{country} should show oil-related volatility")
    
    def test_post_communist_transitions(self):
        """Test countries with post-communist transitions"""
        transition_countries = ['RU', 'CN']  # Different transition paths
        
        for country in transition_countries:
            country_data = self.df[self.df['country_code'] == country].sort_values('year')
            if len(country_data) >= 2:
                first_income = country_data.iloc[0]['median_ppp_income']
                last_income = country_data.iloc[-1]['median_ppp_income']
                
                # Should show substantial growth over the period
                growth_factor = last_income / first_income
                self.assertGreater(growth_factor, 1.2, f"{country} should show transition-related growth")

if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)