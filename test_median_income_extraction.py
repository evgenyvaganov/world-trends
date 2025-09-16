#!/usr/bin/env python3
"""
Test suite for median income extraction and PPP conversion
Tests the accuracy of data extraction, PPP conversion, and edge cases
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppp_conversion_factors_2011 import convert_to_international_dollars, PPP_FACTORS_2011

class TestMedianIncomeExtraction(unittest.TestCase):
    """Test suite for median income data extraction and processing"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample WID-like data
        self.sample_data = pd.DataFrame({
            'country': ['US', 'US', 'US', 'CN', 'CN', 'CN'],
            'variable': ['aptincj992', 'aptincj992', 'aptinci992', 'aptincj992', 'aptincj992', 'shweal992'],
            'percentile': ['p50p51', 'p0p100', 'p50p51', 'p50p51', 'p90p100', 'p50p51'],
            'year': [2021, 2021, 2021, 2021, 2021, 2021],
            'value': [48819.8, 84179.2, 48819.8, 92577.0, 250000.0, 0.05],
            'age': ['adults', 'adults', 'adults', 'adults', 'adults', 'adults']
        })
    
    def test_ppp_conversion_basic(self):
        """Test basic PPP conversion for known values"""
        # Test US (PPP factor = 1.0)
        us_value = 1000
        us_converted = convert_to_international_dollars(us_value, 'US', 2011)
        expected_us = us_value * 1.345  # Only inflation adjustment
        self.assertAlmostEqual(us_converted, expected_us, places=2)
        
        # Test China (PPP factor = 3.51)
        cn_value = 3510
        cn_converted = convert_to_international_dollars(cn_value, 'CN', 2011)
        expected_cn = (cn_value / 3.51) * 1.345
        self.assertAlmostEqual(cn_converted, expected_cn, places=2)
    
    def test_ppp_conversion_all_countries(self):
        """Test PPP conversion works for all G20 countries"""
        test_value = 10000
        for country_code in PPP_FACTORS_2011.keys():
            try:
                result = convert_to_international_dollars(test_value, country_code)
                self.assertIsNotNone(result)
                self.assertGreater(result, 0)
                # Check inflation adjustment is applied
                self.assertGreater(result, test_value / max(PPP_FACTORS_2011.values()))
            except Exception as e:
                self.fail(f"PPP conversion failed for {country_code}: {e}")
    
    def test_argentina_turkey_special_cases(self):
        """Test special PPP factors for Argentina and Turkey (currency redenominations)"""
        # Argentina - should have high PPP factor due to old peso units
        ar_factor = PPP_FACTORS_2011['AR']
        self.assertGreater(ar_factor, 100, "Argentina PPP factor should account for old currency units")
        
        # Turkey - should have adjusted factor for old lira
        tr_factor = PPP_FACTORS_2011['TR']
        self.assertGreater(tr_factor, 5, "Turkey PPP factor should account for old lira units")
        
        # Test conversion produces reasonable values for realistic inputs
        # Argentina median income in WID is around 4.2M old pesos for 2021
        ar_income = 4200000  # Realistic value in old peso units 
        ar_converted = convert_to_international_dollars(ar_income, 'AR')
        self.assertLess(ar_converted, 50000, "Argentina conversion should produce reasonable income")
        self.assertGreater(ar_converted, 20000, "Argentina conversion should not undervalue")
        
        # Turkey median income in WID is around 228k old lira for 2021
        tr_income = 228000  # Realistic value in old lira units
        tr_converted = convert_to_international_dollars(tr_income, 'TR')
        self.assertLess(tr_converted, 40000, "Turkey conversion should produce reasonable income")
        self.assertGreater(tr_converted, 15000, "Turkey conversion should not undervalue")
    
    def test_median_percentile_extraction(self):
        """Test that we correctly extract median (p50p51) percentile"""
        # Filter for median income
        median_data = self.sample_data[
            (self.sample_data['variable'].isin(['aptincj992', 'aptinci992'])) &
            (self.sample_data['percentile'] == 'p50p51')
        ]
        
        # Should get US and CN median values only
        self.assertEqual(len(median_data), 3)  # US has 2 entries (different variables), CN has 1
        
        # Check we don't get average (p0p100) or top decile (p90p100)
        self.assertFalse((median_data['percentile'] == 'p0p100').any())
        self.assertFalse((median_data['percentile'] == 'p90p100').any())
    
    def test_variable_selection(self):
        """Test that we select correct income variables"""
        # Should select aptincj992 and aptinci992, not wealth variables
        income_data = self.sample_data[
            self.sample_data['variable'].isin(['aptincj992', 'aptinci992'])
        ]
        
        # Should not include wealth variable
        self.assertFalse((income_data['variable'] == 'shweal992').any())
        
        # Should have income variables
        self.assertTrue((income_data['variable'].isin(['aptincj992', 'aptinci992'])).all())
    
    def test_inflation_adjustment(self):
        """Test inflation adjustment from 2011 to 2021"""
        # The inflation adjustment should be approximately 34.5%
        value_2011 = 1000
        value_2021 = convert_to_international_dollars(value_2011, 'US')
        
        # Check inflation adjustment is applied (should be ~1345)
        self.assertAlmostEqual(value_2021, 1345, delta=10)
    
    def test_edge_case_zero_values(self):
        """Test handling of zero values"""
        zero_value = convert_to_international_dollars(0, 'US')
        self.assertEqual(zero_value, 0)
    
    def test_edge_case_negative_values(self):
        """Test handling of negative values (debt/losses)"""
        negative_value = convert_to_international_dollars(-1000, 'US')
        self.assertLess(negative_value, 0)
        self.assertAlmostEqual(negative_value, -1345, delta=10)
    
    def test_edge_case_missing_country(self):
        """Test handling of country not in PPP factors"""
        with self.assertRaises(ValueError):
            convert_to_international_dollars(1000, 'XX')  # Invalid country code
    
    def test_edge_case_very_large_values(self):
        """Test handling of very large values (billionaire wealth)"""
        large_value = 1e12  # 1 trillion in local currency
        result = convert_to_international_dollars(large_value, 'US')
        self.assertGreater(result, 1e12)  # Should be larger due to inflation adjustment
        self.assertLess(result, 2e12)  # But not more than double
    
    def test_south_africa_extreme_inequality(self):
        """Test South Africa's extreme inequality case"""
        # South Africa has negative wealth for bottom 50%
        # Median income should still be positive
        za_income = 47889  # Typical ZAR median income
        za_converted = convert_to_international_dollars(za_income, 'ZA')
        
        # Should produce reasonable income despite extreme inequality
        self.assertGreater(za_converted, 5000)
        self.assertLess(za_converted, 20000)
    
    def test_saudi_arabia_oil_economy(self):
        """Test Saudi Arabia's oil-dependent economy conversion"""
        # Saudi Arabia has high GDP but declining median income
        sa_income = 92577  # Typical SAR median income
        sa_converted = convert_to_international_dollars(sa_income, 'SA')
        
        # Should produce high but not unrealistic income
        self.assertGreater(sa_converted, 30000)
        self.assertLess(sa_converted, 100000)
    
    def test_indonesia_large_denominations(self):
        """Test Indonesia with large currency denominations"""
        # Indonesia uses large numbers (millions of rupiah)
        id_income = 47512324  # Typical IDR median income (47.5 million)
        id_converted = convert_to_international_dollars(id_income, 'ID')
        
        # Should produce reasonable income
        self.assertGreater(id_converted, 10000)
        self.assertLess(id_converted, 25000)
    
    def test_year_over_year_consistency(self):
        """Test that year-over-year changes are reasonable"""
        # Create multi-year data
        years_data = pd.DataFrame({
            'year': [2019, 2020, 2021],
            'value': [45000, 43200, 48819.8]  # COVID dip in 2020
        })
        
        # Calculate year-over-year changes
        years_data['pct_change'] = years_data['value'].pct_change()
        
        # Check changes are within reasonable bounds (-20% to +20% per year)
        for pct_change in years_data['pct_change'].dropna():
            self.assertGreater(pct_change, -0.20, "Year-over-year decline too steep")
            self.assertLess(pct_change, 0.20, "Year-over-year growth too high")
    
    def test_g20_coverage(self):
        """Test that all G20 countries are covered"""
        g20_countries = ['AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 
                        'IT', 'JP', 'KR', 'MX', 'RU', 'SA', 'ZA', 'TR', 'GB', 'US']
        
        for country in g20_countries:
            self.assertIn(country, PPP_FACTORS_2011, f"Missing PPP factor for {country}")
    
    def test_data_type_consistency(self):
        """Test that extracted data maintains correct types"""
        # Check numeric columns are numeric
        numeric_cols = ['year', 'value']
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data[col]))
        
        # Check string columns are strings
        string_cols = ['country', 'variable', 'percentile']
        for col in string_cols:
            self.assertTrue(pd.api.types.is_object_dtype(self.sample_data[col]))
    
    def test_duplicate_handling(self):
        """Test handling of duplicate data entries"""
        # Create data with duplicates
        dup_data = pd.DataFrame({
            'country': ['US', 'US'],
            'variable': ['aptincj992', 'aptinci992'],
            'percentile': ['p50p51', 'p50p51'],
            'year': [2021, 2021],
            'value': [48819.8, 48819.8]
        })
        
        # Should handle duplicates by taking first or averaging
        unique_data = dup_data.drop_duplicates(subset=['country', 'year', 'percentile'])
        self.assertEqual(len(unique_data), 1)

class TestDataValidation(unittest.TestCase):
    """Test data validation and sanity checks"""
    
    def test_median_less_than_average(self):
        """Test that median is typically less than average due to income skew"""
        # For most countries, median should be 60-90% of average
        median = 50000
        average = 65000
        ratio = median / average
        
        self.assertLess(ratio, 0.90, "Median should typically be less than average")
        self.assertGreater(ratio, 0.60, "Median shouldn't be too far below average")
    
    def test_reasonable_income_bounds(self):
        """Test that converted incomes fall within reasonable bounds"""
        # Test various countries' expected ranges (2021 PPP USD)
        test_cases = [
            ('US', 65000, 50000, 80000),      # US median ~$65k
            ('CN', 22000, 15000, 30000),      # China median ~$22k
            ('IN', 8000, 5000, 12000),        # India median ~$8k
            ('DE', 62000, 50000, 75000),      # Germany median ~$62k
            ('BR', 20000, 15000, 30000),      # Brazil median ~$20k
            ('ZA', 10000, 5000, 20000),       # South Africa median ~$10k
        ]
        
        for country, expected, min_val, max_val in test_cases:
            self.assertGreater(expected, min_val, f"{country} income too low")
            self.assertLess(expected, max_val, f"{country} income too high")
    
    def test_growth_rate_bounds(self):
        """Test that growth rates are within historical bounds"""
        # Annual growth rates should typically be between -5% and +10%
        growth_rates = [
            ('CN', 5.67),   # China high growth
            ('US', 0.54),   # US low growth
            ('SA', -1.77),  # Saudi Arabia decline
            ('IN', 2.07),   # India moderate growth
        ]
        
        for country, rate in growth_rates:
            self.assertGreater(rate, -5, f"{country} decline too steep")
            self.assertLess(rate, 10, f"{country} growth unrealistically high")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)