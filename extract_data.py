#!/usr/bin/env python3
"""
Data Extraction: WID to Clean CSV Conversion

Extracts the 3 key metrics from original WID datasets and creates
clean, analysis-ready CSV files in ./datasets/:
1. ppp_income_g20.csv - Real PPP income per person (2021 ICP)
2. wealth_distribution_g20.csv - Net worth by percentile brackets
3. inequality_gini_g20.csv - Calculated Gini coefficients

Usage: python extract_data.py
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List
from ppp_conversion_factors_2011 import convert_to_international_dollars, get_ppp_factor
from datetime import datetime

class WIDDataExtractor:
    """Extracts and cleans WID data into analysis-ready CSV files"""
    
    def __init__(self, source_path: str = "datasets/wid_all_data", output_path: str = "datasets"):
        self.source_path = source_path
        self.output_path = output_path
        self.g20_countries = {
            'AR': 'Argentina', 'AU': 'Australia', 'BR': 'Brazil', 'CA': 'Canada',
            'CN': 'China', 'FR': 'France', 'DE': 'Germany', 'IN': 'India',
            'ID': 'Indonesia', 'IT': 'Italy', 'JP': 'Japan', 'KR': 'South Korea',
            'MX': 'Mexico', 'RU': 'Russia', 'SA': 'Saudi Arabia', 'ZA': 'South Africa',
            'TR': 'Turkey', 'GB': 'United Kingdom', 'US': 'United States'
        }
        self.target_years = list(range(1995, 2025))
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
    
    def load_country_data(self, country_code: str) -> pd.DataFrame:
        """Load and preprocess country data"""
        filepath = os.path.join(self.source_path, f"WID_data_{country_code}.csv")
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found for {country_code}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath, delimiter=';')
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Filter for target years and remove invalid data
            df = df[
                df['year'].isin(self.target_years) & 
                df['value'].notna() & 
                (df['value'] != 0)
            ]
            
            return df
        
        except Exception as e:
            print(f"Error loading {country_code}: {e}")
            return pd.DataFrame()
    
    def extract_mean_median_income(self) -> pd.DataFrame:
        """Extract both MEAN and MEDIAN PPP income data to calculate MMR"""
        print("Extracting MEAN and MEDIAN PPP income data...")
        
        all_income_data = []
        
        for country_code, country_name in self.g20_countries.items():
            print(f"  Processing {country_name} ({country_code})...")
            
            df = self.load_country_data(country_code)
            if df.empty:
                continue
            
            # Extract BOTH median and mean pre-tax PPP income
            # aptincj992/aptinci992 with p50p51 = MEDIAN (50th percentile)
            # aptincj992/aptinci992 with p0p100 = MEAN (average of entire population)
            
            income_data = df[
                (df['variable'].isin(['aptincj992', 'aptinci992'])) &
                (df['percentile'].isin(['p50p51', 'p0p100']))  # MEDIAN and MEAN
            ].copy()
            
            # Enforce consistent PPP baseline (992 = 1992 PPP baseline)
            if not income_data.empty:
                available_vars = income_data['variable'].unique()
                if 'aptincj992' in available_vars:
                    # Prefer equal-split adults with 1992 PPP baseline
                    income_data = income_data[income_data['variable'] == 'aptincj992'].copy()
                elif 'aptinci992' in available_vars:
                    # Fall back to individual adults with 1992 PPP baseline
                    income_data = income_data[income_data['variable'] == 'aptinci992'].copy()
                else:
                    # No 1992 PPP baseline available - skip this country
                    income_data = pd.DataFrame()
            
            if not income_data.empty:
                # Add country information
                income_data['country_code'] = country_code
                income_data['country_name'] = country_name
                
                # Convert to international dollars using PPP factors
                ppp_factor = get_ppp_factor(country_code)
                if ppp_factor is not None:
                    income_data['value_intl_dollars'] = income_data['value'].apply(
                        lambda x: convert_to_international_dollars(x, country_code)
                    )
                    
                    all_income_data.append(income_data[['country_code', 'country_name', 'year', 'percentile', 'value_intl_dollars']].copy())
                    
                    # Log conversion for first year as verification
                    first_value = income_data['value'].iloc[0]
                    first_converted = income_data['value_intl_dollars'].iloc[0]
                    first_year = income_data['year'].iloc[0]
                    print(f"    💱 {first_year}: {first_value:,.0f} LCU → ${first_converted:,.0f} (PPP factor: {ppp_factor})")
        
        if all_income_data:
            combined_df = pd.concat(all_income_data, ignore_index=True)
            
            # Pivot to have median and mean as separate columns
            pivot_df = combined_df.pivot_table(
                index=['country_code', 'country_name', 'year'],
                columns='percentile',
                values='value_intl_dollars',
                aggfunc='first'
            ).reset_index()
            
            # Rename columns
            pivot_df.columns.name = None
            if 'p0p100' in pivot_df.columns:
                pivot_df = pivot_df.rename(columns={'p0p100': 'mean_ppp_income'})
            if 'p50p51' in pivot_df.columns:
                pivot_df = pivot_df.rename(columns={'p50p51': 'median_ppp_income'})
            
            # Calculate Mean to Median Ratio (MMR)
            if 'mean_ppp_income' in pivot_df.columns and 'median_ppp_income' in pivot_df.columns:
                pivot_df['mmr'] = pivot_df['mean_ppp_income'] / pivot_df['median_ppp_income']
            
            # Sort and save
            pivot_df = pivot_df.sort_values(['country_code', 'year'])
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'income_mean_median_mmr_g20.csv')
            pivot_df.to_csv(output_file, index=False)
            
            print(f"  ✅ Mean/Median/MMR income data saved: {output_file}")
            print(f"     Records: {len(pivot_df)}, Countries: {pivot_df['country_code'].nunique()}")
            
            return pivot_df
        
        print("  ❌ No income data extracted")
        return pd.DataFrame()

    def extract_ppp_income(self) -> pd.DataFrame:
        """Extract MEDIAN PPP income data (aptinc992 - pre-tax PPP income)"""
        print("Extracting MEDIAN PPP income data...")
        
        all_income_data = []
        
        for country_code, country_name in self.g20_countries.items():
            print(f"  Processing {country_name} ({country_code})...")
            
            df = self.load_country_data(country_code)
            if df.empty:
                continue
            
            # Extract MEDIAN pre-tax PPP income using CONSISTENT PPP BASELINE across countries
            # Priority order: aptincj992 (equal-split adults, 1992 PPP) > aptinci992 (individual adults, 1992 PPP)
            # Use p50p51 which represents the median (50th percentile) income
            
            income_data = df[
                (df['variable'].isin(['aptincj992', 'aptinci992'])) &
                (df['percentile'] == 'p50p51')  # MEDIAN instead of average
            ].copy()
            
            # Enforce consistent PPP baseline (992 = 1992 PPP baseline)
            if not income_data.empty:
                available_vars = income_data['variable'].unique()
                if 'aptincj992' in available_vars:
                    # Prefer equal-split adults with 1992 PPP baseline
                    income_data = income_data[income_data['variable'] == 'aptincj992'].copy()
                elif 'aptinci992' in available_vars:
                    # Fall back to individual adults with 1992 PPP baseline
                    income_data = income_data[income_data['variable'] == 'aptinci992'].copy()
                else:
                    # No 1992 PPP baseline available - skip this country
                    income_data = pd.DataFrame()
            
            if not income_data.empty:
                # Add country information
                income_data['country_code'] = country_code
                income_data['country_name'] = country_name
                
                # CURRENCY STANDARDIZATION: Convert to international dollars using PPP factors
                ppp_factor = get_ppp_factor(country_code)
                if ppp_factor is not None:
                    # Convert from local currency to international dollars (2011 PPP baseline)
                    income_data['value_intl_dollars'] = income_data['value'].apply(
                        lambda x: convert_to_international_dollars(x, country_code)
                    )
                    
                    # Select and rename columns
                    income_clean = income_data[['country_code', 'country_name', 'year', 'value_intl_dollars']].copy()
                    income_clean = income_clean.rename(columns={'value_intl_dollars': 'median_ppp_income'})
                    
                    all_income_data.append(income_clean)
                    
                    # Log conversion for first year as verification
                    first_value = income_data['value'].iloc[0]
                    first_converted = income_data['value_intl_dollars'].iloc[0]
                    first_year = income_data['year'].iloc[0]
                    print(f"    💱 {first_year}: {first_value:,.0f} LCU → ${first_converted:,.0f} (PPP factor: {ppp_factor})")
                else:
                    print(f"    ⚠️ No PPP factor available for {country_name} - skipping currency conversion")
        
        if all_income_data:
            combined_df = pd.concat(all_income_data, ignore_index=True)
            combined_df = combined_df.sort_values(['country_code', 'year'])
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'median_ppp_income_g20.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"  ✅ Median PPP Income data saved: {output_file}")
            print(f"     Records: {len(combined_df)}, Countries: {combined_df['country_code'].nunique()}")
            
            return combined_df
        
        print("  ❌ No PPP income data extracted")
        return pd.DataFrame()
    
    def extract_wealth_distribution(self) -> pd.DataFrame:
        """Extract wealth distribution data by percentiles"""
        print("Extracting wealth distribution data...")
        
        all_wealth_data = []
        wealth_percentiles = ['p0p10', 'p0p50', 'p50p100', 'p90p100', 'p99p100']
        
        for country_code, country_name in self.g20_countries.items():
            print(f"  Processing {country_name} ({country_code})...")
            
            df = self.load_country_data(country_code)
            if df.empty:
                continue
            
            # Extract wealth share data
            wealth_data = df[
                (df['variable'].str.contains('shweal.*992', regex=True, na=False)) &
                (df['percentile'].isin(wealth_percentiles))
            ].copy()
            
            if not wealth_data.empty:
                # Add country information
                wealth_data['country_code'] = country_code
                wealth_data['country_name'] = country_name
                
                # Create readable percentile labels
                percentile_labels = {
                    'p0p10': 'bottom_10pct',
                    'p0p50': 'bottom_50pct',
                    'p50p100': 'top_50pct',
                    'p90p100': 'top_10pct',
                    'p99p100': 'top_1pct'
                }
                
                wealth_data['percentile_group'] = wealth_data['percentile'].map(percentile_labels)
                
                # Select and rename columns
                wealth_clean = wealth_data[
                    ['country_code', 'country_name', 'year', 'percentile_group', 'value']
                ].copy()
                wealth_clean = wealth_clean.rename(columns={'value': 'wealth_share'})
                
                all_wealth_data.append(wealth_clean)
        
        if all_wealth_data:
            combined_df = pd.concat(all_wealth_data, ignore_index=True)
            combined_df = combined_df.sort_values(['country_code', 'year', 'percentile_group'])
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'wealth_distribution_g20.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"  ✅ Wealth distribution data saved: {output_file}")
            print(f"     Records: {len(combined_df)}, Countries: {combined_df['country_code'].nunique()}")
            
            return combined_df
        
        print("  ❌ No wealth distribution data extracted")
        return pd.DataFrame()
    
    def calculate_gini_coefficient(self, income_percentiles: pd.DataFrame) -> float:
        """
        Calculate Gini coefficient using DIRECT INCOME SHARES approach
        
        NEW APPROACH:
        - Convert WID cumulative percentile data to individual income shares
        - Use standard Gini formula: sum of absolute differences
        - More mathematically robust than Lorenz curve integration
        """
        if income_percentiles.empty:
            return np.nan
        
        # Get cumulative percentile data (p0pX format)
        cumulative_data = income_percentiles[
            income_percentiles['percentile'].str.startswith('p0p')
        ].copy()
        
        if cumulative_data.empty:
            return np.nan
        
        # Extract percentile values and sort
        def extract_percentile(percentile_str):
            """Extract numeric percentile from string like 'p0p50' -> 50"""
            try:
                return float(percentile_str.split('p0p')[1])
            except:
                return np.nan
        
        cumulative_data['percentile_value'] = cumulative_data['percentile'].apply(extract_percentile)
        cumulative_data = cumulative_data[cumulative_data['percentile_value'].notna()]
        cumulative_data = cumulative_data.sort_values('percentile_value')
        
        # Must have total population income (p0p100)
        total_income_row = cumulative_data[cumulative_data['percentile_value'] == 100.0]
        if total_income_row.empty:
            return np.nan
        
        total_income = total_income_row['value'].iloc[0]
        if total_income <= 0:
            return np.nan
        
        # Smart zero filtering (same as before)
        sorted_data = cumulative_data.sort_values('percentile_value')
        consecutive_zeros = 0
        for _, row in sorted_data.iterrows():
            if row['value'] <= 0:
                consecutive_zeros += 1
            else:
                break
        
        if consecutive_zeros > 0:
            first_nonzero_percentile = sorted_data.iloc[consecutive_zeros]['percentile_value']
            has_low_percentile_zeros = first_nonzero_percentile <= 20
        else:
            has_low_percentile_zeros = False
        
        has_sudden_jump = False
        if consecutive_zeros > 0 and consecutive_zeros < len(sorted_data) - 1:
            first_nonzero_value = sorted_data.iloc[consecutive_zeros]['value']
            has_sudden_jump = (first_nonzero_value / total_income) > 0.5
        
        zero_threshold = len(sorted_data) * 0.1
        is_likely_artifact = (
            consecutive_zeros > zero_threshold and 
            has_low_percentile_zeros and 
            has_sudden_jump
        )
        
        if is_likely_artifact:
            cumulative_data = cumulative_data[cumulative_data['value'] > 0]
        
        if len(cumulative_data) < 3:  # Need minimum 3 points for trapezoidal integration
            return np.nan
        # Note: 3-4 points give lower precision, 5+ points give high precision
        
        # CORRECTED APPROACH: Calculate Gini using proper Lorenz curve method
        # WID p0pX = cumulative income of bottom X% of population
        # Use trapezoidal rule to calculate area under Lorenz curve
        
        # Normalize to income shares [0,1] and population shares [0,1]
        cumulative_data['income_share'] = cumulative_data['value'] / total_income
        cumulative_data['pop_share'] = cumulative_data['percentile_value'] / 100.0
        
        # Add point (0,0) if not present for complete Lorenz curve
        if not ((cumulative_data['pop_share'] == 0) & (cumulative_data['income_share'] == 0)).any():
            zero_point = pd.DataFrame({
                'percentile_value': [0.0],
                'value': [0.0], 
                'income_share': [0.0],
                'pop_share': [0.0]
            })
            cumulative_data = pd.concat([zero_point, cumulative_data], ignore_index=True)
            cumulative_data = cumulative_data.sort_values('pop_share')
        
        # Calculate area under Lorenz curve using trapezoidal rule
        lorenz_area = 0.0
        prev_x, prev_y = 0.0, 0.0
        
        for _, row in cumulative_data.iterrows():
            x = row['pop_share']    # Population share (0 to 1)
            y = row['income_share'] # Cumulative income share (0 to 1)
            
            if x > prev_x:  # Avoid duplicate points
                # Trapezoidal area: width * average_height
                area_segment = (x - prev_x) * (y + prev_y) / 2
                lorenz_area += area_segment
                prev_x, prev_y = x, y
        
        # Gini coefficient = 1 - 2 * (area under Lorenz curve)
        # Perfect equality: area = 0.5, Gini = 0
        # Perfect inequality: area = 0, Gini = 1
        gini = 1 - 2 * lorenz_area
        
        # Ensure Gini is in valid range [0, 1]
        gini = max(0, min(1, gini))
        
        return gini
    
    def extract_inequality_data(self) -> pd.DataFrame:
        """Extract and calculate Gini coefficients from income percentile data"""
        print("Extracting inequality data (Gini coefficients)...")
        
        all_gini_data = []
        
        for country_code, country_name in self.g20_countries.items():
            print(f"  Processing {country_name} ({country_code})...")
            
            df = self.load_country_data(country_code)
            if df.empty:
                continue
            
            # Extract income percentile data for Gini calculation
            # Use equal-split adults (aptincj992) for consistent methodology across years
            # Fallback to individual adults (aptinci992) if equal-split not available
            income_percentiles = df[
                (df['variable'].isin(['aptincj992', 'aptinci992'])) &
                (df['percentile'].str.startswith('p0p'))
            ].copy()
            
            # Prefer equal-split adults if both are available
            if not income_percentiles.empty:
                available_vars = income_percentiles['variable'].unique()
                if 'aptincj992' in available_vars and 'aptinci992' in available_vars:
                    # Use only equal-split adults for consistency
                    income_percentiles = income_percentiles[
                        income_percentiles['variable'] == 'aptincj992'
                    ].copy()
                elif 'aptincj992' in available_vars:
                    # Use equal-split adults
                    income_percentiles = income_percentiles[
                        income_percentiles['variable'] == 'aptincj992'
                    ].copy()
                else:
                    # Use individual adults as fallback
                    income_percentiles = income_percentiles[
                        income_percentiles['variable'] == 'aptinci992'
                    ].copy()
            
            if not income_percentiles.empty:
                # Calculate Gini for each available year
                for year in sorted(income_percentiles['year'].unique()):
                    year_data = income_percentiles[income_percentiles['year'] == year]
                    
                    gini_coeff = self.calculate_gini_coefficient(year_data)
                    
                    if not np.isnan(gini_coeff):
                        all_gini_data.append({
                            'country_code': country_code,
                            'country_name': country_name,
                            'year': year,
                            'gini_coefficient': gini_coeff,
                            'data_source': 'calculated_from_percentiles'
                        })
        
        if all_gini_data:
            combined_df = pd.DataFrame(all_gini_data)
            combined_df = combined_df.sort_values(['country_code', 'year'])
            
            # Save to CSV
            output_file = os.path.join(self.output_path, 'inequality_gini_g20.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"  ✅ Gini coefficient data saved: {output_file}")
            print(f"     Records: {len(combined_df)}, Countries: {combined_df['country_code'].nunique()}")
            
            return combined_df
        
        print("  ❌ No Gini coefficient data calculated")
        return pd.DataFrame()
    
    def create_metadata_file(self):
        """Create metadata file describing the extracted datasets"""
        metadata = {
            "extraction_info": {
                "source": "World Inequality Database (WID)",
                "extraction_date": pd.Timestamp.now().isoformat(),
                "target_period": "1995-2024",
                "countries": "G20 nations",
                "total_countries": len(self.g20_countries)
            },
            "datasets": {
                "ppp_income_g20.csv": {
                    "description": "Real PPP income per person (2021 ICP baseline)",
                    "source_variable": "aptinc992 (pre-tax PPP income, adults)",
                    "percentile": "p0p100 (total population)",
                    "unit": "2021 PPP USD",
                    "columns": ["country_code", "country_name", "year", "ppp_income_2021"]
                },
                "wealth_distribution_g20.csv": {
                    "description": "Net worth distribution by percentile brackets",
                    "source_variable": "shweal992 (wealth shares, adults)",
                    "percentiles": ["bottom_10pct", "bottom_50pct", "top_50pct", "top_10pct", "top_1pct"],
                    "unit": "Share of total wealth (0-1)",
                    "columns": ["country_code", "country_name", "year", "percentile_group", "wealth_share"]
                },
                "inequality_gini_g20.csv": {
                    "description": "Gini coefficients calculated from income percentile data",
                    "calculation_method": "Approximated from available income percentiles",
                    "unit": "Gini coefficient (0-1, higher = more unequal)",
                    "columns": ["country_code", "country_name", "year", "gini_coefficient", "data_source"],
                    "note": "Simplified calculation - may not match official Gini statistics"
                }
            },
            "country_codes": self.g20_countries
        }
        
        # Save metadata as JSON
        import json
        metadata_file = os.path.join(self.output_path, 'extraction_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Metadata saved: {metadata_file}")
    
    def extract_all_metrics(self) -> Dict[str, pd.DataFrame]:
        """Extract all four metrics and save to clean CSV files"""
        print("=== WID DATA EXTRACTION ===")
        print(f"Source: World Inequality Database (wid.world, accessed {datetime.now().strftime('%Y-%m-%d')})")
        print("Extracting 4 key metrics from WID datasets...\n")
        
        results = {}
        
        # Extract Mean/Median/MMR Income data
        results['mean_median_mmr'] = self.extract_mean_median_income()
        print()
        
        # Extract PPP Income data (median only, for backward compatibility)
        results['ppp_income'] = self.extract_ppp_income()
        print()
        
        # Extract Wealth Distribution data
        results['wealth_distribution'] = self.extract_wealth_distribution()
        print()
        
        # Extract Inequality data (Gini coefficients)
        results['inequality'] = self.extract_inequality_data()
        print()
        
        # Create metadata file
        print("Creating metadata file...")
        self.create_metadata_file()
        print()
        
        # Summary
        print("=" * 60)
        print("EXTRACTION SUMMARY:")
        print("=" * 60)
        
        total_records = 0
        for metric, df in results.items():
            if not df.empty:
                countries = df['country_code'].nunique() if 'country_code' in df.columns else 0
                records = len(df)
                total_records += records
                print(f"{metric.replace('_', ' ').title()}: {records} records, {countries} countries")
            else:
                print(f"{metric.replace('_', ' ').title()}: No data extracted")
        
        print(f"\nTotal records extracted: {total_records}")
        print(f"Output directory: {self.output_path}/")
        
        if total_records > 0:
            print("\n✅ Data extraction completed successfully!")
            print(f"Data Source: World Inequality Database (wid.world, accessed {datetime.now().strftime('%Y-%m-%d')})")
            print("Ready for visualization phase.")
        else:
            print("\n❌ No data extracted. Check source datasets.")
        
        return results

def main():
    """Main extraction function"""
    extractor = WIDDataExtractor()
    
    # Run extraction for all metrics
    results = extractor.extract_all_metrics()
    
    return results

if __name__ == "__main__":
    main()