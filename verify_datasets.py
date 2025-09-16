#!/usr/bin/env python3
"""
Dataset Verification: WID Data Completeness Check

Verifies that the original WID datasets contain all necessary data
for G20 economic inequality analysis (1995-2024).

Usage: python verify_datasets.py
"""

import os
import pandas as pd
from typing import Dict, List, Tuple

class WIDDatasetVerifier:
    """Verifies WID dataset completeness for research requirements"""
    
    def __init__(self, data_path: str = "datasets/wid_all_data"):
        self.data_path = data_path
        self.g20_countries = {
            'AR': 'Argentina', 'AU': 'Australia', 'BR': 'Brazil', 'CA': 'Canada',
            'CN': 'China', 'FR': 'France', 'DE': 'Germany', 'IN': 'India',
            'ID': 'Indonesia', 'IT': 'Italy', 'JP': 'Japan', 'KR': 'South Korea',
            'MX': 'Mexico', 'RU': 'Russia', 'SA': 'Saudi Arabia', 'ZA': 'South Africa',
            'TR': 'Turkey', 'GB': 'United Kingdom', 'US': 'United States'
        }
        self.target_years = list(range(1995, 2025))  # 1995-2024
        self.verification_results = {}
    
    def check_file_exists(self, country_code: str) -> bool:
        """Check if WID data file exists for country"""
        filepath = os.path.join(self.data_path, f"WID_data_{country_code}.csv")
        return os.path.exists(filepath)
    
    def load_country_data(self, country_code: str) -> pd.DataFrame:
        """Load and clean country data"""
        filepath = os.path.join(self.data_path, f"WID_data_{country_code}.csv")
        
        try:
            df = pd.read_csv(filepath, delimiter=';')
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Filter for target years
            df = df[df['year'].isin(self.target_years)]
            return df
        
        except Exception as e:
            print(f"Error loading {country_code}: {e}")
            return pd.DataFrame()
    
    def check_income_data(self, df: pd.DataFrame, country_code: str) -> Dict:
        """Check PPP income data availability (aptinc992 - pre-tax PPP income)"""
        income_data = df[
            (df['variable'].str.contains('aptinc.*992', regex=True, na=False)) &
            (df['percentile'] == 'p0p100')
        ]
        
        available_years = sorted(income_data['year'].dropna().unique())
        
        return {
            'metric': 'PPP Income (aptinc992)',
            'total_records': len(income_data),
            'years_available': len(available_years),
            'year_range': f"{min(available_years)}-{max(available_years)}" if available_years else "No data",
            'missing_years': [y for y in self.target_years if y not in available_years],
            'coverage_pct': (len(available_years) / len(self.target_years)) * 100,
            'status': 'COMPLETE' if len(available_years) >= 25 else 'INCOMPLETE'
        }
    
    def check_wealth_data(self, df: pd.DataFrame, country_code: str) -> Dict:
        """Check wealth distribution data (shweal variables)"""
        wealth_percentiles = ['p0p10', 'p0p50', 'p50p100', 'p90p100', 'p99p100']
        
        wealth_data = df[
            (df['variable'].str.contains('shweal', regex=True, na=False)) &
            (df['percentile'].isin(wealth_percentiles))
        ]
        
        available_years = sorted(wealth_data['year'].dropna().unique())
        percentiles_found = wealth_data['percentile'].unique()
        
        return {
            'metric': 'Wealth Distribution (shweal)',
            'total_records': len(wealth_data),
            'years_available': len(available_years),
            'year_range': f"{min(available_years)}-{max(available_years)}" if available_years else "No data",
            'percentiles_found': sorted(percentiles_found),
            'missing_percentiles': [p for p in wealth_percentiles if p not in percentiles_found],
            'coverage_pct': (len(available_years) / len(self.target_years)) * 100,
            'status': 'COMPLETE' if len(available_years) >= 20 and len(percentiles_found) >= 3 else 'INCOMPLETE'
        }
    
    def check_inequality_data(self, df: pd.DataFrame, country_code: str) -> Dict:
        """Check data availability for Gini calculation (income percentiles)"""
        income_percentiles = ['p0p50', 'p50p90', 'p90p100', 'p99p100']
        
        percentile_data = df[
            (df['variable'].str.contains('aptinc', regex=True, na=False)) &
            (df['percentile'].isin(income_percentiles))
        ]
        
        available_years = sorted(percentile_data['year'].dropna().unique())
        percentiles_found = percentile_data['percentile'].unique()
        
        return {
            'metric': 'Inequality Data (for Gini calculation)',
            'total_records': len(percentile_data),
            'years_available': len(available_years),
            'year_range': f"{min(available_years)}-{max(available_years)}" if available_years else "No data",
            'percentiles_found': sorted(percentiles_found),
            'coverage_pct': (len(available_years) / len(self.target_years)) * 100,
            'status': 'COMPLETE' if len(available_years) >= 20 and len(percentiles_found) >= 2 else 'INCOMPLETE'
        }
    
    def verify_country(self, country_code: str) -> Dict:
        """Comprehensive verification for a single country"""
        country_name = self.g20_countries[country_code]
        
        print(f"Verifying {country_name} ({country_code})...")
        
        # Check if file exists
        if not self.check_file_exists(country_code):
            return {
                'country_code': country_code,
                'country_name': country_name,
                'file_exists': False,
                'status': 'MISSING FILE'
            }
        
        # Load data
        df = self.load_country_data(country_code)
        
        if df.empty:
            return {
                'country_code': country_code,
                'country_name': country_name,
                'file_exists': True,
                'status': 'EMPTY DATA'
            }
        
        # Check each metric
        income_check = self.check_income_data(df, country_code)
        wealth_check = self.check_wealth_data(df, country_code)
        inequality_check = self.check_inequality_data(df, country_code)
        
        # Overall status
        all_complete = all(check['status'] == 'COMPLETE' for check in [income_check, wealth_check, inequality_check])
        
        return {
            'country_code': country_code,
            'country_name': country_name,
            'file_exists': True,
            'total_records': len(df),
            'year_range': f"{df['year'].min():.0f}-{df['year'].max():.0f}" if not df.empty else "No data",
            'income_data': income_check,
            'wealth_data': wealth_check,
            'inequality_data': inequality_check,
            'overall_status': 'COMPLETE' if all_complete else 'INCOMPLETE'
        }
    
    def verify_all_countries(self) -> Dict:
        """Verify all G20 countries"""
        print("=== WID DATASET VERIFICATION ===")
        print("Checking data completeness for G20 countries (1995-2024)\n")
        
        results = {}
        summary = {
            'total_countries': len(self.g20_countries),
            'countries_verified': 0,
            'complete_countries': 0,
            'incomplete_countries': 0,
            'missing_files': 0
        }
        
        for country_code in self.g20_countries.keys():
            result = self.verify_country(country_code)
            results[country_code] = result
            
            summary['countries_verified'] += 1
            
            if not result.get('file_exists', False):
                summary['missing_files'] += 1
            elif result.get('overall_status') == 'COMPLETE':
                summary['complete_countries'] += 1
            else:
                summary['incomplete_countries'] += 1
        
        self.verification_results = {
            'summary': summary,
            'detailed_results': results
        }
        
        return self.verification_results
    
    def print_summary_report(self):
        """Print formatted verification summary"""
        if not self.verification_results:
            print("No verification results available. Run verify_all_countries() first.")
            return
        
        summary = self.verification_results['summary']
        results = self.verification_results['detailed_results']
        
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        print(f"Total G20 Countries: {summary['total_countries']}")
        print(f"Countries Verified: {summary['countries_verified']}")
        print(f"Complete Data: {summary['complete_countries']}")
        print(f"Incomplete Data: {summary['incomplete_countries']}")
        print(f"Missing Files: {summary['missing_files']}")
        
        print(f"\nData Completeness: {(summary['complete_countries']/summary['total_countries'])*100:.1f}%")
        
        # Complete countries
        if summary['complete_countries'] > 0:
            print(f"\n✅ COUNTRIES WITH COMPLETE DATA ({summary['complete_countries']}):")
            for code, result in results.items():
                if result.get('overall_status') == 'COMPLETE':
                    print(f"   {result['country_name']} ({code}) - {result.get('total_records', 0)} records")
        
        # Incomplete countries
        if summary['incomplete_countries'] > 0:
            print(f"\n⚠️  COUNTRIES WITH INCOMPLETE DATA ({summary['incomplete_countries']}):")
            for code, result in results.items():
                if result.get('overall_status') == 'INCOMPLETE':
                    print(f"   {result['country_name']} ({code})")
                    if 'income_data' in result:
                        print(f"     - Income: {result['income_data']['status']} ({result['income_data']['coverage_pct']:.1f}%)")
                    if 'wealth_data' in result:
                        print(f"     - Wealth: {result['wealth_data']['status']} ({result['wealth_data']['coverage_pct']:.1f}%)")
                    if 'inequality_data' in result:
                        print(f"     - Inequality: {result['inequality_data']['status']} ({result['inequality_data']['coverage_pct']:.1f}%)")
        
        # Missing files
        if summary['missing_files'] > 0:
            print(f"\n❌ MISSING DATA FILES ({summary['missing_files']}):")
            for code, result in results.items():
                if not result.get('file_exists', False):
                    print(f"   {result['country_name']} ({code})")
    
    def export_verification_report(self, filename: str = "verification_report.json"):
        """Export verification results to JSON"""
        import json
        
        if not self.verification_results:
            print("No verification results to export.")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        
        print(f"Verification report exported to {filename}")

def main():
    """Main verification function"""
    verifier = WIDDatasetVerifier()
    
    # Run verification
    results = verifier.verify_all_countries()
    
    # Print summary
    verifier.print_summary_report()
    
    # Export results
    verifier.export_verification_report()
    
    # Final recommendation
    complete_pct = (results['summary']['complete_countries'] / results['summary']['total_countries']) * 100
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    
    if complete_pct >= 80:
        print(f"✅ Dataset is SUFFICIENT for research ({complete_pct:.1f}% complete)")
        print("   Proceed to data extraction phase.")
    elif complete_pct >= 60:
        print(f"⚠️  Dataset is PARTIALLY SUFFICIENT ({complete_pct:.1f}% complete)")
        print("   Research can proceed but may need supplementary data for some countries.")
    else:
        print(f"❌ Dataset is INSUFFICIENT ({complete_pct:.1f}% complete)")
        print("   Additional data sources required for comprehensive analysis.")

if __name__ == "__main__":
    main()