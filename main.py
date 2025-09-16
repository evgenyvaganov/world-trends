#!/usr/bin/env python3
"""
This is the main orchestrator that can run the complete pipeline or individual subprograms:
1. verify_datasets.py - Original dataset verification
2. extract_data.py - Data extraction to clean CSV files  
3. visualize_trends.py - Visualization from clean datasets
"""

import sys
import os
import argparse

def run_verification():
    """Run dataset verification"""
    print("=== RUNNING DATASET VERIFICATION ===\n")
    from verify_datasets import main as verify_main
    verify_main()

def run_extraction():
    """Run data extraction"""
    print("=== RUNNING DATA EXTRACTION ===\n")
    from extract_data import main as extract_main
    extract_main()

def run_visualization():
    """Run trend visualization"""
    print("=== RUNNING TREND VISUALIZATION ===\n")
    from visualize_trends import main as visualize_main
    visualize_main()

def run_complete_pipeline():
    """Run the complete analysis pipeline"""
    print("=" * 80)
    print("WORLD TRENDS ANALYSIS: G20 Economic Inequality (1995-2024)")
    print("=" * 80)
    print("Running complete pipeline...\n")
    
    try:
        # Step 1: Verify datasets
        print("STEP 1/3: Dataset Verification")
        print("-" * 40)
        run_verification()
        print()
        
        # Step 2: Extract clean data
        print("STEP 2/3: Data Extraction")
        print("-" * 40)
        run_extraction()
        print()
        
        # Step 3: Create visualizations
        print("STEP 3/3: Trend Visualization")
        print("-" * 40)
        run_visualization()
        print()
        
        print("=" * 80)
        print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("=" * 80)
        print("Check the following directories for results:")
        print("  📊 plots/ - Generated visualizations")
        print("  📋 datasets/ - Clean CSV files")
        print("  📄 verification_report.json - Dataset verification report")
        
    except Exception as e:
        print(f"❌ Pipeline failed at current step: {e}")
        sys.exit(1)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="World Trends Analysis: G20 Economic Inequality (1995-2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available subprograms:
  verify    - Verify original WID datasets completeness
  extract   - Extract clean CSV files from WID data  
  visualize - Create analysis plots from clean data
  all       - Run complete pipeline (default)

Examples:
  python main.py              # Run complete pipeline
  python main.py verify       # Only verify datasets
  python main.py extract      # Only extract data
  python main.py visualize    # Only create visualizations
        """
    )
    
    parser.add_argument(
        'command', 
        nargs='?', 
        default='all',
        choices=['verify', 'extract', 'visualize', 'all'],
        help='Subprogram to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.command == 'verify':
        run_verification()
    elif args.command == 'extract':
        run_extraction()
    elif args.command == 'visualize':
        run_visualization()
    elif args.command == 'all':
        run_complete_pipeline()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()