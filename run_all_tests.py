#!/usr/bin/env python3
"""
Test runner for all median income extraction tests
Runs comprehensive test suite and generates summary report
"""

import unittest
import sys
import time
from io import StringIO

def run_test_suite():
    """Run all test suites and generate summary"""
    print("=" * 80)
    print("MEDIAN INCOME EXTRACTION - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    start_time = time.time()
    
    # Test modules to run
    test_modules = [
        'test_median_income_extraction',
        'test_full_extraction_pipeline',
        'test_gini_algorithm',
        'test_gini_corner_cases',
        'test_mmr_algorithm',
        'test_wealth_distribution'
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for module in test_modules:
        print(f"\n{'-' * 60}")
        print(f"Running {module}")
        print(f"{'-' * 60}")
        
        # Capture test output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=1)
        
        try:
            # Import and run the test module
            imported_module = __import__(module)
            suite = unittest.TestLoader().loadTestsFromModule(imported_module)
            result = runner.run(suite)
            
            # Update counters
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            
            # Print results
            print(f"Tests run: {result.testsRun}")
            if result.failures:
                print(f"Failures: {len(result.failures)}")
                for test, _ in result.failures:
                    print(f"  FAIL: {test}")
            if result.errors:
                print(f"Errors: {len(result.errors)}")
                for test, _ in result.errors:
                    print(f"  ERROR: {test}")
            
            if result.wasSuccessful():
                print("✓ ALL TESTS PASSED")
            else:
                print("✗ SOME TESTS FAILED")
                
        except Exception as e:
            print(f"ERROR loading {module}: {e}")
            total_errors += 1
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED - Median income extraction is working correctly!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Please review the failures above")
        return False

if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)