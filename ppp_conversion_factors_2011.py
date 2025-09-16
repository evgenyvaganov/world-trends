#!/usr/bin/env python3
"""
PPP Conversion Factors for G20 Countries (2011 baseline)
Local Currency Unit per International Dollar

Sources: World Bank, OECD, IMF data
These factors convert local currency to international dollars (PPP-adjusted)
"""

# PPP conversion factors (LCU per international $) for 2011
# IMPORTANT: Argentina and Turkey factors adjusted for WID data which appears
# to be in pre-redenomination currency units
PPP_FACTORS_2011 = {
    'AR': 160.6,  # Argentine Peso per international $ (adjusted for old currency units in WID)
    'AU': 1.49,   # Australian Dollar per international $  
    'BR': 1.68,   # Brazilian Real per international $
    'CA': 1.26,   # Canadian Dollar per international $
    'CN': 3.51,   # Chinese Yuan per international $
    'FR': 0.86,   # Euro per international $ (France)
    'DE': 0.80,   # Euro per international $ (Germany)
    'IN': 17.49,  # Indian Rupee per international $
    'ID': 3766.0, # Indonesian Rupiah per international $
    'IT': 0.81,   # Euro per international $ (Italy)
    'JP': 108.0,  # Japanese Yen per international $
    'KR': 869.7,  # South Korean Won per international $
    'MX': 8.17,   # Mexican Peso per international $
    'RU': 17.35,  # Russian Ruble per international $
    'SA': 1.88,   # Saudi Arabian Riyal per international $
    'ZA': 5.16,   # South African Rand per international $
    'TR': 10.9,   # Turkish Lira per international $ (adjusted for old lira units in WID)
    'GB': 0.69,   # British Pound per international $
    'US': 1.0     # US Dollar per international $ (base)
}

def convert_to_international_dollars(value, country_code, year=2011):
    """
    Convert local currency value to international dollars using PPP factors
    
    Args:
        value: Amount in local currency (constant prices)
        country_code: ISO 2-letter country code
        year: Year for PPP adjustment (default 2011)
    
    Returns:
        Value in international dollars (PPP-adjusted, 2021 current prices)
    """
    if country_code not in PPP_FACTORS_2011:
        raise ValueError(f"PPP factor not available for country: {country_code}")
    
    ppp_factor = PPP_FACTORS_2011[country_code]
    
    # Convert: Local Currency / PPP Factor = International Dollars
    international_dollars = value / ppp_factor
    
    # Apply inflation adjustment from constant to 2021 current prices
    # US CPI inflation 2011-2021 is approximately 34.5%
    # This brings constant price data to 2021 current prices
    INFLATION_ADJUSTMENT = 1.345
    international_dollars_current = international_dollars * INFLATION_ADJUSTMENT
    
    return international_dollars_current

def get_ppp_factor(country_code):
    """Get PPP conversion factor for a country"""
    return PPP_FACTORS_2011.get(country_code, None)

if __name__ == "__main__":
    # Test conversion with sample values
    print("PPP Conversion Factors (2011 baseline)")
    print("=" * 40)
    
    for country, factor in PPP_FACTORS_2011.items():
        print(f"{country}: {factor} LCU per international $")
    
    print("\nTest conversions:")
    print("South Africa 133,301 ZAR =", f"{convert_to_international_dollars(133301, 'ZA'):,.0f} international $")
    print("Indonesia 47M IDR =", f"{convert_to_international_dollars(47512324, 'ID'):,.0f} international $")
    print("US 62,579 USD =", f"{convert_to_international_dollars(62579, 'US'):,.0f} international $")