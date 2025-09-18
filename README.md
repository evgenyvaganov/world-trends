# World Trends Analysis: G20 Economic Inequality (1995-2023)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A comprehensive data analysis project examining economic inequality trends across G20 nations using the World Inequality Database (WID). This project analyzes five key metrics: Median PPP income, Mean PPP income, Mean to Median Ratio (MMR), Wealth distribution across percentiles, and Gini coefficients.

## 📊 Project Overview

This project provides a complete pipeline for analyzing economic inequality across the G20 nations from 1995 to 2023. It includes:

- **Data Verification**: Validates completeness of WID datasets
- **Data Extraction**: Processes raw WID data into clean, analysis-ready CSV files
- **Visualization**: Generates comprehensive plots showing trends and comparisons

### Key Metrics Analyzed

1. **Median PPP Income per Person** (2021 ICP baseline)
   - Median pre-tax national income per adult
   - Purchasing Power Parity adjusted for international comparison
   - Shows typical income levels across countries

2. **Mean PPP Income per Person** (2021 ICP baseline)
   - Average pre-tax national income per adult
   - Purchasing Power Parity adjusted for international comparison
   - Used together with median to calculate MMR

3. **Mean to Median Ratio (MMR)** - Income distribution skewness indicator
   - Ratio of mean income to median income (Mean ÷ Median)
   - MMR = 1.0 indicates perfect equality (mean = median)
   - MMR > 1.0 indicates right-skewed distribution (wealthy outliers pull mean above median)
   - MMR < 1.0 is rare but possible in specific economic conditions
   - Higher MMR values suggest greater income concentration at the top

4. **Wealth Distribution** by percentile brackets (bottom 10%, 50%, top 50%, 10%, 1%)
   - Share of total wealth held by different income groups
   - Reveals concentration of wealth within societies
   - Tracks changes in wealth inequality over time

5. **Gini Coefficients** (calculated from income percentile data)
   - Single measure of income inequality (0 = perfect equality, 1 = maximum inequality)
   - Calculated using Lorenz curve integration from WID percentile data
   - Enables direct comparison of inequality levels between countries

### G20 Countries Covered

Argentina (AR), Australia (AU), Brazil (BR), Canada (CA), China (CN), France (FR), Germany (DE), India (IN), Indonesia (ID), Italy (IT), Japan (JP), South Korea (KR), Mexico (MX), Russia (RU), Saudi Arabia (SA), South Africa (ZA), Turkey (TR), United Kingdom (GB), United States (US)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/world-trends.git
cd world-trends

# Set up environment (conda)
conda create -n world-trends python=3.11 -y
conda activate world-trends
conda install pandas numpy matplotlib seaborn -y

# Download WID data (if running extraction)
# Visit https://wid.world/data/ and download full dataset to datasets/wid_all_data/

# Run the complete pipeline
python main.py all

# Or run individual components
python main.py verify    # Check data completeness
python main.py extract   # Extract and process data
python main.py visualize # Generate visualizations
```

## 📦 Installation

### Prerequisites

- Linux/macOS/Windows with WSL2
- ~500MB free disk space for conda installation
- ~2GB for full environment with packages

### Installation

#### Step 1: Install Miniconda

```bash
# Download Miniconda installer (choose appropriate for your system)
# For Linux/WSL (x86_64):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For Linux/WSL (ARM64/aarch64):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

# For macOS (Intel):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# For macOS (Apple Silicon):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Make installer executable and run
bash Miniconda3-latest-*.sh

# Follow the prompts:
# - Press ENTER to review license
# - Type 'yes' to accept license
# - Press ENTER to confirm installation location or specify custom path
# - Type 'yes' when asked to initialize Miniconda3

# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc for zsh users
```

#### Step 2: Create Conda Environment

```bash
# Navigate to project directory
cd /path/to/world-trends

# Create conda environment with Python 3.11
conda create -n world-trends python=3.11 -y

# Activate the environment
conda activate world-trends
```

#### Step 3: Install Required Packages

```bash
# Install all required packages via conda
conda install pandas numpy matplotlib seaborn -y

# Alternative: Install via pip if conda packages unavailable
# pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
# Test if all packages are installed correctly
python -c "import pandas, numpy, matplotlib, seaborn; print('✅ All packages installed successfully!')"
```

#### Step 5: Download WID Dataset (Optional)

If you plan to run the data extraction pipeline, you'll need to download the WID dataset:

1. Visit https://wid.world/data/
2. Click "DOWNLOAD FULL DATASET" 
3. Extract the ZIP file to `datasets/wid_all_data/`
4. Verify the directory contains WID_data_*.csv files (should be ~845 CSV files)

**Note**: This step is optional. The WID dataset is only required for running the extraction pipeline. Pre-generated visualizations and CSV files are already included in the repository.

## 📁 Project Structure

```
world-trends/
├── main.py                  # Main orchestrator script
├── verify_datasets.py       # Dataset verification module
├── extract_data.py          # Data extraction and cleaning module
├── visualize_trends.py      # Visualization generation module
├── requirements.txt         # Python package dependencies
├── ppp_conversion_factors_2011.py  # PPP conversion factors
├── run_all_tests.py         # Test runner script
├── test_*.py               # Various test files
├── datasets/
│   ├── wid_all_data/       # Original WID data files (source)
│   ├── median_ppp_income_g20.csv    # Median PPP income data (generated)
│   ├── income_mean_median_mmr_g20.csv # Mean/median/MMR data (generated)
│   ├── wealth_distribution_g20.csv  # Wealth distribution data (generated)
│   ├── inequality_gini_g20.csv      # Gini coefficients (generated)
│   └── extraction_metadata.json     # Extraction metadata (generated)
└── plots/                   # Generated visualization outputs
    ├── income_trends_g20.png/pdf
    ├── wealth_distribution_2023.png/pdf
    ├── wealth_trends_g20.png/pdf
    ├── inequality_trends_g20.png/pdf
    ├── mmr_trends_g20.png/pdf
    └── comparative_dashboard.png/pdf
```

## 🎯 Usage

### Running the Complete Pipeline

```bash
# Activate conda environment
conda activate world-trends

# Run complete analysis pipeline (all steps)
python main.py

# Or explicitly:
python main.py all
```

### Running Individual Components

```bash
# Run only dataset verification
python main.py verify

# Run only data extraction
python main.py extract

# Run only visualization generation
python main.py visualize
```

### Expected Output

After running the complete pipeline, you'll find:

1. **Clean Datasets** in `datasets/`:
   - `median_ppp_income_g20.csv` - Median PPP income trends for all G20 countries
   - `wealth_distribution_g20.csv` - Wealth distribution by percentiles
   - `inequality_gini_g20.csv` - Calculated Gini coefficients
   - `income_mean_median_mmr_g20.csv` - Mean, median income and MMR values
   - `extraction_metadata.json` - Metadata about the extraction process

2. **Visualizations** in `plots/`:
   - Income trends over time for all G20 countries
   - Wealth distribution snapshot (2023) and trends over time
   - Inequality (Gini coefficient) trend analysis
   - Mean to Median Ratio (MMR) trends and rankings
   - Comprehensive comparative dashboard (3x3 grid)

3. **Reports**:
   - `verification_report.json` - Dataset completeness verification

## 🔧 Troubleshooting

### SSL Certificate Issues with Conda

If you encounter SSL certificate errors when installing packages:

```bash
# Temporarily disable SSL verification (use with caution)
conda config --set ssl_verify no

# Install packages
conda install [packages]

# Re-enable SSL verification after installation
conda config --set ssl_verify yes
```

### Environment Activation Issues

If `conda activate` doesn't work:

```bash
# Initialize conda for your shell
conda init bash  # or zsh, fish, etc.

# Restart your terminal or reload configuration
source ~/.bashrc
```

### Memory Issues

For systems with limited memory, install packages one at a time:

```bash
conda install pandas -y
conda install numpy -y
conda install matplotlib -y
# ... continue for other packages
```

## 📊 Data Source

This project uses data from the [World Inequality Database (WID)](https://wid.world/), which provides extensive data on income and wealth inequality worldwide.

The WID dataset can be downloaded following the instructions in [Step 5 of the Installation Guide](#step-5-download-wid-dataset-optional).

The complete WID dataset includes:
- Country-specific data files (WID_data_XX.csv)
- Metadata files (WID_metadata_XX.csv)
- Country reference file (WID_countries.csv)
- 845 CSV files covering all countries and regions
- Complete historical data through 2023

## 🛠️ Development

### Adding New Metrics

To add new metrics for analysis:

1. Modify `extract_data.py` to extract additional variables from WID data
2. Update `visualize_trends.py` to create new plot types
3. Update the pipeline in `main.py` if needed

### Extending Country Coverage

To analyze additional countries beyond G20:

1. Update the `g20_countries` dictionary in `extract_data.py`
2. Ensure the corresponding WID data files exist in `datasets/wid_all_data/`
3. Re-run the extraction pipeline

## 📝 Requirements

### System Requirements
- Python 3.11 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Python Packages
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data License Note**: The underlying data from the World Inequality Database (WID) is subject to its own terms of use. Please refer to [WID's terms](https://wid.world/terms-of-use/) for data usage restrictions.

## 🙏 Acknowledgments

- World Inequality Database (WID) for providing comprehensive inequality data
- The open-source community for the excellent data analysis tools

## 📧 Contact

For questions, feedback, or collaboration opportunities, please:
- Open an issue on this repository
- Submit a pull request with improvements
- Refer to the [Contributing](#-contributing) section

---

**Note**: This project processes economic data for research purposes. The Gini coefficient calculations are simplified approximations and may not match official statistics. For authoritative data, please refer to official sources.