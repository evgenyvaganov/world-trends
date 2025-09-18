#!/usr/bin/env python3
"""
Trend Visualization: Clean Dataset Analysis & Plotting

Creates comprehensive visualizations from the clean CSV datasets:
- PPP income trends over time
- Wealth distribution analysis 
- Inequality (Gini) trends
- Comparative cross-country analysis

Usage: python visualize_trends.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime

class TrendsVisualizer:
    """Creates visualizations from clean extracted datasets"""
    
    def __init__(self, data_path: str = "datasets", output_path: str = "plots"):
        self.data_path = data_path
        self.output_path = output_path
        self.datasets = {}
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Set up plotting style
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Configure matplotlib and seaborn styling"""
        plt.style.use('default')  # Use default instead of deprecated seaborn style
        sns.set_palette("tab10")
        self.attribution_text = f"Source: World Inequality Database (wid.world, accessed {datetime.now().strftime('%Y-%m-%d')})"
        
        # Set global plot parameters
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.titlesize': 16,
            'figure.figsize': [12, 8],
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def add_attribution(self, fig):
        """Add WID attribution text to figure"""
        fig.text(0.5, 0.01, self.attribution_text, ha='center', fontsize=8, 
                style='italic', color='gray')
    
    def load_clean_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all clean CSV datasets"""
        print("Loading clean datasets...")
        
        dataset_files = {
            'ppp_income': 'ppp_income_g20.csv',
            'wealth_distribution': 'wealth_distribution_g20.csv',
            'inequality': 'inequality_gini_g20.csv',
            'mean_median_mmr': 'income_mean_median_mmr_g20.csv'
        }
        
        loaded_datasets = {}
        
        for dataset_name, filename in dataset_files.items():
            filepath = os.path.join(self.data_path, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    loaded_datasets[dataset_name] = df
                    print(f"  ✅ Loaded {dataset_name}: {len(df)} records")
                except Exception as e:
                    print(f"  ❌ Error loading {dataset_name}: {e}")
            else:
                print(f"  ⚠️  File not found: {filename}")
        
        self.datasets = loaded_datasets
        return loaded_datasets
    
    def plot_income_trends(self, save: bool = True, show: bool = True):
        """Plot PPP income trends for G20 countries"""
        if 'ppp_income' not in self.datasets:
            print("❌ PPP income data not available")
            return
        
        df = self.datasets['ppp_income']
        
        print("Creating PPP income trends plot...")
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each country
        countries = df['country_code'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
        
        for i, country in enumerate(countries):
            country_data = df[df['country_code'] == country].sort_values('year')
            country_name = country_data['country_name'].iloc[0]
            
            ax.plot(country_data['year'], country_data['median_ppp_income'], 
                   label=f"{country_name} ({country})", 
                   linewidth=2.5, marker='o', markersize=4, 
                   color=colors[i], alpha=0.8)
        
        ax.set_title('Real Median PPP Income per Person - G20 Countries (1995-2023)', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Median Income (2021 PPP USD)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        self.add_attribution(fig)
        
        if save:
            plt.savefig(os.path.join(self.output_path, 'income_trends_g20.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, 'income_trends_g20.pdf'), 
                       bbox_inches='tight')
            print(f"  💾 Saved: income_trends_g20.png/.pdf")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_wealth_distribution(self, year: int = 2023, save: bool = True, show: bool = True):
        """Plot wealth distribution by percentiles for a specific year"""
        if 'wealth_distribution' not in self.datasets:
            print("❌ Wealth distribution data not available")
            return
        
        df = self.datasets['wealth_distribution']
        
        # Filter for specified year (or closest available)
        available_years = df['year'].unique()
        target_year = min(available_years, key=lambda x: abs(x - year))
        
        if abs(target_year - year) > 2:
            print(f"⚠️  Requested year {year} not available, using {target_year}")
        
        year_data = df[df['year'] == target_year]
        
        if year_data.empty:
            print(f"❌ No wealth distribution data for {target_year}")
            return
        
        print(f"Creating wealth distribution plot for {target_year}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Stacked bar chart with NON-OVERLAPPING groups
        countries = year_data['country_code'].unique()
        
        # Create matrix for plotting with non-overlapping percentiles
        wealth_matrix = []
        country_names = []
        
        for country in countries:
            country_data = year_data[year_data['country_code'] == country]
            if not country_data.empty:
                country_name = country_data['country_name'].iloc[0]
                country_names.append(f"{country_name}\n({country})")
                
                # Get the raw percentile values
                bottom_10 = country_data[country_data['percentile_group'] == 'bottom_10pct']['wealth_share'].iloc[0] if not country_data[country_data['percentile_group'] == 'bottom_10pct'].empty else 0
                bottom_50 = country_data[country_data['percentile_group'] == 'bottom_50pct']['wealth_share'].iloc[0] if not country_data[country_data['percentile_group'] == 'bottom_50pct'].empty else 0
                top_50 = country_data[country_data['percentile_group'] == 'top_50pct']['wealth_share'].iloc[0] if not country_data[country_data['percentile_group'] == 'top_50pct'].empty else 0
                top_10 = country_data[country_data['percentile_group'] == 'top_10pct']['wealth_share'].iloc[0] if not country_data[country_data['percentile_group'] == 'top_10pct'].empty else 0
                top_1 = country_data[country_data['percentile_group'] == 'top_1pct']['wealth_share'].iloc[0] if not country_data[country_data['percentile_group'] == 'top_1pct'].empty else 0
                
                # Calculate non-overlapping groups
                # Bottom 10% stays as is
                # 10-50% = Bottom 50% minus Bottom 10%
                # 50-90% = Top 50% minus Top 10%
                # 90-99% = Top 10% minus Top 1%
                # Top 1% stays as is
                
                bottom_0_10 = bottom_10
                bottom_10_50 = bottom_50 - bottom_10
                middle_50_90 = top_50 - top_10
                top_90_99 = top_10 - top_1
                top_99_100 = top_1
                
                wealth_row = [bottom_0_10, bottom_10_50, middle_50_90, top_90_99, top_99_100]
                wealth_matrix.append(wealth_row)
        
        if wealth_matrix:
            wealth_df = pd.DataFrame(wealth_matrix, 
                                   columns=['Bottom 10%', '10-50%', '50-90%', '90-99%', 'Top 1%'],
                                   index=country_names)
            
            wealth_df.plot(kind='bar', stacked=True, ax=ax1, 
                          colormap='RdYlBu_r', alpha=0.8)
            ax1.set_title(f'Wealth Distribution by Percentiles - G20 ({target_year})')
            ax1.set_xlabel('Country')
            ax1.set_ylabel('Wealth Share (%)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
            ax1.legend(title='Percentile Group', bbox_to_anchor=(1.05, 1))
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Focus on top 1% and top 10%
        if wealth_matrix:
            top1_data = []
            top10_data = []
            
            for country in countries:
                country_data = year_data[year_data['country_code'] == country]
                country_name = country_data['country_name'].iloc[0] if not country_data.empty else country
                
                top1 = country_data[country_data['percentile_group'] == 'top_1pct']['wealth_share']
                top10 = country_data[country_data['percentile_group'] == 'top_10pct']['wealth_share']
                
                if not top1.empty and not top10.empty:
                    top1_data.append((country_name, country, top1.iloc[0]))
                    top10_data.append((country_name, country, top10.iloc[0]))
            
            if top1_data:
                # Sort by top 1% wealth share
                top1_data.sort(key=lambda x: x[2], reverse=True)
                
                countries_sorted = [x[1] for x in top1_data]
                names_sorted = [f"{x[0]}\n({x[1]})" for x in top1_data]
                top1_values = [x[2] for x in top1_data]
                top10_values = [next(y[2] for y in top10_data if y[1] == x[1]) for x in top1_data]
                
                x_pos = np.arange(len(countries_sorted))
                width = 0.35
                
                ax2.bar(x_pos - width/2, top10_values, width, label='Top 10%', alpha=0.8)
                ax2.bar(x_pos + width/2, top1_values, width, label='Top 1%', alpha=0.8)
                
                ax2.set_title(f'Wealth Concentration - Top Percentiles ({target_year})')
                ax2.set_xlabel('Country')
                ax2.set_ylabel('Wealth Share (%)')
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(names_sorted, rotation=45, ha='right')
                ax2.legend()
        
        plt.suptitle(f'G20 Wealth Distribution Analysis ({target_year})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.add_attribution(fig)
        
        if save:
            plt.savefig(os.path.join(self.output_path, f'wealth_distribution_{target_year}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, f'wealth_distribution_{target_year}.pdf'), 
                       bbox_inches='tight')
            print(f"  💾 Saved: wealth_distribution_{target_year}.png/.pdf")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_wealth_distribution_trends(self, save: bool = True, show: bool = True):
        """Plot wealth distribution trends over time for all G20 countries"""
        if 'wealth_distribution' not in self.datasets:
            print("❌ Wealth distribution data not available")
            return
        
        df = self.datasets['wealth_distribution']
        
        print("Creating wealth distribution trends plot...")
        
        # Create 5 subplots for all percentile groups
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        countries = df['country_code'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
        
        # Define percentile groups and their properties
        percentile_configs = [
            ('bottom_10pct', 'Bottom 10%', axes[0], (-5, 5)),
            ('bottom_50pct', 'Bottom 50%', axes[1], (-5, 15)),
            ('top_50pct', 'Top 50%', axes[2], (85, 105)),
            ('top_10pct', 'Top 10%', axes[3], (40, 90)),
            ('top_1pct', 'Top 1%', axes[4], (15, 60))
        ]
        
        # Plot each percentile group
        for percentile_group, title, ax, y_limits in percentile_configs:
            for i, country in enumerate(countries):
                country_data = df[(df['country_code'] == country) & 
                                 (df['percentile_group'] == percentile_group)].sort_values('year')
                if not country_data.empty:
                    country_name = country_data['country_name'].iloc[0]
                    ax.plot(country_data['year'], country_data['wealth_share'] * 100, 
                           label=f"{country} ({country_name})", 
                           linewidth=1.5, marker='o', markersize=2,
                           color=colors[i], alpha=0.7)
            
            ax.set_title(f'{title} Wealth Share Trends', fontweight='bold', fontsize=11)
            ax.set_xlabel('Year', fontsize=9)
            ax.set_ylabel('Wealth Share (%)', fontsize=9)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_limits)
            ax.tick_params(labelsize=8)
            
            # Add zero line for bottom percentiles
            if percentile_group in ['bottom_10pct', 'bottom_50pct']:
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # Add 100% line for top 50%
            if percentile_group == 'top_50pct':
                ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Hide the 6th subplot (we only have 5 percentile groups)
        axes[5].axis('off')
        
        # Add summary text in the empty subplot
        axes[5].text(0.1, 0.8, 'Key Insights:', fontsize=12, fontweight='bold')
        axes[5].text(0.1, 0.65, '• Top 1% owns 20-55% of wealth', fontsize=10)
        axes[5].text(0.1, 0.55, '• Top 10% owns 55-85% of wealth', fontsize=10)
        axes[5].text(0.1, 0.45, '• Top 50% owns 85-103% of wealth', fontsize=10)
        axes[5].text(0.1, 0.35, '• Bottom 50% owns -3% to 7% of wealth', fontsize=10)
        axes[5].text(0.1, 0.25, '• Bottom 10% often has negative wealth', fontsize=10)
        axes[5].text(0.1, 0.1, 'Note: Negative wealth = debt exceeds assets', fontsize=9, style='italic')
        
        plt.suptitle('Wealth Distribution Trends by Percentile Group - G20 Countries (1995-2023)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.add_attribution(fig)
        
        if save:
            plt.savefig(os.path.join(self.output_path, 'wealth_trends_g20.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, 'wealth_trends_g20.pdf'), 
                       bbox_inches='tight')
            print(f"  💾 Saved: wealth_trends_g20.png/.pdf")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_mmr_trends(self, save: bool = True, show: bool = True):
        """Plot Mean to Median Ratio (MMR) trends over time"""
        if 'mean_median_mmr' not in self.datasets:
            print("❌ Mean/Median/MMR data not available")
            return
        
        df = self.datasets['mean_median_mmr']
        
        print("Creating MMR trends plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        countries = df['country_code'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
        
        # Plot 1: MMR trends over time
        for i, country in enumerate(countries):
            country_data = df[df['country_code'] == country].sort_values('year')
            if not country_data.empty:
                country_name = country_data['country_name'].iloc[0]
                ax1.plot(country_data['year'], country_data['mmr'], 
                        label=f"{country_name} ({country})", 
                        linewidth=2, marker='o', markersize=3,
                        color=colors[i], alpha=0.8)
        
        ax1.set_title('Mean to Median Ratio (MMR) Trends - G20 Countries (1995-2023)', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('MMR (Mean ÷ Median)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect equality (MMR=1)')
        ax1.set_ylim(1.0, 4.5)
        
        # Plot 2: Latest MMR ranking
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].sort_values('mmr', ascending=True)
        
        if not latest_data.empty:
            country_labels = [f"{row['country_name']}\n({row['country_code']})" 
                            for _, row in latest_data.iterrows()]
            
            # Color bars based on MMR level
            colors_bars = []
            for mmr in latest_data['mmr']:
                if mmr > 2.5:
                    colors_bars.append('red')       # Extreme skew
                elif mmr > 2.0:
                    colors_bars.append('orange')    # High skew
                elif mmr > 1.5:
                    colors_bars.append('yellow')    # Moderate skew
                else:
                    colors_bars.append('green')     # Low skew
            
            bars = ax2.barh(range(len(latest_data)), latest_data['mmr'], color=colors_bars, alpha=0.7)
            ax2.set_yticks(range(len(latest_data)))
            ax2.set_yticklabels(country_labels, fontsize=8)
            ax2.set_xlabel('MMR (Mean ÷ Median)')
            ax2.set_title(f'Income Skewness Ranking - G20 Countries ({latest_year})')
            ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
            ax2.text(1.05, len(latest_data)-1, 'Perfect\nEquality', fontsize=8, ha='left')
            
            # Add MMR value labels on bars
            for i, (bar, mmr) in enumerate(zip(bars, latest_data['mmr'])):
                ax2.text(mmr + 0.05, i, f'{mmr:.2f}', va='center', fontsize=8)
        
        plt.suptitle('Income Distribution Skewness Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.add_attribution(fig)
        
        if save:
            plt.savefig(os.path.join(self.output_path, 'mmr_trends_g20.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, 'mmr_trends_g20.pdf'), 
                       bbox_inches='tight')
            print(f"  💾 Saved: mmr_trends_g20.png/.pdf")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_inequality_trends(self, save: bool = True, show: bool = True):
        """Plot Gini coefficient trends"""
        if 'inequality' not in self.datasets:
            print("❌ Inequality data not available")
            return
        
        df = self.datasets['inequality']
        
        print("Creating inequality trends plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Time series of Gini coefficients
        countries = df['country_code'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
        
        for i, country in enumerate(countries):
            country_data = df[df['country_code'] == country].sort_values('year')
            country_name = country_data['country_name'].iloc[0]
            
            ax1.plot(country_data['year'], country_data['gini_coefficient'], 
                    label=f"{country_name} ({country})", 
                    linewidth=2, marker='s', markersize=4,
                    color=colors[i], alpha=0.8)
        
        ax1.set_title('Inequality Trends - Gini Coefficients (1995-2023)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Gini Coefficient')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Latest Gini values ranking
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].copy()
        
        if not latest_data.empty:
            latest_data = latest_data.sort_values('gini_coefficient', ascending=True)
            
            country_labels = [f"{row['country_name']}\n({row['country_code']})" 
                            for _, row in latest_data.iterrows()]
            
            bars = ax2.barh(range(len(latest_data)), latest_data['gini_coefficient'], alpha=0.7)
            ax2.set_yticks(range(len(latest_data)))
            ax2.set_yticklabels(country_labels)
            ax2.set_xlabel('Gini Coefficient')
            ax2.set_title(f'Inequality Ranking - G20 Countries ({latest_year})')
            
            # Color bars based on inequality level
            for i, (bar, gini) in enumerate(zip(bars, latest_data['gini_coefficient'])):
                if gini < 0.3:
                    bar.set_color('green')
                elif gini < 0.4:
                    bar.set_color('yellow')
                elif gini < 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        plt.suptitle('G20 Income Inequality Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.add_attribution(fig)
        
        if save:
            plt.savefig(os.path.join(self.output_path, 'inequality_trends_g20.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, 'inequality_trends_g20.pdf'), 
                       bbox_inches='tight')
            print(f"  💾 Saved: inequality_trends_g20.png/.pdf")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparative_dashboard(self, save: bool = True, show: bool = True):
        """Create comprehensive comparative analysis dashboard"""
        print("Creating comparative analysis dashboard...")
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Income Growth Rates (top left, double width)
        if 'ppp_income' in self.datasets:
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_growth_rates(ax1)
        
        # 2. Income vs Inequality Scatter (top right)
        if 'ppp_income' in self.datasets and 'inequality' in self.datasets:
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_income_inequality_scatter(ax2)
        
        # 3. MMR Latest Rankings (middle left)
        if 'mean_median_mmr' in self.datasets:
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_mmr_ranking(ax3)
        
        # 4. Wealth Concentration Top 1% (middle center)
        if 'wealth_distribution' in self.datasets:
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_wealth_concentration(ax4)
        
        # 5. Latest Rankings Summary (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_latest_rankings(ax5)
        
        # 6. Key Metrics Summary (bottom, triple width)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_key_metrics_expanded(ax6)
        
        plt.suptitle('G20 Economic Inequality Dashboard - Key Metrics Overview', 
                    fontsize=16, fontweight='bold')
        
        self.add_attribution(fig)
        
        if save:
            plt.savefig(os.path.join(self.output_path, 'comparative_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, 'comparative_dashboard.pdf'), 
                       bbox_inches='tight')
            print(f"  💾 Saved: comparative_dashboard.png/.pdf")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_growth_rates(self, ax):
        """Helper: Plot income growth rates"""
        if 'ppp_income' not in self.datasets:
            return
        
        df = self.datasets['ppp_income']
        
        growth_data = []
        for country in df['country_code'].unique():
            country_data = df[df['country_code'] == country].sort_values('year')
            if len(country_data) > 1:
                first_income = country_data.iloc[0]['median_ppp_income']
                last_income = country_data.iloc[-1]['median_ppp_income']
                years_span = country_data.iloc[-1]['year'] - country_data.iloc[0]['year']
                
                if years_span > 0 and first_income > 0:
                    growth_rate = ((last_income / first_income) ** (1/years_span) - 1) * 100
                    country_name = country_data.iloc[0]['country_name']
                    growth_data.append((country_name, country, growth_rate))
        
        if growth_data:
            growth_data.sort(key=lambda x: x[2], reverse=True)
            names = [f"{x[0]} ({x[1]})" for x in growth_data]
            rates = [x[2] for x in growth_data]
            
            bars = ax.bar(range(len(names)), rates, alpha=0.7)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel('Annual Growth Rate (%)')
            ax.set_title('Average Annual Income Growth Rates')
            
            # Color bars based on growth rate
            for bar, rate in zip(bars, rates):
                if rate > 5:
                    bar.set_color('green')
                elif rate > 2:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
    
    def _plot_income_inequality_scatter(self, ax):
        """Helper: Income vs Inequality scatter plot"""
        if not all(k in self.datasets for k in ['ppp_income', 'inequality']):
            return
        
        # Get latest year data
        income_df = self.datasets['ppp_income']
        inequality_df = self.datasets['inequality']
        
        latest_year = min(income_df['year'].max(), inequality_df['year'].max())
        
        income_latest = income_df[income_df['year'] == latest_year]
        inequality_latest = inequality_df[inequality_df['year'] == latest_year]
        
        # Merge data
        merged = pd.merge(income_latest, inequality_latest, on='country_code')
        
        if not merged.empty:
            ax.scatter(merged['median_ppp_income'], merged['gini_coefficient'], 
                      s=100, alpha=0.7)
            
            for _, row in merged.iterrows():
                ax.annotate(row['country_code'], 
                          (row['median_ppp_income'], row['gini_coefficient']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('PPP Income')
            ax.set_ylabel('Gini Coefficient')
            ax.set_title(f'Income vs Inequality ({latest_year})')
    
    def _plot_latest_rankings(self, ax):
        """Helper: Show latest rankings for key metrics"""
        # Get latest year data
        latest_data = {}
        
        if 'inequality' in self.datasets:
            df = self.datasets['inequality']
            latest_year = df['year'].max()
            gini_data = df[df['year'] == latest_year].sort_values('gini_coefficient')
            
            # Top 3 most equal
            most_equal = gini_data.head(3)
            # Top 3 most unequal  
            most_unequal = gini_data.tail(3)
            
            y_pos = 0.9
            ax.text(0.5, y_pos, 'Inequality Rankings (Gini)', fontsize=11, fontweight='bold',
                   ha='center', transform=ax.transAxes)
            
            y_pos -= 0.15
            ax.text(0.2, y_pos, 'Most Equal:', fontsize=9, fontweight='bold',
                   transform=ax.transAxes)
            for _, row in most_equal.iterrows():
                y_pos -= 0.08
                ax.text(0.2, y_pos, f"{row['country_code']}: {row['gini_coefficient']:.3f}", 
                       fontsize=8, transform=ax.transAxes)
            
            y_pos = 0.75 - 0.15
            ax.text(0.6, y_pos, 'Most Unequal:', fontsize=9, fontweight='bold',
                   transform=ax.transAxes)
            for _, row in most_unequal.iterrows():
                y_pos -= 0.08
                ax.text(0.6, y_pos, f"{row['country_code']}: {row['gini_coefficient']:.3f}", 
                       fontsize=8, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Latest Rankings', fontsize=10)
    
    def _plot_wealth_concentration(self, ax):
        """Helper: Wealth concentration plot"""
        if 'wealth_distribution' not in self.datasets:
            return
        
        df = self.datasets['wealth_distribution']
        latest_year = df['year'].max()
        
        # Get top 1% wealth share
        top1_data = df[
            (df['year'] == latest_year) & 
            (df['percentile_group'] == 'top_1pct')
        ].sort_values('wealth_share', ascending=False)
        
        if not top1_data.empty:
            countries = [f"{row['country_name']}\n({row['country_code']})" 
                        for _, row in top1_data.iterrows()]
            values = top1_data['wealth_share'].values
            
            bars = ax.bar(range(len(countries)), values, alpha=0.7)
            ax.set_xticks(range(len(countries)))
            ax.set_xticklabels(countries, rotation=45, ha='right')
            ax.set_ylabel('Top 1% Wealth Share')
            ax.set_title(f'Wealth Concentration ({latest_year})')
    
    
    def _plot_mmr_ranking(self, ax):
        """Helper: Show MMR rankings"""
        if 'mean_median_mmr' not in self.datasets:
            return
        
        df = self.datasets['mean_median_mmr']
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].sort_values('mmr', ascending=True)
        
        if not latest_data.empty:
            # Take top 10 for readability
            plot_data = latest_data.head(10)
            countries = [f"{row['country_code']}" for _, row in plot_data.iterrows()]
            mmr_values = plot_data['mmr'].values
            
            # Color bars
            colors_bars = []
            for mmr in mmr_values:
                if mmr > 2.5:
                    colors_bars.append('red')
                elif mmr > 2.0:
                    colors_bars.append('orange')
                elif mmr > 1.5:
                    colors_bars.append('yellow')
                else:
                    colors_bars.append('green')
            
            bars = ax.bar(range(len(countries)), mmr_values, color=colors_bars, alpha=0.7)
            ax.set_xticks(range(len(countries)))
            ax.set_xticklabels(countries, rotation=45)
            ax.set_ylabel('MMR')
            ax.set_title(f'Income Skewness (MMR) {latest_year}')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    def _plot_key_metrics_expanded(self, ax):
        """Helper: Display expanded key summary metrics"""
        metrics_text = []
        
        # Income metrics
        if 'mean_median_mmr' in self.datasets:
            df = self.datasets['mean_median_mmr']
            latest_year = df['year'].max()
            latest = df[df['year'] == latest_year]
            
            highest_mmr = latest.nlargest(1, 'mmr')
            lowest_mmr = latest.nsmallest(1, 'mmr')
            
            if not highest_mmr.empty:
                metrics_text.append("INCOME DISTRIBUTION SKEWNESS (MMR):")
                metrics_text.append(f"  Most Skewed: {highest_mmr.iloc[0]['country_code']} (MMR: {highest_mmr.iloc[0]['mmr']:.2f})")
                metrics_text.append(f"  Least Skewed: {lowest_mmr.iloc[0]['country_code']} (MMR: {lowest_mmr.iloc[0]['mmr']:.2f})")
                metrics_text.append("")
        
        if 'ppp_income' in self.datasets:
            df = self.datasets['ppp_income']
            latest_year = df['year'].max()
            latest = df[df['year'] == latest_year]
            
            highest_income = latest.nlargest(1, 'median_ppp_income')
            lowest_income = latest.nsmallest(1, 'median_ppp_income')
            
            if not highest_income.empty:
                metrics_text.append("MEDIAN INCOME LEVELS:")
                metrics_text.append(f"  Highest: {highest_income.iloc[0]['country_code']}: ${highest_income.iloc[0]['median_ppp_income']:,.0f}")
                metrics_text.append(f"  Lowest: {lowest_income.iloc[0]['country_code']}: ${lowest_income.iloc[0]['median_ppp_income']:,.0f}")
                metrics_text.append("")
        
        if 'wealth_distribution' in self.datasets:
            df = self.datasets['wealth_distribution']
            latest_year = df['year'].max()
            top1 = df[(df['year'] == latest_year) & (df['percentile_group'] == 'top_1pct')]
            
            if not top1.empty:
                max_concentration = top1.nlargest(1, 'wealth_share')
                min_concentration = top1.nsmallest(1, 'wealth_share')
                metrics_text.append("WEALTH CONCENTRATION (TOP 1%):")
                metrics_text.append(f"  Highest: {max_concentration.iloc[0]['country_code']}: {max_concentration.iloc[0]['wealth_share']*100:.1f}%")
                metrics_text.append(f"  Lowest: {min_concentration.iloc[0]['country_code']}: {min_concentration.iloc[0]['wealth_share']*100:.1f}%")
        
        # Display text in three columns
        col_width = 0.33
        col_starts = [0.05, 0.35, 0.65]
        
        lines_per_col = len(metrics_text) // 3 + 1
        for i, line in enumerate(metrics_text):
            col = i // lines_per_col
            row = i % lines_per_col
            
            if col < 3:  # Only show if fits in 3 columns
                x_pos = col_starts[col]
                y_pos = 0.9 - (row * 0.12)
                
                if line.startswith("  "):
                    ax.text(x_pos + 0.02, y_pos, line, fontsize=9, transform=ax.transAxes)
                elif line == "":
                    continue
                else:
                    ax.text(x_pos, y_pos, line, fontsize=10, fontweight='bold', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Key Metrics Summary - All Indicators', fontsize=12)
    
    def create_all_visualizations(self, show: bool = True):
        """Create all visualization plots"""
        print("=== TREND VISUALIZATION ===")
        print("Creating comprehensive analysis plots...\n")
        
        # Load datasets
        self.load_clean_datasets()
        print()
        
        if not self.datasets:
            print("❌ No datasets available for visualization")
            return
        
        # Create individual plots
        print("1. Income Trends Analysis:")
        self.plot_income_trends(show=show)
        print()
        
        print("2. Wealth Distribution Analysis:")
        self.plot_wealth_distribution(show=show)
        print()
        
        print("3. Wealth Distribution Trends:")
        self.plot_wealth_distribution_trends(show=show)
        print()
        
        print("4. Inequality Trends Analysis:")
        self.plot_inequality_trends(show=show)
        print()
        
        print("5. MMR Trends Analysis:")
        self.plot_mmr_trends(show=show)
        print()
        
        print("6. Comparative Dashboard:")
        self.plot_comparative_dashboard(show=show)
        print()
        
        # Summary
        print("=" * 60)
        print("VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"All plots saved to: {self.output_path}/")
        
        plot_files = [f for f in os.listdir(self.output_path) if f.endswith(('.png', '.pdf'))]
        for plot_file in sorted(plot_files):
            print(f"  📊 {plot_file}")
        
        print(f"\nTotal files created: {len(plot_files)}")

def main():
    """Main visualization function"""
    visualizer = TrendsVisualizer()
    
    # Create all visualizations
    visualizer.create_all_visualizations(show=False)  # Set to True to display plots

if __name__ == "__main__":
    main()