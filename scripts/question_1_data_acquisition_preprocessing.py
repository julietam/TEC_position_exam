#!/usr/bin/env python3
"""
Question 1: Comprehensive Data Acquisition and Preprocessing for CO2 Emissions Analysis

This script downloads and preprocesses CO2 emissions data along with multiple 
socio-economic and environmental indicators from the World Bank's Climate Change database.

Author: Data Analysis Team
Date: 2025-09-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import wbdata
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class WorldBankDataProcessor:
    """Class to handle World Bank data acquisition and preprocessing"""

    def __init__(self, start_year=1990, end_year=2022):
        """
        Initialize the processor with date range

        Args:
            start_year (int): Starting year for data collection
            end_year (int): Ending year for data collection
        """
        self.start_year = start_year
        self.end_year = end_year
        self.date_range = (datetime(start_year, 1, 1),
                           datetime(end_year, 1, 1))
        self.raw_data = {}
        self.processed_data = None
        self.summary_stats = {}

        # Define indicators of interest
        self.indicators = {
            # CO2 Emissions (using newer Climate Change indicators)
            # Total CO2 emissions excluding LUCF (Mt CO2)
            'CC.CO2.EMSE.EL': 'CO2_emissions_Mt',
            'CC.CO2.EMSE.EN': 'CO2_emissions_energy_Mt',  # CO2 emissions from energy sector
            # CO2 emissions from transportation
            'CC.CO2.EMSE.TR': 'CO2_emissions_transport_Mt',

            # Economic indicators
            'NY.GDP.MKTP.CD': 'GDP_current_USD',
            'NY.GDP.PCAP.CD': 'GDP_per_capita_current_USD',
            'NY.GDP.MKTP.KD.ZG': 'GDP_growth_annual',
            'NE.IMP.GNFS.ZS': 'Imports_percent_GDP',
            'NE.EXP.GNFS.ZS': 'Exports_percent_GDP',

            # Population and demographics
            'SP.POP.TOTL': 'Population_total',
            'SP.POP.GROW': 'Population_growth_annual',
            'SP.URB.TOTL.IN.ZS': 'Urban_population_percent',
            'SP.DYN.LE00.IN': 'Life_expectancy_years',

            # Energy indicators
            'EG.USE.ELEC.KH.PC': 'Electric_power_consumption_per_capita',
            'EG.USE.COMM.FO.ZS': 'Fossil_fuel_energy_consumption_percent',
            'EG.FEC.RNEW.ZS': 'Renewable_energy_consumption_percent',
            'EG.ELC.ACCS.ZS': 'Access_to_electricity_percent',

            # Education
            'SE.ADT.LITR.ZS': 'Literacy_rate_adult_total',
            'SE.XPD.TOTL.GD.ZS': 'Education_expenditure_percent_GDP',
            'SE.SEC.ENRR': 'School_enrollment_secondary',

            # Health
            'SH.XPD.CHEX.GD.ZS': 'Health_expenditure_percent_GDP',
            'SH.H2O.BASW.ZS': 'Access_to_basic_water_services_percent',

            # Infrastructure and technology
            'IT.NET.USER.ZS': 'Internet_users_percent',
            'IS.ROD.PAVE.ZS': 'Roads_paved_percent',
            'IT.CEL.SETS.P2': 'Mobile_cellular_subscriptions_per_100',

            # Environmental
            'AG.LND.FRST.ZS': 'Forest_area_percent_land',
            'AG.LND.ARBL.ZS': 'Arable_land_percent',
            'EN.ATM.PM25.MC.M3': 'PM2.5_air_pollution',
            'ER.H2O.FWTL.K3': 'Annual_freshwater_withdrawals_total',

            # Trade and industry
            'NV.IND.MANF.ZS': 'Manufacturing_value_added_percent_GDP',
            'NV.AGR.TOTL.ZS': 'Agriculture_value_added_percent_GDP',
            'NV.SRV.TOTL.ZS': 'Services_value_added_percent_GDP',
        }

    def download_data(self):
        """Download data from World Bank API"""
        print("Starting data download from World Bank API...")

        for indicator_code, indicator_name in self.indicators.items():
            try:
                print(f"Downloading {indicator_name}...")
                data = wbdata.get_dataframe(
                    {indicator_code: indicator_name},
                    date=self.date_range
                )

                if not data.empty:
                    self.raw_data[indicator_name] = data
                    print(f"Successfully downloaded {indicator_name}")
                else:
                    print(f"No data available for {indicator_name}")

            except Exception as e:
                print(f"Error downloading {indicator_name}: {str(e)}")

        print(
            f"Data download completed. Retrieved {len(self.raw_data)} indicators.")

    def preprocess_data(self):
        """Clean and preprocess the downloaded data"""
        print("\nStarting data preprocessing...")

        if not self.raw_data:
            raise ValueError(
                "No data available. Please run download_data() first.")

        # Combine all indicators into a single DataFrame
        combined_data = []

        for indicator_name, data in self.raw_data.items():
            # Reset index to get country and date as columns
            df = data.reset_index()
            df = df.rename(columns={indicator_name: 'value'})
            df['indicator'] = indicator_name
            combined_data.append(df[['country', 'date', 'indicator', 'value']])

        # Concatenate all data
        long_data = pd.concat(combined_data, ignore_index=True)

        # Pivot to wide format
        wide_data = long_data.pivot_table(
            index=['country', 'date'],
            columns='indicator',
            values='value'
        ).reset_index()

        # Clean column names
        wide_data.columns.name = None

        print(f"Initial dataset shape: {wide_data.shape}")

        # Remove countries with too much missing data (>70% missing)
        missing_threshold = 0.7
        missing_by_country = wide_data.groupby('country').apply(
            lambda x: x.isnull().sum().sum() / (len(x) * len(self.indicators))
        )

        countries_to_keep = missing_by_country[missing_by_country <
                                               missing_threshold].index
        wide_data = wide_data[wide_data['country'].isin(countries_to_keep)]

        print(
            f"After removing countries with >{missing_threshold*100}% missing data: {wide_data.shape}")

        # Handle missing values
        wide_data = self._handle_missing_values(wide_data)

        # Handle outliers
        wide_data = self._handle_outliers(wide_data)

        # Feature engineering
        wide_data = self._engineer_features(wide_data)

        self.processed_data = wide_data
        print(f"Final processed dataset shape: {wide_data.shape}")

    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        print("Handling missing values...")

        # Count missing values before processing
        initial_missing = data.isnull().sum().sum()
        print(f"Initial missing values: {initial_missing}")

        # Get numeric columns (excluding date)
        numeric_columns = data.select_dtypes(
            include=[np.number]).columns.tolist()
        if 'date' in numeric_columns:
            numeric_columns.remove('date')

        # Step 1: Forward/backward fill within each country's time series
        for country in data['country'].unique():
            country_mask = data['country'] == country
            country_data = data.loc[country_mask].sort_values('date')

            # Only fill missing values, preserve existing ones
            for col in numeric_columns:
                if country_data[col].isnull().any():
                    filled_values = country_data[col].ffill().bfill()
                    # Only update null values
                    null_mask = data.loc[country_mask, col].isnull()
                    data.loc[country_mask & null_mask,
                             col] = filled_values[null_mask]

        # Step 2: For remaining missing values, use yearly median (only for null values)
        for col in numeric_columns:
            if data[col].isnull().any():
                yearly_medians = data.groupby('date')[col].median()
                for year in yearly_medians.index:
                    if not pd.isna(yearly_medians[year]):
                        year_mask = data['date'] == year
                        missing_mask = data[col].isnull()
                        combined_mask = year_mask & missing_mask
                        data.loc[combined_mask, col] = yearly_medians[year]

        # Step 3: Final pass - global median only for remaining nulls
        remaining_missing = data[numeric_columns].isnull().sum()
        for col in numeric_columns:
            if remaining_missing[col] > 0:
                global_median = data[col].median()
                if not pd.isna(global_median):
                    data[col] = data[col].fillna(global_median)

        final_missing = data.isnull().sum().sum()
        print(f"Final missing values: {final_missing}")
        print(f"Missing values filled: {initial_missing - final_missing}")

        return data

    def _handle_outliers(self, data):
        """Handle outliers using IQR method"""
        print("Handling outliers...")

        numeric_columns = data.select_dtypes(
            include=[np.number]).columns.tolist()
        numeric_columns.remove('date') if 'date' in numeric_columns else None

        outlier_counts = {}

        for col in numeric_columns:
            # Skip outlier handling for CO2 emissions to preserve real values
            if 'CO2_emissions' in col:
                outlier_counts[col] = 0
                print(
                    f"Skipping outlier handling for {col} to preserve emission data")
                continue

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR  # Using 3 IQR instead of 1.5 to be less aggressive
            upper_bound = Q3 + 3 * IQR

            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outliers.sum()
            outlier_counts[col] = outlier_count

            if outlier_count > 0:
                # Cap outliers instead of removing them
                data[col] = np.clip(data[col], lower_bound, upper_bound)
                print(f"Capped {outlier_count} outliers in {col}")

        self.outlier_summary = outlier_counts
        return data

    def _engineer_features(self, data):
        """Create additional features"""
        print("Engineering additional features...")

        # Economic features
        if 'GDP_current_USD' in data.columns and 'Population_total' in data.columns:
            data['GDP_per_capita_calculated'] = data['GDP_current_USD'] / \
                data['Population_total']

        # CO2 per capita calculation (Mt CO2 -> tonnes per capita)
        if 'CO2_emissions_Mt' in data.columns and 'Population_total' in data.columns:
            # Convert Mt CO2 to tonnes per capita: 1 Mt = 1,000,000 tonnes
            data['CO2_emissions_per_capita'] = (
                data['CO2_emissions_Mt'] * 1_000_000) / data['Population_total']

        # CO2 efficiency metrics
        if 'CO2_emissions_Mt' in data.columns and 'GDP_current_USD' in data.columns:
            data['CO2_intensity_per_GDP'] = data['CO2_emissions_Mt'] / \
                (data['GDP_current_USD'] / 1e12)  # Mt CO2 per trillion USD

        # Energy efficiency
        if 'Electric_power_consumption_per_capita' in data.columns and 'GDP_per_capita_current_USD' in data.columns:
            data['Energy_efficiency'] = data['GDP_per_capita_current_USD'] / \
                data['Electric_power_consumption_per_capita']

        # Development index (simple combination)
        development_indicators = ['GDP_per_capita_current_USD',
                                  'Life_expectancy_years', 'Literacy_rate_adult_total']
        available_dev_indicators = [
            col for col in development_indicators if col in data.columns]

        if len(available_dev_indicators) >= 2:
            # Normalize each indicator to 0-1 scale
            scaler = StandardScaler()
            normalized_dev = pd.DataFrame(
                scaler.fit_transform(data[available_dev_indicators]),
                columns=available_dev_indicators,
                index=data.index
            )
            data['Development_index'] = normalized_dev.mean(axis=1)

        return data

    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\nGenerating summary statistics...")

        if self.processed_data is None:
            raise ValueError(
                "No processed data available. Run preprocess_data() first.")

        data = self.processed_data
        numeric_data = data.select_dtypes(include=[np.number])

        # Basic statistics
        self.summary_stats['basic_stats'] = numeric_data.describe()

        # Missing value summary
        self.summary_stats['missing_values'] = data.isnull(
        ).sum().sort_values(ascending=False)

        # Correlation matrix for key indicators
        key_indicators = [
            'CO2_emissions_per_capita', 'GDP_per_capita_current_USD',
            'Electric_power_consumption_per_capita', 'Urban_population_percent',
            'Life_expectancy_years', 'Education_expenditure_percent_GDP'
        ]
        available_key_indicators = [
            col for col in key_indicators if col in numeric_data.columns]

        if len(available_key_indicators) >= 2:
            self.summary_stats['correlations'] = numeric_data[available_key_indicators].corr(
            )

        # Country and time coverage
        self.summary_stats['country_count'] = data['country'].nunique()

        # Handle date conversion
        if data['date'].dtype == 'object':  # String dates
            dates = pd.to_datetime(data['date'])
            self.summary_stats['year_range'] = (
                dates.min().year, dates.max().year)
        else:  # Already datetime
            self.summary_stats['year_range'] = (
                data['date'].min().year, data['date'].max().year)

        self.summary_stats['total_observations'] = len(data)

        # Top CO2 emitters
        if 'CO2_emissions_Mt' in data.columns:
            latest_year = data['date'].max()
            latest_data = data[data['date'] == latest_year]
            self.summary_stats['top_co2_emitters'] = latest_data.nlargest(
                10, 'CO2_emissions_Mt')[['country', 'CO2_emissions_Mt']]
        elif 'CO2_emissions_per_capita' in data.columns:
            latest_year = data['date'].max()
            latest_data = data[data['date'] == latest_year]
            self.summary_stats['top_co2_emitters_per_capita'] = latest_data.nlargest(
                10, 'CO2_emissions_per_capita')[['country', 'CO2_emissions_per_capita']]

    def create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")

        if self.processed_data is None:
            raise ValueError("No processed data available.")

        # Set up matplotlib
        plt.style.use('seaborn-v0_8')

        # 1. CO2 emissions over time by region/income group
        if 'CO2_emissions_per_capita' in self.processed_data.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get latest year for finding top countries
            latest_year = self.processed_data['date'].max()
            latest_data = self.processed_data[self.processed_data['date']
                                              == latest_year]

            # Get top CO2 emitting countries per capita
            top_countries = latest_data.nlargest(10, 'CO2_emissions_per_capita')[
                'country'].tolist()
            print(f"Top CO2 emitters per capita: {top_countries[:5]}")
            print("Note: These are small territories/city-states with extreme per capita values due to small populations")

            for country in top_countries[:5]:  # Show top 5 for clarity
                country_data = self.processed_data[self.processed_data['country'] == country].copy(
                )

                if len(country_data) > 0:
                    # Convert dates to datetime for proper plotting
                    if country_data['date'].dtype in ['int64', 'int32']:
                        country_data['date'] = pd.to_datetime(
                            country_data['date'], format='%Y')
                    elif country_data['date'].dtype == 'object':
                        country_data['date'] = pd.to_datetime(
                            country_data['date'])

                    # Sort by date for proper line plotting
                    country_data = country_data.sort_values('date')

                    # Plot CO2 per capita data
                    ax.plot(country_data['date'], country_data['CO2_emissions_per_capita'],
                            marker='o', label=country, linewidth=2, markersize=4)

            # Set plot labels and formatting
            ax.set_title('CO2 Emissions Per Capita Over Time - Top Countries',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('CO2 Emissions Per Capita (metric tons)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Format x-axis to show years properly
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'co2_emissions_time_series.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Correlation heatmap
        numeric_data = self.processed_data.select_dtypes(include=[np.number])
        correlation_cols = [col for col in numeric_data.columns if col not in [
            'date'] and numeric_data[col].var() > 0]

        if len(correlation_cols) > 1:
            corr_matrix = numeric_data[correlation_cols].corr()

            fig, ax = plt.subplots(figsize=(15, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f', cbar_kws={"shrink": .8})
            ax.set_title('Correlation Matrix of Key Indicators',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(
                output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Scatter plot (CO2 vs GDP)
        if 'GDP_per_capita_current_USD' in self.processed_data.columns and 'CO2_emissions_per_capita' in self.processed_data.columns:
            # Get latest year data
            latest_year = self.processed_data['date'].max()
            latest_data = self.processed_data[self.processed_data['date']
                                              == latest_year]

            print(
                f"Creating scatter plot with {len(latest_data)} countries for year {latest_year}")

            fig, ax = plt.subplots(figsize=(10, 8))

            # Filter out missing data
            plot_data = latest_data.dropna(
                subset=['GDP_per_capita_current_USD', 'CO2_emissions_per_capita'])

            if len(plot_data) > 0:
                # CO2 vs GDP scatter plot
                scatter = ax.scatter(plot_data['GDP_per_capita_current_USD'],
                                     plot_data['CO2_emissions_per_capita'],
                                     alpha=0.6, s=60, c='steelblue', edgecolors='darkblue')

                ax.set_xlabel('GDP Per Capita (USD)')
                ax.set_ylabel('CO2 Emissions Per Capita (metric tons)')
                ax.set_title(f'CO2 Emissions vs GDP Per Capita ({latest_year})',
                             fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Add trend line if enough data points
                if len(plot_data) > 5:
                    x = plot_data['GDP_per_capita_current_USD']
                    y = plot_data['CO2_emissions_per_capita']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
                            label=f'Trend line (R² = {np.corrcoef(x, y)[0, 1]**2:.3f})')
                    ax.legend()

                print(
                    f"Scatter plot created with {len(plot_data)} data points")
            else:
                ax.text(0.5, 0.5, 'No data available for scatter plot',
                        ha='center', va='center', transform=ax.transAxes)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'co2_vs_gdp_scatter.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        print("Visualizations saved to output directory")

    def save_data(self, output_dir):
        """Save processed data and summary statistics"""
        print("\nSaving processed data and statistics...")

        # Save processed data
        if self.processed_data is not None:
            self.processed_data.to_csv(os.path.join(
                output_dir, 'processed_world_bank_data.csv'), index=False)
            self.processed_data.to_excel(os.path.join(
                output_dir, 'processed_world_bank_data.xlsx'), index=False)

        # Save summary statistics
        summary_file = os.path.join(output_dir, 'summary_statistics.json')

        # Convert DataFrames to JSON-serializable format
        json_stats = {}
        for key, value in self.summary_stats.items():
            if isinstance(value, pd.DataFrame):
                json_stats[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                json_stats[key] = value.to_dict()
            else:
                json_stats[key] = value

        with open(summary_file, 'w') as f:
            json.dump(json_stats, f, indent=2, default=str)

        print("Data and statistics saved successfully")


def main():
    """Main execution function"""
    print("=" * 60)
    print("WORLD BANK CO2 EMISSIONS DATA ANALYSIS")
    print("=" * 60)

    # Initialize processor
    processor = WorldBankDataProcessor(start_year=1990, end_year=2022)

    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Execute pipeline
        processor.download_data()
        processor.preprocess_data()
        processor.generate_summary_statistics()
        processor.create_visualizations(output_dir)
        processor.save_data(output_dir)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(
            f"Countries analyzed: {processor.summary_stats['country_count']}")
        print(
            f"Time period: {processor.summary_stats['year_range'][0]}-{processor.summary_stats['year_range'][1]}")
        print(
            f"Total observations: {processor.summary_stats['total_observations']}")
        print(f"Outputs saved to: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
