# World Bank CO2 Emissions Data Analysis

## Overview

This project provides a comprehensive solution for downloading, preprocessing, and analyzing CO2 emissions data along with a wide range of socio-economic and environmental indicators from the World Bank's Climate Change database. The analysis aims to understand the relationships between CO2 emissions and various development factors across countries and time periods.

## Dataset Description

The project downloads and integrates **30+ indicators** across multiple domains:

### Core CO2 Emissions Indicators

- CO2 emissions (kt)
- CO2 emissions per capita
- CO2 emissions per GDP unit

### Economic Indicators

- GDP (current USD)
- GDP per capita
- GDP growth rate
- Import/Export percentages
- Manufacturing, Agriculture, Services value-added

### Energy Indicators

- Electric power consumption per capita
- Fossil fuel energy consumption percentage
- Renewable energy consumption percentage
- Access to electricity percentage

### Social Development Indicators

- Population metrics (total, growth, urbanization)
- Life expectancy
- Literacy rates
- Education expenditure
- Health expenditure
- Access to basic services

### Environmental & Infrastructure

- Forest area percentage
- Air pollution (PM2.5)
- Water resources
- Internet penetration
- Mobile subscriptions
- Road infrastructure

## Project Structure

```
co2_analysis/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/                              # Raw data storage (created automatically)
├── scripts/
│   └── data_acquisition_preprocessing.py  # Main analysis script
├── outputs/                           # Generated outputs
│   ├── processed_world_bank_data.csv  # Clean dataset
│   ├── processed_world_bank_data.xlsx # Excel format
│   ├── summary_statistics.json       # Statistical summaries
│   ├── co2_emissions_time_series.png  # Visualization 1
│   ├── correlation_heatmap.png       # Visualization 2
│   └── co2_vs_gdp_scatter.png        # Visualization 3
└── docs/
    └── data_summary.md               # Detailed analysis results
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for data download

### Installation Steps

1. **Clone or download this project**

   ```bash
   cd co2_analysis
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Analysis

1. **Navigate to the scripts directory**

   ```bash
   cd scripts
   ```

2. **Execute the main script**
   ```bash
   python data_acquisition_preprocessing.py
   ```

### Expected Output

The script will:

1. Download data from World Bank API (this may take 5-10 minutes)
2. Clean and preprocess the data
3. Generate summary statistics
4. Create visualizations
5. Save all outputs to the `outputs/` directory

### Progress Monitoring

The script provides detailed console output showing:

- Download progress for each indicator
- Data preprocessing steps
- Final dataset statistics
- File save confirmations

## Methodology

### Data Acquisition

- **Source**: World Bank Open Data API
- **Time Period**: 1990-2022 (33 years)
- **Coverage**: All available countries
- **Method**: Uses the `wbdata` Python library for API access

### Data Preprocessing Pipeline

1. **Data Integration**

   - Combines all indicators into a unified dataset
   - Restructures from long to wide format
   - Maintains country-year structure

2. **Quality Control**

   - Removes countries with >70% missing data
   - Ensures temporal consistency

3. **Missing Value Handling**

   - Forward/backward fill within countries (time series interpolation)
   - Yearly median imputation for cross-country gaps
   - Global median as final fallback

4. **Outlier Treatment**

   - Uses Inter-Quartile Range (IQR) method
   - Conservative approach: 3×IQR threshold
   - Caps outliers rather than removing them

5. **Feature Engineering**
   - GDP per capita calculations
   - CO2 intensity metrics
   - Energy efficiency ratios
   - Composite development index

### Statistical Analysis

- Descriptive statistics for all variables
- Correlation analysis between key indicators
- Identification of top CO2 emitters
- Time series trends visualization

## Key Outputs

### 1. Processed Dataset

- **Format**: CSV and Excel
- **Structure**: Country-year panel data
- **Variables**: 30+ cleaned indicators + engineered features
- **Coverage**: Countries with sufficient data quality

### 2. Summary Statistics

- Basic descriptive statistics (mean, std, min, max, percentiles)
- Missing value reports
- Correlation matrices
- Top emitter rankings

### 3. Visualizations

- **Time Series**: CO2 emissions trends for top emitting countries
- **Correlation Heatmap**: Relationships between all indicators
- **Scatter Plot**: CO2 vs GDP relationship with trend line

### 4. Metadata

- Data quality reports
- Processing logs
- Variable definitions

## Key Findings Preview

The analysis typically reveals:

1. **Strong Correlations**

   - GDP per capita ↔ CO2 emissions per capita
   - Energy consumption ↔ Economic development
   - Urbanization ↔ Infrastructure development

2. **Regional Patterns**

   - Developed countries: High GDP, high emissions per capita
   - Developing countries: Growing emissions with economic growth
   - Island nations: Unique energy profiles

3. **Temporal Trends**
   - Decoupling in some developed countries (GDP growth > emission growth)
   - Rapid industrialization impacts in emerging economies

## Customization Options

### Modify Time Period

```python
processor = WorldBankDataProcessor(start_year=2000, end_year=2020)
```

### Add New Indicators

Add to the `indicators` dictionary in the `WorldBankDataProcessor` class:

```python
'NEW.INDICATOR.CODE': 'descriptive_name'
```

### Adjust Data Quality Thresholds

Modify the `missing_threshold` variable in `preprocess_data()` method.

## Dependencies

### Core Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **wbdata**: World Bank API access
- **requests**: HTTP requests

### Analysis & Visualization

- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive plots
- **scikit-learn**: Data preprocessing tools
- **scipy**: Statistical functions

### Data Export

- **openpyxl**: Excel file generation
- **json**: Metadata export

## Troubleshooting

### Common Issues

1. **Network Errors**

   - Ensure stable internet connection
   - World Bank API may have rate limits
   - Script includes automatic retry logic

2. **Package Installation Issues**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Memory Issues**

   - Large datasets may require 2GB+ RAM
   - Reduce time period if experiencing issues

4. **Missing Data Warnings**
   - Normal for some indicators/countries
   - Script handles this automatically

### Performance Tips

- First run takes longer (data download)
- Subsequent runs can reuse downloaded data
- Consider using smaller time windows for testing

## Further Reading

- [World Bank Open Data](https://data.worldbank.org/)
- [Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/)
- [wbdata Documentation](https://wbdata.readthedocs.io/)

## Contributing

To extend this analysis:

1. Add new indicators by updating the `indicators` dictionary
2. Implement additional preprocessing methods
3. Create new visualization functions
4. Add statistical tests or modeling components

## License

This project is provided for educational and research purposes. World Bank data is subject to their terms of use.

---

**Authors**: Data Analysis Team  
**Last Updated**: September 29, 2025  
**Version**: 1.0
