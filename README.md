# Global CO2 Emissions Analysis: Comprehensive Climate Policy Framework

## Project Overview

This project provides a comprehensive framework for climate policy analysis through five integrated research questions covering data acquisition, predictive modeling, scenario analysis, policy classification, and strategic investment recommendations. The analysis uses World Bank climate and development data (1990-2022) to deliver actionable insights for policymakers, researchers, and investors.

### Business Questions Addressed

1. **Data Foundation**: Comprehensive CO2 emissions data pipeline and exploratory analysis
2. **Economic Impact**: Predictive modeling of GDP-emission relationships with scenario analysis
3. **Technology Assessment**: Electric vehicle adoption impact estimation using Fermi analysis
4. **Policy Intelligence**: Classification framework identifying successful emission reduction strategies
5. **Investment Strategy**: Strategic renewable energy investment prioritization framework

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

## Repository Structure

```
co2_analysis/
├── README.md                                    # Project overview and setup guide
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # Project license
├── .gitignore                                  # Git ignore rules
│
├── scripts/                                    # Analysis scripts (one per question)
│   ├── question_1_data_acquisition_preprocessing.py
│   ├── question_2_predictive_modeling_scenario_analysis.py
│   ├── question_3_ev_sensitivity_analysis.py
│   ├── question_4_emissions_reduction_classifier.py
│   └── question_5_strategic_renewable_investment_analysis.py
│
├── data/                                       # Raw data storage (auto-created)
│
├── outputs/                                    # Analysis results and visualizations
│   ├── processed_world_bank_data.csv              # Master clean dataset
│   ├── processed_world_bank_data.xlsx             # Excel format
│   ├── summary_statistics.json                    # Overall statistics
│   │
│   ├── question_1/                             # Data pipeline outputs
│   │   ├── processed_world_bank_data.csv
│   │   ├── summary_statistics.json
│   │   └── *.png (visualizations)
│   │
│   ├── question_2/                             # Predictive modeling results
│   │   ├── best_model_random_forest.pkl
│   │   ├── gdp_scenario_results.csv
│   │   ├── modeling_results.json
│   │   └── *.png (model performance plots)
│   │
│   ├── question_3/                             # EV impact analysis
│   │   ├── ev_impact_by_country.csv
│   │   ├── ev_global_summary.json
│   │   └── *.png (country rankings, sensitivity)
│   │
│   ├── question_4/                             # Classification analysis
│   │   ├── classification_country_predictions.csv
│   │   ├── feature_importance_rankings.csv
│   │   ├── policy_insights_summary.json
│   │   └── *.png (performance plots)
│   │
│   └── question_5/                             # Investment strategy
│       ├── strategic_recommendations_by_country.csv
│       ├── investment_prioritization_rankings.csv
│       ├── expected_outcomes_analysis.json
│       └── *.png (strategic visualizations)
│
├── docs/                                       # Technical documentation (94KB)
│   ├── README.md                               # Documentation overview
│   ├── question_1_data_pipeline_and_analysis.md
│   ├── question_2_methodology_and_results.md
│   ├── question_3_ev_fermi_analysis.md
│   ├── question_4_classification_and_policy.md
│   └── question_5_strategic_renewable_investment_analysis.md
│
└── METHODOLOGY_VERIFICATION_REPORT.md          # Comprehensive verification report
```

## Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher (tested with Python 3.13)
- **Memory**: 4GB+ RAM recommended for full analysis
- **Storage**: ~500MB for data and outputs
- **Internet**: Stable connection for World Bank API data download
- **Operating System**: Windows, macOS, or Linux

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd co2_analysis
   ```

2. **Create virtual environment (strongly recommended)**
   ```bash
   python -m venv venv

   # Activate environment:
   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All dependencies installed successfully')"
   ```

## Usage Guide

### Quick Start (Full Analysis Pipeline)

```bash
# Run all analyses in sequence (recommended first-time setup)
cd scripts
python question_1_data_acquisition_preprocessing.py
python question_2_predictive_modeling_scenario_analysis.py
python question_3_ev_sensitivity_analysis.py
python question_4_emissions_reduction_classifier.py
python question_5_strategic_renewable_investment_analysis.py
```

### Individual Analysis Modules

#### Question 1: Data Pipeline and EDA
```bash
cd scripts
python question_1_data_acquisition_preprocessing.py
```
**Purpose**: Downloads World Bank data, performs cleaning, and generates foundational dataset  
**Runtime**: 5-10 minutes (includes API download)  
**Outputs**: Clean dataset (CSV/Excel), summary statistics, correlation analysis  
**Key Files**: `outputs/processed_world_bank_data.csv`, visualizations

#### Question 2: Predictive Modeling
```bash
python question_2_predictive_modeling_scenario_analysis.py
```
**Purpose**: GDP-emission relationship modeling with scenario analysis  
**Runtime**: 3-5 minutes  
**Outputs**: Trained models, scenario results, performance metrics  
**Key Files**: `outputs/question_2/gdp_scenario_results.csv`, model pickle files

#### Question 3: EV Impact Analysis
```bash
python question_3_ev_sensitivity_analysis.py
```
**Purpose**: Fermi estimation of global EV adoption impact  
**Runtime**: 1-2 minutes  
**Outputs**: Country-level impact estimates, sensitivity analysis  
**Key Files**: `outputs/question_3/ev_impact_by_country.csv`, rankings

#### Question 4: Policy Classification
```bash
python question_4_emissions_reduction_classifier.py
```
**Purpose**: Identifies characteristics of successful emission-reducing countries  
**Runtime**: 2-3 minutes  
**Outputs**: Country predictions, feature importance, policy insights  
**Key Files**: `outputs/question_4/classification_country_predictions.csv`

#### Question 5: Investment Strategy
```bash
python question_5_strategic_renewable_investment_analysis.py
```
**Purpose**: Strategic renewable energy investment prioritization  
**Runtime**: 2-4 minutes  
**Outputs**: Investment recommendations, country rankings, ROI analysis  
**Key Files**: `outputs/question_5/strategic_recommendations_by_country.csv`

### Execution Notes

- **Dependencies**: Question 2-5 require Question 1 to be run first (for base dataset)
- **Memory Usage**: Peak usage ~2-3GB during model training
- **Output Structure**: Each question creates its own output subdirectory
- **Logging**: Comprehensive console output shows progress and results

## Output Interpretation Guide

### Key Data Files

#### Master Dataset
- **`processed_world_bank_data.csv`**: Clean, analysis-ready dataset with 30+ indicators
- **Structure**: Country-year panel data (1990-2022)
- **Usage**: Foundation for all subsequent analyses

#### Question 1: Exploratory Analysis
- **`summary_statistics.json`**: Comprehensive data quality and correlation metrics
- **Visualizations**: Time series trends, correlation heatmaps, GDP-emission relationships
- **Interpretation**: Baseline understanding of global emission patterns

#### Question 2: Predictive Models
- **`gdp_scenario_results.csv`**: Country-level predictions for 10% GDP increase scenarios
- **`best_model_random_forest.pkl`**: Trained model (R² = 0.996) for future use
- **`modeling_results.json`**: Model performance comparison and feature importance
- **Key Insight**: Average 0.98% CO2 decrease per 10% GDP increase (varies by country)

#### Question 3: EV Impact
- **`ev_impact_by_country.csv`**: Detailed country-level EV adoption impact estimates
- **`ev_global_summary.json`**: Global aggregates: 2,332 Mt CO2 reduction potential
- **Key Insight**: US (499 Mt), China (275 Mt), India (118 Mt) show largest absolute reductions

#### Question 4: Policy Classification
- **`classification_country_predictions.csv`**: Success probability for each country
- **`feature_importance_rankings.csv`**: Key characteristics of successful countries
- **`policy_insights_summary.json`**: Strategic recommendations by country profile
- **Key Insight**: 30/261 countries achieved significant reductions; demographic dynamism most important

#### Question 5: Investment Strategy
- **`strategic_recommendations_by_country.csv`**: Prioritized investment opportunities
- **`expected_outcomes_analysis.json`**: ROI projections and risk assessments
- **Key Insight**: Côte d'Ivoire, Turkmenistan, Algeria top investment priorities

### Visualization Outputs

#### Model Performance Plots
- **Question 2**: Model comparison, residual analysis, feature importance
- **Question 4**: ROC curves, confusion matrices, classification performance
- **Question 5**: Investment scenario comparisons, country rankings

#### Strategic Analysis Charts
- **Question 3**: Country impact rankings, sensitivity analysis ranges
- **Question 5**: Technology portfolio optimization, risk-return profiles

### Business Intelligence Extraction

#### For Policymakers
1. **Country Targeting**: Use Q4 predictions to identify high-success-probability countries
2. **Economic Planning**: Apply Q2 GDP-emission relationships for policy impact assessment
3. **Technology Strategy**: Leverage Q3 EV analysis for transport decarbonization planning

#### For Investors
1. **Market Prioritization**: Q5 investment rankings with quantified ROI expectations
2. **Risk Assessment**: Conservative/moderate/aggressive scenario comparisons
3. **Technology Allocation**: Solar PV (35%), Wind (25%) optimal portfolio weights

#### For Researchers
1. **Model Replication**: Saved pickle files enable scenario testing with new data
2. **Methodology Extension**: Documented assumptions allow parameter sensitivity testing
3. **Data Updates**: Pipeline designed for annual refresh with new World Bank data

## Technical Overview

### Data Foundation
- **Source**: World Bank Open Data API with 30+ development indicators
- **Coverage**: 200+ countries, 1990-2022 (33 years), 6,000+ observations
- **Quality Control**: Systematic filtering, missing value imputation, outlier treatment
- **Integration**: Unified country-year panel structure for analysis

### Analytical Methods
- **Predictive Modeling**: Random Forest regression (R² = 0.996) with 5-fold cross-validation
- **Classification**: SVM with stratified sampling for imbalanced data (F1 = 0.429)
- **Fermi Estimation**: Transparent parameter-based calculation with sensitivity analysis
- **Investment Analysis**: Monte Carlo simulation with multi-criteria decision framework
- **Scenario Modeling**: Ceteris paribus analysis for policy impact assessment

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

## Documentation

This project includes extensive technical documentation (94KB across 5 files):

- **`docs/README.md`**: Documentation overview and standards
- **`docs/question_1_data_pipeline_and_analysis.md`**: Complete data processing methodology
- **`docs/question_2_methodology_and_results.md`**: Predictive modeling technical details
- **`docs/question_3_ev_fermi_analysis.md`**: Fermi estimation framework and assumptions
- **`docs/question_4_classification_and_policy.md`**: Classification methodology and policy insights
- **`docs/question_5_strategic_renewable_investment_analysis.md`**: Investment framework and strategic analysis
- **`METHODOLOGY_VERIFICATION_REPORT.md`**: Comprehensive verification of all analyses

Each document includes:
- Executive summary with key findings
- Detailed methodology and parameter documentation
- Results interpretation and business implications
- Explicit assumptions and limitations
- Future enhancement recommendations

## Dependencies

### Core Libraries
- **pandas** (2.0+): Data manipulation and analysis
- **numpy** (1.24+): Numerical computations
- **wbdata**: World Bank API access
- **requests**: HTTP requests for data download

### Machine Learning
- **scikit-learn** (1.3+): Modeling, preprocessing, and validation
- **scipy**: Statistical functions and optimization
- **joblib**: Model serialization

### Visualization
- **matplotlib** (3.7+): Basic plotting and visualizations
- **seaborn** (0.12+): Statistical visualizations
- **plotly**: Interactive charts (if needed)

### Data Export
- **openpyxl**: Excel file generation
- **json**: Metadata and results export

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
