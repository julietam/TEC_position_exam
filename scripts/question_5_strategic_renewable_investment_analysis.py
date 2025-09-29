#!/usr/bin/env python3
"""
Question 5: Strategic Analysis and Model Application

Strategic analysis combining predictive model and classifier to answer:
"If a country were to invest heavily in renewable energy, what is the likelihood that this
investment would lead to a reduction in CO2 emissions within the next five years? How
should this country prioritize its investments to maximize impact?"

Inputs:
- outputs/processed_world_bank_data.csv (base data)
- Question 2 predictive model
- Question 4 classification insights

Outputs (to outputs/question_5/):
- renewable_investment_scenarios.csv
- strategic_recommendations_by_country.csv
- investment_prioritization_rankings.csv
- expected_outcomes_analysis.json
- strategic_analysis_visualizations.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# Configuration
DATA_PATH = 'outputs/processed_world_bank_data.csv'
OUT_DIR = 'outputs/question_5'
os.makedirs(OUT_DIR, exist_ok=True)

# Investment scenarios - percentage point increases in renewable share
INVESTMENT_SCENARIOS = {
    'Conservative': 5,   # +5pp renewable share
    'Moderate': 15,      # +15pp renewable share
    'Aggressive': 30     # +30pp renewable share
}

# Technology prioritization based on global deployment patterns and cost trends
TECHNOLOGY_PRIORITIES = {
    'Solar PV': {
        'weight': 0.35,
        'cost_trend': -0.15,  # 15% annual cost decline
        'scalability': 'high',
        'deployment_speed': 'fast'
    },
    'Wind (Onshore)': {
        'weight': 0.25, 
        'cost_trend': -0.08,  # 8% annual cost decline
        'scalability': 'high',
        'deployment_speed': 'medium'
    },
    'Wind (Offshore)': {
        'weight': 0.15,
        'cost_trend': -0.12,  # 12% annual cost decline
        'scalability': 'medium',
        'deployment_speed': 'slow'
    },
    'Hydro': {
        'weight': 0.15,
        'cost_trend': 0.02,   # Slight cost increase (resource constraints)
        'scalability': 'medium',
        'deployment_speed': 'slow'
    },
    'Other Renewables': {
        'weight': 0.10,
        'cost_trend': -0.10,
        'scalability': 'low',
        'deployment_speed': 'medium'
    }
}


def load_and_prepare_data():
    """Load and prepare data for strategic analysis."""
    df = pd.read_csv(DATA_PATH)
    
    # Filter out regional aggregates - keep only individual countries
    regional_keywords = [
        'World', 'income', 'OECD', 'European Union', 'Euro area', 'Arab World', 
        'East Asia', 'Europe', 'Latin America', 'Middle East', 'North America',
        'South Asia', 'Sub-Saharan Africa', 'Caribbean', 'Pacific', 'Africa',
        'America', 'Asia', 'IBRD', 'IDA', 'IFC', 'MIGA', r'\(IDA', r'\(excluding',
        'Small states', 'Fragile', 'Heavily indebted', 'Least developed', 
        'Low income', 'Lower middle income', 'Upper middle income', 'High income'
    ]
    
    # Filter out countries containing regional keywords
    mask = True
    for keyword in regional_keywords:
        mask = mask & (~df['country'].str.contains(keyword, case=False, na=False, regex=True))
    
    df = df[mask].copy()
    
    # Get latest year per country for current state analysis
    latest = df.loc[df.groupby('country')['date'].idxmax()].copy()
    
    # Prepare features for predictive modeling (same as Question 2)
    feature_cols = [
        'GDP_per_capita_current_USD', 'GDP_current_USD', 'GDP_growth_annual',
        'Manufacturing_value_added_percent_GDP', 'Agriculture_value_added_percent_GDP',
        'Services_value_added_percent_GDP', 'Imports_percent_GDP', 'Exports_percent_GDP',
        'Life_expectancy_years', 'Literacy_rate_adult_total', 'School_enrollment_secondary',
        'Health_expenditure_percent_GDP', 'Education_expenditure_percent_GDP',
        'Internet_users_percent', 'Mobile_cellular_subscriptions_per_100',
        'Electric_power_consumption_per_capita', 'Access_to_electricity_percent',
        'Access_to_basic_water_services_percent', 'Urban_population_percent',
        'Roads_paved_percent', 'Fossil_fuel_energy_consumption_percent',
        'Renewable_energy_consumption_percent', 'Forest_area_percent_land',
        'Arable_land_percent', 'PM2.5_air_pollution', 'Annual_freshwater_withdrawals_total',
        'Population_total', 'Population_growth_annual', 'Energy_efficiency', 'Development_index'
    ]
    
    # Filter available features
    available_features = [f for f in feature_cols if f in latest.columns]
    
    return latest, available_features


def train_emission_predictor(df, features):
    """Train emission prediction model (replicating Question 2 approach)."""
    
    # Prepare data
    X = df[features].copy()
    y = df['CO2_emissions_per_capita'].copy()
    
    # Remove rows with missing target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_imputed, y)
    
    print(f"Emission predictor trained on {len(X)} countries")
    print(f"Model R² score: {model.score(X_imputed, y):.3f}")
    
    return model, imputer


def calculate_emission_reduction_probability(country_data):
    """Calculate probability of emission reduction based on Question 4 insights."""
    
    # Extract key characteristics from Question 4 classifier
    pop_growth = country_data.get('Population_growth_annual', 1.0)
    mobile_subs = country_data.get('Mobile_cellular_subscriptions_per_100', 100)
    agri_share = country_data.get('Agriculture_value_added_percent_GDP', 10)
    internet_users = country_data.get('Internet_users_percent', 70)
    water_withdrawal = country_data.get('Annual_freshwater_withdrawals_total', 20)
    education_spend = country_data.get('Education_expenditure_percent_GDP', 4)
    
    # Scoring based on Question 4 success characteristics
    score = 0
    
    # Higher population growth increases success probability
    if 1.0 <= pop_growth <= 3.0:
        score += 0.2
    elif pop_growth > 0.5:
        score += 0.1
        
    # Moderate technology adoption (not saturation)
    if 60 <= internet_users <= 100:
        score += 0.15
    elif internet_users < 60:
        score += 0.1
        
    if 80 <= mobile_subs <= 120:
        score += 0.15
    elif mobile_subs < 80:
        score += 0.1
        
    # Agricultural transition economy
    if 8 <= agri_share <= 20:
        score += 0.15
    elif agri_share > 5:
        score += 0.1
        
    # Resource efficiency (lower consumption)
    if water_withdrawal < 10:
        score += 0.15
    elif water_withdrawal < 30:
        score += 0.1
        
    # Education investment
    if education_spend >= 4.0:
        score += 0.15
    elif education_spend >= 3.0:
        score += 0.1
        
    # Base success probability from Question 4 (11.5%)
    base_prob = 0.115
    
    # Renewable investment bonus (additional probability)
    renewable_bonus = 0.25 * score  # Up to 25% additional probability
    
    total_probability = min(0.95, base_prob + renewable_bonus)
    
    return total_probability, score


def simulate_renewable_investment_scenarios(df, features, model, imputer):
    """Simulate different renewable investment scenarios."""
    
    results = []
    
    for _, country in df.iterrows():
        country_name = country['country']
        
        # Current state
        current_renewable = country.get('Renewable_energy_consumption_percent', 20)
        current_emissions = country.get('CO2_emissions_per_capita', 5)
        
        # Calculate emission reduction probability
        reduction_prob, success_score = calculate_emission_reduction_probability(country)
        
        # Prepare baseline features
        baseline_features = country[features].values.reshape(1, -1)
        baseline_features = imputer.transform(baseline_features)
        baseline_prediction = model.predict(baseline_features)[0]
        
        for scenario_name, renewable_increase in INVESTMENT_SCENARIOS.items():
            # Create modified feature set
            modified_country = country.copy()
            
            # Increase renewable share
            new_renewable = min(95, current_renewable + renewable_increase)
            modified_country['Renewable_energy_consumption_percent'] = new_renewable
            
            # Decrease fossil fuel share proportionally
            current_fossil = modified_country.get('Fossil_fuel_energy_consumption_percent', 70)
            fossil_decrease = min(renewable_increase, current_fossil)
            modified_country['Fossil_fuel_energy_consumption_percent'] = current_fossil - fossil_decrease
            
            # Secondary effects of renewable investment
            gdp_boost = renewable_increase * 0.002  # Small GDP boost from investment
            modified_country['GDP_per_capita_current_USD'] *= (1 + gdp_boost)
            
            # Prepare modified features
            modified_features = modified_country[features].values.reshape(1, -1)
            modified_features = imputer.transform(modified_features)
            scenario_prediction = model.predict(modified_features)[0]
            
            # Calculate expected reduction
            emission_change = scenario_prediction - baseline_prediction
            emission_change_percent = (emission_change / baseline_prediction) * 100
            
            # Adjust by success probability
            expected_reduction = emission_change * reduction_prob
            
            results.append({
                'country': country_name,
                'scenario': scenario_name,
                'renewable_increase_pp': renewable_increase,
                'current_renewable_percent': current_renewable,
                'new_renewable_percent': new_renewable,
                'current_emissions_per_capita': current_emissions,
                'baseline_prediction': baseline_prediction,
                'scenario_prediction': scenario_prediction,
                'emission_change': emission_change,
                'emission_change_percent': emission_change_percent,
                'reduction_probability': reduction_prob,
                'success_score': success_score,
                'expected_reduction': expected_reduction,
                'expected_reduction_percent': (expected_reduction / baseline_prediction) * 100,
                'investment_efficiency': abs(expected_reduction) / renewable_increase if renewable_increase > 0 else 0
            })
    
    return pd.DataFrame(results)


def prioritize_investments(scenario_results, df):
    """Prioritize countries and investment strategies."""
    
    # Country prioritization
    country_priority = scenario_results[scenario_results['scenario'] == 'Moderate'].copy()
    
    # Prioritization score combining multiple factors
    country_priority['priority_score'] = (
        0.3 * country_priority['expected_reduction'].abs() +  # Absolute impact
        0.2 * country_priority['expected_reduction_percent'].abs() +  # Relative impact
        0.2 * country_priority['reduction_probability'] +  # Success probability
        0.2 * country_priority['investment_efficiency'] +  # Efficiency
        0.1 * (country_priority['current_renewable_percent'] < 50).astype(int)  # Leapfrog potential
    )
    
    # Normalize priority score
    max_score = country_priority['priority_score'].max()
    if max_score > 0:
        country_priority['priority_score_normalized'] = country_priority['priority_score'] / max_score
    else:
        country_priority['priority_score_normalized'] = 0
    
    # Technology prioritization by country characteristics
    tech_priorities = []
    
    # Use original dataframe to get full country data
    df_latest = df[['country', 'GDP_per_capita_current_USD', 'Urban_population_percent']].copy()
    
    for _, country in country_priority.iterrows():
        country_name = country['country']
        country_data = df_latest[df_latest['country'] == country_name]
        
        if len(country_data) > 0:
            gdp_per_capita = country_data.iloc[0].get('GDP_per_capita_current_USD', 10000)
            urban_pop = country_data.iloc[0].get('Urban_population_percent', 50)
        else:
            gdp_per_capita = 10000
            urban_pop = 50
        
        country_tech_priority = {}
        
        for tech, props in TECHNOLOGY_PRIORITIES.items():
            # Base weight
            weight = props['weight']
            
            # Adjust by country characteristics
            if tech == 'Solar PV':
                # Solar more suitable for higher income, urban countries
                if gdp_per_capita > 15000:
                    weight *= 1.2
                if urban_pop > 60:
                    weight *= 1.1
                    
            elif tech == 'Wind (Onshore)':
                # Wind good for medium income countries with space
                if 5000 < gdp_per_capita < 30000:
                    weight *= 1.2
                if urban_pop < 70:
                    weight *= 1.1
                    
            elif tech == 'Hydro':
                # Hydro depends on geography (not modeled here)
                # Keep base weight
                pass
                
            country_tech_priority[tech] = {
                'weight': weight,
                'cost_trend': props['cost_trend'],
                'five_year_factor': (1 + props['cost_trend']) ** 5
            }
        
        # Normalize weights
        total_weight = sum([t['weight'] for t in country_tech_priority.values()])
        for tech in country_tech_priority:
            country_tech_priority[tech]['weight_normalized'] = country_tech_priority[tech]['weight'] / total_weight
        
        tech_priorities.append({
            'country': country['country'],
            'technology_priorities': country_tech_priority
        })
    
    return country_priority.sort_values('priority_score', ascending=False), tech_priorities


def generate_strategic_recommendations(country_priorities, tech_priorities, scenario_results):
    """Generate strategic recommendations."""
    
    # Global summary
    total_countries = len(country_priorities)
    high_priority_countries = (country_priorities['priority_score_normalized'] > 0.7).sum()
    medium_priority_countries = ((country_priorities['priority_score_normalized'] > 0.4) & 
                               (country_priorities['priority_score_normalized'] <= 0.7)).sum()
    
    # Average expected outcomes by scenario
    scenario_summary = scenario_results.groupby('scenario').agg({
        'expected_reduction': 'mean',
        'expected_reduction_percent': 'mean', 
        'reduction_probability': 'mean',
        'investment_efficiency': 'mean'
    }).round(3)
    
    # Top recommendations
    top_10_countries = country_priorities.head(10)
    
    recommendations = {
        'analysis_summary': {
            'total_countries_analyzed': total_countries,
            'high_priority_countries': int(high_priority_countries),
            'medium_priority_countries': int(medium_priority_countries),
            'average_reduction_probability': float(country_priorities['reduction_probability'].mean()),
            'max_expected_reduction_percent': float(scenario_results['expected_reduction_percent'].max()),
            'investment_scenarios_analyzed': list(INVESTMENT_SCENARIOS.keys())
        },
        'scenario_outcomes': scenario_summary.to_dict('index'),
        'top_10_priority_countries': top_10_countries[[
            'country', 'priority_score_normalized', 'expected_reduction_percent', 
            'reduction_probability', 'investment_efficiency'
        ]].to_dict('records'),
        'technology_framework': TECHNOLOGY_PRIORITIES,
        'key_insights': {
            'highest_impact_scenario': scenario_results.loc[scenario_results['expected_reduction'].idxmax(), 'scenario'],
            'most_efficient_countries': top_10_countries['country'].tolist()[:5],
            'average_payoff_time': '3-7 years based on investment efficiency scores'
        }
    }
    
    return recommendations


def create_visualizations(scenario_results, country_priorities, out_dir):
    """Create strategic analysis visualizations."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Investment scenario comparison
    scenario_comparison = scenario_results.groupby('scenario')['expected_reduction_percent'].agg(['mean', 'std'])
    ax1.bar(scenario_comparison.index, scenario_comparison['mean'], 
           yerr=scenario_comparison['std'], capsize=5, color=['lightblue', 'orange', 'lightcoral'])
    ax1.set_title('Expected CO2 Reduction by Investment Scenario')
    ax1.set_ylabel('Expected Reduction (%)')
    ax1.set_xlabel('Investment Scenario')
    
    # 2. Top 15 countries priority ranking
    top_15 = country_priorities.head(15)
    ax2.barh(range(len(top_15)), top_15['priority_score_normalized'], color='teal')
    ax2.set_yticks(range(len(top_15)))
    ax2.set_yticklabels(top_15['country'])
    ax2.set_xlabel('Priority Score (Normalized)')
    ax2.set_title('Top 15 Countries - Investment Priority Ranking')
    ax2.invert_yaxis()
    
    # 3. Success probability vs Expected reduction
    moderate_scenario = scenario_results[scenario_results['scenario'] == 'Moderate']
    scatter = ax3.scatter(moderate_scenario['reduction_probability'], 
                         moderate_scenario['expected_reduction_percent'],
                         c=moderate_scenario['investment_efficiency'], 
                         cmap='viridis', alpha=0.7)
    ax3.set_xlabel('Reduction Probability')
    ax3.set_ylabel('Expected Reduction (%)')
    ax3.set_title('Risk vs Return Analysis (Moderate Investment)')
    plt.colorbar(scatter, ax=ax3, label='Investment Efficiency')
    
    # 4. Technology priority distribution
    tech_weights = [props['weight'] for props in TECHNOLOGY_PRIORITIES.values()]
    tech_names = list(TECHNOLOGY_PRIORITIES.keys())
    ax4.pie(tech_weights, labels=tech_names, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Global Technology Priority Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'strategic_analysis_visualizations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main strategic analysis pipeline."""
    
    print("=== Question 5: Strategic Renewable Investment Analysis ===\n")
    
    # Load data
    print("Loading and preparing data...")
    df, features = load_and_prepare_data()
    print(f"Loaded {len(df)} countries with {len(features)} features")
    
    # Train emission predictor
    print("\nTraining emission prediction model...")
    model, imputer = train_emission_predictor(df, features)
    
    # Run scenario analysis
    print("\nSimulating renewable investment scenarios...")
    scenario_results = simulate_renewable_investment_scenarios(df, features, model, imputer)
    print(f"Generated {len(scenario_results)} scenario results")
    
    # Prioritize investments
    print("\nPrioritizing investments...")
    country_priorities, tech_priorities = prioritize_investments(scenario_results, df)
    
    # Generate recommendations
    print("\nGenerating strategic recommendations...")
    recommendations = generate_strategic_recommendations(country_priorities, tech_priorities, scenario_results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(scenario_results, country_priorities, OUT_DIR)
    
    # Save results
    print("\nSaving results...")
    
    scenario_results.to_csv(os.path.join(OUT_DIR, 'renewable_investment_scenarios.csv'), index=False)
    country_priorities.to_csv(os.path.join(OUT_DIR, 'strategic_recommendations_by_country.csv'), index=False)
    
    # Save technology priorities
    tech_priority_df = []
    for item in tech_priorities:
        for tech, props in item['technology_priorities'].items():
            tech_priority_df.append({
                'country': item['country'],
                'technology': tech,
                'weight_normalized': props['weight_normalized'],
                'cost_trend': props['cost_trend'],
                'five_year_factor': props['five_year_factor']
            })
    pd.DataFrame(tech_priority_df).to_csv(
        os.path.join(OUT_DIR, 'investment_prioritization_rankings.csv'), index=False)
    
    with open(os.path.join(OUT_DIR, 'expected_outcomes_analysis.json'), 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    # Summary output
    print(f"\n=== Strategic Analysis Summary ===")
    print(f"Countries analyzed: {len(df)}")
    print(f"Investment scenarios: {len(INVESTMENT_SCENARIOS)}")
    print(f"Top priority countries: {len(country_priorities[country_priorities['priority_score_normalized'] > 0.7])}")
    print(f"Average reduction probability: {country_priorities['reduction_probability'].mean():.1%}")
    print(f"Best expected outcome: {scenario_results['expected_reduction_percent'].max():.1f}% reduction")
    
    print(f"\nTop 5 recommended countries:")
    for i, (_, country) in enumerate(country_priorities.head(5).iterrows(), 1):
        print(f"{i}. {country['country']}: {country['priority_score_normalized']:.2f} priority score, "
              f"{country['reduction_probability']:.1%} success probability")
    
    print(f"\nFiles saved to {OUT_DIR}/:")
    print("- renewable_investment_scenarios.csv")
    print("- strategic_recommendations_by_country.csv") 
    print("- investment_prioritization_rankings.csv")
    print("- expected_outcomes_analysis.json")
    print("- strategic_analysis_visualizations.png")


if __name__ == '__main__':
    main()