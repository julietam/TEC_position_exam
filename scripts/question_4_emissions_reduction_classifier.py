#!/usr/bin/env python3
"""
Question 4: Classification and Policy Implications

Build a classifier to identify countries likely to achieve significant CO2 emissions reduction
in the next decade. Analyze characteristics of successful emission-reducing countries and
provide policy recommendations.

Inputs:
- outputs/processed_world_bank_data.csv (from Question 1 pipeline)

Outputs (to outputs/question_4/):
- emissions_reduction_targets.csv
- classification_performance_metrics.csv
- feature_importance_rankings.csv
- classification_country_predictions.csv
- classifier_performance_plots.png
- feature_importance_plot.png
- policy_insights_clusters.png

Business Question:
"What are the common characteristics of countries that successfully reduce emissions,
and how can policymakers in other nations apply these insights?"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.impute import SimpleImputer
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# Configuration
DATA_PATH = 'outputs/processed_world_bank_data.csv'
OUT_DIR = 'outputs/question_4'
os.makedirs(OUT_DIR, exist_ok=True)

# Emissions reduction thresholds for classification
SIGNIFICANT_REDUCTION_THRESHOLD = -2.0  # % per year over analysis period
ANALYSIS_PERIOD_START = 2010  # Start period for trend analysis
ANALYSIS_PERIOD_END = 2022    # End period for trend analysis


def calculate_emissions_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CO2 emissions trends for target variable creation."""
    
    # Filter to analysis period
    period_data = df[(df['date'] >= ANALYSIS_PERIOD_START) & 
                     (df['date'] <= ANALYSIS_PERIOD_END)].copy()
    
    # Calculate per capita emissions trends by country
    country_trends = []
    
    for country in period_data['country'].unique():
        country_data = period_data[period_data['country'] == country].copy()
        
        if len(country_data) < 5:  # Need minimum data points
            continue
            
        # Sort by year
        country_data = country_data.sort_values('date')
        
        # Calculate linear trend (% change per year)
        years = country_data['date'].values
        emissions = country_data['CO2_emissions_per_capita'].values
        
        if len(emissions) > 0 and not np.all(np.isnan(emissions)):
            # Remove NaN values
            valid_idx = ~np.isnan(emissions)
            if np.sum(valid_idx) >= 3:  # Need at least 3 valid points
                years_valid = years[valid_idx]
                emissions_valid = emissions[valid_idx]
                
                # Calculate compound annual growth rate (CAGR)
                if len(years_valid) >= 2 and emissions_valid[0] > 0:
                    start_emissions = emissions_valid[0]
                    end_emissions = emissions_valid[-1]
                    years_span = years_valid[-1] - years_valid[0]
                    
                    if years_span > 0 and end_emissions > 0:
                        cagr = ((end_emissions / start_emissions) ** (1/years_span) - 1) * 100
                        
                        # Get most recent year's data for features
                        latest_data = country_data.iloc[-1].copy()
                        
                        country_trends.append({
                            'country': country,
                            'emissions_trend_percent_per_year': cagr,
                            'significant_reduction': 1 if cagr <= SIGNIFICANT_REDUCTION_THRESHOLD else 0,
                            'start_year': int(years_valid[0]),
                            'end_year': int(years_valid[-1]),
                            'start_emissions': start_emissions,
                            'end_emissions': end_emissions,
                            'data_points': len(years_valid),
                            **{col: latest_data[col] for col in latest_data.index 
                               if col not in ['country', 'date']}
                        })
    
    return pd.DataFrame(country_trends)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features for classification."""
    
    # Define feature categories
    economic_features = [
        'GDP_per_capita_current_USD', 'GDP_current_USD', 'GDP_growth_annual',
        'Manufacturing_value_added_percent_GDP', 'Agriculture_value_added_percent_GDP',
        'Services_value_added_percent_GDP', 'Imports_percent_GDP', 'Exports_percent_GDP'
    ]
    
    social_features = [
        'Life_expectancy_years', 'Literacy_rate_adult_total', 'School_enrollment_secondary',
        'Health_expenditure_percent_GDP', 'Education_expenditure_percent_GDP',
        'Internet_users_percent', 'Mobile_cellular_subscriptions_per_100'
    ]
    
    infrastructure_features = [
        'Electric_power_consumption_per_capita', 'Access_to_electricity_percent',
        'Access_to_basic_water_services_percent', 'Urban_population_percent',
        'Roads_paved_percent'
    ]
    
    environmental_features = [
        'Fossil_fuel_energy_consumption_percent', 'Renewable_energy_consumption_percent',
        'Forest_area_percent_land', 'Arable_land_percent', 'PM2.5_air_pollution',
        'Annual_freshwater_withdrawals_total', 'Population_total', 'Population_growth_annual'
    ]
    
    # Engineered features
    engineered_features = [
        'Energy_efficiency', 'Development_index'
    ]
    
    all_features = (economic_features + social_features + infrastructure_features + 
                   environmental_features + engineered_features)
    
    # Filter features that exist in the data
    available_features = [f for f in all_features if f in df.columns]
    
    # Prepare feature matrix
    X = df[available_features].copy()
    y = df['significant_reduction'].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    feature_info = {
        'economic': [f for f in economic_features if f in available_features],
        'social': [f for f in social_features if f in available_features],
        'infrastructure': [f for f in infrastructure_features if f in available_features],
        'environmental': [f for f in environmental_features if f in available_features],
        'engineered': [f for f in engineered_features if f in available_features]
    }
    
    return X_imputed, y, feature_info


def train_classifiers(X, y, feature_names):
    """Train multiple classification models and compare performance."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=1000
        ),
        'SVM': SVC(
            random_state=42, class_weight='balanced', probability=True
        )
    }
    
    # Train and evaluate models
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for SVM and Logistic Regression
        if name in ['SVM', 'Logistic Regression']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # Train model
        model.fit(X_train_use, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_use)
        y_prob = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_use, y_train, cv=cv, scoring='f1')
        
        # AUC if probabilities available
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'scaler': scaler if name in ['SVM', 'Logistic Regression'] else None
        }
        
        print(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV F1: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    return results, X_train, X_test, y_train, y_test


def analyze_feature_importance(results, X, feature_names, feature_info):
    """Analyze feature importance from best model."""
    
    # Find best model based on F1 score
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['f1']:.3f})")
    
    # Get feature importance - try multiple models if best doesn't have it
    importance_df = None
    
    # Try the best model first
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print(f"Using feature importance from {best_model_name}")
        
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_[0])
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print(f"Using coefficients from {best_model_name}")
        
    else:
        # Try to find a model with feature importance
        print(f"{best_model_name} doesn't provide feature importance, trying other models...")
        
        for model_name, model_result in results.items():
            model = model_result['model']
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                print(f"Using feature importance from {model_name} instead")
                break
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                print(f"Using coefficients from {model_name} instead")
                break
    
    if importance_df is not None:
        # Add feature categories
        def get_feature_category(feature):
            for category, features in feature_info.items():
                if feature in features:
                    return category
            return 'other'
        
        importance_df['category'] = importance_df['feature'].apply(get_feature_category)
    
    return importance_df, best_model_name


def create_visualizations(results, importance_df, out_dir):
    """Create performance and feature importance visualizations."""
    
    # 1. Model Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance metrics
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    performance_data = []
    for model in models:
        for metric in metrics:
            if metric in results[model]:
                performance_data.append({
                    'Model': model,
                    'Metric': metric.title(),
                    'Score': results[model][metric]
                })
    
    perf_df = pd.DataFrame(performance_data)
    sns.barplot(data=perf_df, x='Model', y='Score', hue='Metric', ax=ax1)
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # ROC Curves
    for name, result in results.items():
        if result['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
            auc = result['auc']
            ax2.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    
    # Confusion Matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_result = results[best_model_name]
    
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_model_name}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Feature Importance
    if importance_df is not None:
        top_features = importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', 
                   hue='category', ax=ax4)
        ax4.set_title('Top 15 Feature Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'classifier_performance_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Feature Importance Plot
    if importance_df is not None:
        plt.figure(figsize=(12, 10))
        
        # Color by category
        category_colors = plt.cm.Set3(np.linspace(0, 1, len(importance_df['category'].unique())))
        color_map = dict(zip(importance_df['category'].unique(), category_colors))
        colors = [color_map[cat] for cat in importance_df.head(20)['category']]
        
        plt.barh(range(20), importance_df.head(20)['importance'], color=colors)
        plt.yticks(range(20), importance_df.head(20)['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Features for Emissions Reduction Classification')
        plt.gca().invert_yaxis()
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cat], label=cat.title()) 
                          for cat in color_map.keys()]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'feature_importance_plot.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def generate_policy_insights(df_with_targets, importance_df, results):
    """Generate policy insights based on classification results."""
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"\n=== Policy Insights Based on {best_model_name} ===\n")
    
    # Analyze characteristics of successful vs unsuccessful countries
    successful = df_with_targets[df_with_targets['significant_reduction'] == 1]
    unsuccessful = df_with_targets[df_with_targets['significant_reduction'] == 0]
    
    print(f"Countries with Significant Emissions Reduction: {len(successful)}")
    print(f"Countries without Significant Reduction: {len(unsuccessful)}")
    
    # Top distinguishing features
    if importance_df is not None:
        print(f"\nTop 10 Distinguishing Characteristics:")
        for i, row in importance_df.head(10).iterrows():
            feature = row['feature']
            category = row['category']
            
            if feature in df_with_targets.columns:
                successful_mean = successful[feature].mean()
                unsuccessful_mean = unsuccessful[feature].mean()
                
                print(f"{i+1}. {feature} ({category.title()})")
                print(f"   Successful countries: {successful_mean:.2f}")
                print(f"   Other countries: {unsuccessful_mean:.2f}")
                print(f"   Difference: {successful_mean - unsuccessful_mean:.2f}")
                print()
    
    # Country examples
    print("Examples of Successful Countries:")
    successful_examples = successful.nlargest(5, 'emissions_trend_percent_per_year')[
        ['country', 'emissions_trend_percent_per_year', 'start_emissions', 'end_emissions']
    ]
    for _, country in successful_examples.iterrows():
        print(f"- {country['country']}: {country['emissions_trend_percent_per_year']:.1f}% per year "
              f"({country['start_emissions']:.1f} → {country['end_emissions']:.1f} tonnes per capita)")
    
    return {
        'successful_countries': len(successful),
        'total_countries': len(df_with_targets),
        'success_rate': len(successful) / len(df_with_targets) * 100,
        'top_features': importance_df.head(10).to_dict('records') if importance_df is not None else [],
        'successful_examples': successful_examples.to_dict('records')
    }


def main():
    """Main analysis pipeline."""
    
    print("=== Question 4: Emissions Reduction Classification Analysis ===\n")
    
    # Load data
    print("Loading and preparing data...")
    df = pd.read_csv(DATA_PATH)
    
    # Calculate emissions trends and create target variable
    print("Calculating emissions trends...")
    df_with_targets = calculate_emissions_trends(df)
    
    if len(df_with_targets) < 20:
        print(f"Error: Only {len(df_with_targets)} countries with sufficient data.")
        return
    
    print(f"Analysis dataset: {len(df_with_targets)} countries")
    print(f"Significant reduction threshold: {SIGNIFICANT_REDUCTION_THRESHOLD}% per year")
    print(f"Analysis period: {ANALYSIS_PERIOD_START}-{ANALYSIS_PERIOD_END}")
    
    # Check class balance
    class_counts = df_with_targets['significant_reduction'].value_counts()
    print(f"\nClass distribution:")
    print(f"Significant reduction (1): {class_counts.get(1, 0)} countries")
    print(f"No significant reduction (0): {class_counts.get(0, 0)} countries")
    
    # Save targets dataset
    df_with_targets.to_csv(
        os.path.join(OUT_DIR, 'emissions_reduction_targets.csv'), 
        index=False
    )
    
    # Prepare features
    print("\nPreparing features for classification...")
    X, y, feature_info = prepare_features(df_with_targets)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature categories: {list(feature_info.keys())}")
    
    # Train classifiers
    print("\nTraining classification models...")
    results, X_train, X_test, y_train, y_test = train_classifiers(X, y, X.columns)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_df, best_model_name = analyze_feature_importance(
        results, X, X.columns, feature_info
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, importance_df, OUT_DIR)
    
    # Generate policy insights
    print("\nGenerating policy insights...")
    policy_insights = generate_policy_insights(df_with_targets, importance_df, results)
    
    # Save results
    print("\nSaving results...")
    
    # Performance metrics
    performance_data = []
    for name, result in results.items():
        performance_data.append({
            'model': name,
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'auc': result['auc'],
            'cv_f1_mean': result['cv_f1_mean'],
            'cv_f1_std': result['cv_f1_std']
        })
    
    pd.DataFrame(performance_data).to_csv(
        os.path.join(OUT_DIR, 'classification_performance_metrics.csv'),
        index=False
    )
    
    # Feature importance
    if importance_df is not None:
        importance_df.to_csv(
            os.path.join(OUT_DIR, 'feature_importance_rankings.csv'),
            index=False
        )
    
    # Country predictions
    best_model = results[best_model_name]['model']
    if best_model_name in ['SVM', 'Logistic Regression']:
        scaler = results[best_model_name]['scaler']
        X_scaled = scaler.transform(X)
        predictions = best_model.predict(X_scaled)
        probabilities = best_model.predict_proba(X_scaled)[:, 1]
    else:
        predictions = best_model.predict(X)
        probabilities = best_model.predict_proba(X)[:, 1]
    
    country_predictions = pd.DataFrame({
        'country': df_with_targets['country'],
        'actual_reduction': df_with_targets['significant_reduction'],
        'predicted_reduction': predictions,
        'reduction_probability': probabilities,
        'emissions_trend_percent_per_year': df_with_targets['emissions_trend_percent_per_year']
    })
    
    country_predictions.to_csv(
        os.path.join(OUT_DIR, 'classification_country_predictions.csv'),
        index=False
    )
    
    # Policy insights summary
    with open(os.path.join(OUT_DIR, 'policy_insights_summary.json'), 'w') as f:
        json.dump(policy_insights, f, indent=2, default=str)
    
    # Final summary
    print(f"\n=== Analysis Complete ===")
    print(f"Best model: {best_model_name}")
    print(f"Best F1 score: {results[best_model_name]['f1']:.3f}")
    print(f"Countries likely to reduce emissions: {policy_insights['successful_countries']}")
    print(f"Overall success rate: {policy_insights['success_rate']:.1f}%")
    
    print(f"\nFiles saved to {OUT_DIR}/:")
    print("- emissions_reduction_targets.csv")
    print("- classification_performance_metrics.csv") 
    print("- feature_importance_rankings.csv")
    print("- classification_country_predictions.csv")
    print("- classifier_performance_plots.png")
    print("- feature_importance_plot.png")
    print("- policy_insights_summary.json")


if __name__ == '__main__':
    main()