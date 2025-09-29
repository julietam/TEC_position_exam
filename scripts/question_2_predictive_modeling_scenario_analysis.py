#!/usr/bin/env python3
"""
Question 2: Predictive Modeling and Scenario Analysis for CO2 Emissions

This script develops predictive models to forecast CO2 emissions based on comprehensive
socio-economic and environmental indicators, and performs scenario analysis to answer:
"If a country increases its GDP by 10%, what is the expected percentage change in CO2 emissions, 
assuming all other factors remain constant?"

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import os
from scipy import stats
from datetime import datetime
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CO2PredictiveModel:
    """Class to handle CO2 emissions predictive modeling and scenario analysis"""
    
    def __init__(self, data_path):
        """
        Initialize the predictive modeling class
        
        Args:
            data_path (str): Path to the processed World Bank data
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.feature_importance = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        print("Loading and preparing data...")
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data shape: {self.data.shape}")
        
        # Filter out regional aggregates and focus on individual countries
        regional_keywords = [
            'World', 'income', 'Europe', 'Asia', 'Africa', 'America', 'Pacific',
            'Middle East', 'North Africa', 'Sub-Saharan', 'Latin', 'Caribbean',
            'East Asia', 'South Asia', 'Central', 'OECD', 'Euro area', 'European',
            'Arab', 'IDA', 'IBRD', 'dividend', 'fragile', 'small states', 'Least',
            'Channel Islands', 'Isle of Man'
        ]
        
        # Filter out rows that contain regional aggregates
        mask = ~self.data['country'].str.contains('|'.join(regional_keywords), case=False, na=False)
        self.data = self.data[mask].copy()
        print(f"After filtering regional aggregates: {self.data.shape}")
        
        # Remove countries with insufficient data (< 5 years of data)
        country_counts = self.data['country'].value_counts()
        valid_countries = country_counts[country_counts >= 5].index
        self.data = self.data[self.data['country'].isin(valid_countries)].copy()
        print(f"After filtering countries with < 5 years of data: {self.data.shape}")
        
        # Handle missing values in CO2_emissions_per_capita (target variable)
        initial_rows = len(self.data)
        self.data = self.data.dropna(subset=['CO2_emissions_per_capita'])
        print(f"Removed {initial_rows - len(self.data)} rows with missing CO2 target values")
        
        return self.data
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Define target variable
        target = 'CO2_emissions_per_capita'
        
        # Define features to exclude from modeling
        exclude_features = [
            'country', 'date', target, 'CO2_emissions_Mt', 
            'CO2_emissions_energy_Mt', 'CO2_emissions_transport_Mt',
            'CO2_intensity_per_GDP',  # This is derived from our target
            'GDP_per_capita_calculated'  # Similar to GDP_per_capita_current_USD
        ]
        
        # Select feature columns
        feature_columns = [col for col in self.data.columns if col not in exclude_features]
        
        # Create feature matrix and target vector
        X = self.data[feature_columns].copy()
        y = self.data[target].copy()
        
        # Check for infinite values
        print(f"Columns with infinite values: {X.columns[np.isinf(X).any()].tolist()}")
        
        # Replace infinite values with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle any remaining missing values
        print(f"Missing values before imputation: {X.isnull().sum().sum()}")
        if X.isnull().sum().sum() > 0:
            # Use median imputation for remaining missing values
            X = X.fillna(X.median())
        
        # Log-transform heavily skewed numeric features with large ranges
        skewed_features = ['GDP_current_USD', 'Population_total', 'Annual_freshwater_withdrawals_total']
        for feature in skewed_features:
            if feature in X.columns:
                # Add small constant to avoid log(0)
                X[feature] = np.log1p(X[feature].clip(lower=0))
                print(f"Log-transformed skewed feature: {feature}")
        
        # Check for constant or near-constant features
        near_constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1 or X[col].std() / X[col].mean() < 0.001:
                near_constant_features.append(col)
                
        if near_constant_features:
            print(f"Removing near-constant features: {near_constant_features}")
            X = X.drop(columns=near_constant_features)
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Features included: {list(X.columns)}")
        
        return X, y, X.columns.tolist()
        
    def split_and_scale_data(self, X, y):
        """Split data into train/test and apply scaling"""
        print("Splitting and scaling data...")
        
        # Split the data (80-20 split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Apply robust scaling (less sensitive to outliers)
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return self.X_train_scaled, self.X_test_scaled
        
    def train_models(self):
        """Train multiple regression models and select the best one"""
        print("Training multiple regression models...")
        
        # Define models to test
        models_to_test = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Train and evaluate models using cross-validation
        model_scores = {}
        
        for name, model in models_to_test.items():
            print(f"Training {name}...")
            
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='r2', n_jobs=-1)
            
            # Fit model on full training data
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_mae = mean_absolute_error(self.y_test, test_pred)
            
            # Store results
            model_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'model': model
            }
            
            print(f"{name} - CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f}), Test R²: {test_r2:.3f}")
        
        # Select best model based on cross-validation performance
        self.best_model_name = max(model_scores.keys(), 
                                 key=lambda k: model_scores[k]['cv_mean'])
        self.best_model = model_scores[self.best_model_name]['model']
        self.results = model_scores
        
        print(f"\nBest model: {self.best_model_name}")
        print(f"CV R²: {model_scores[self.best_model_name]['cv_mean']:.3f}")
        print(f"Test R²: {model_scores[self.best_model_name]['test_r2']:.3f}")
        
        return model_scores
        
    def analyze_feature_importance(self):
        """Analyze feature importance using the best model"""
        print("Analyzing feature importance...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models have built-in feature importance
            importance = self.best_model.feature_importances_
        else:
            # Use permutation importance for other models
            perm_importance = permutation_importance(
                self.best_model, self.X_test_scaled, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            importance = perm_importance.importances_mean
        
        # Create feature importance dataframe
        feature_names = self.X_train.columns
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(self.feature_importance.head(10).to_string(index=False))
        
        return self.feature_importance
        
    def gdp_scenario_analysis(self, gdp_increase_percent=10):
        """
        Perform scenario analysis with GDP increase
        
        Args:
            gdp_increase_percent (float): Percentage increase in GDP per capita
        """
        print(f"\nPerforming GDP scenario analysis ({gdp_increase_percent}% increase)...")
        
        # Get the most recent year of data for each country
        latest_data = self.data.loc[self.data.groupby('country')['date'].idxmax()].copy()
        
        # Filter out rows with missing values in key features
        latest_data = latest_data.dropna(subset=['GDP_per_capita_current_USD', 'CO2_emissions_per_capita'])
        
        print(f"Analyzing {len(latest_data)} countries")
        
        # Prepare baseline data
        baseline_features = latest_data[self.X_train.columns].copy()
        baseline_co2 = latest_data['CO2_emissions_per_capita'].copy()
        
        # Create scenario data with increased GDP
        scenario_features = baseline_features.copy()
        scenario_features['GDP_per_capita_current_USD'] *= (1 + gdp_increase_percent/100)
        
        # Make predictions for baseline and scenario
        baseline_features_scaled = self.scaler.transform(baseline_features.fillna(baseline_features.median()))
        scenario_features_scaled = self.scaler.transform(scenario_features.fillna(scenario_features.median()))
        
        baseline_pred = self.best_model.predict(baseline_features_scaled)
        scenario_pred = self.best_model.predict(scenario_features_scaled)
        
        # Calculate percentage change in CO2 emissions
        co2_change_percent = ((scenario_pred - baseline_pred) / baseline_pred) * 100
        
        # Create results dataframe
        scenario_results = pd.DataFrame({
            'country': latest_data['country'].values,
            'baseline_gdp_per_capita': baseline_features['GDP_per_capita_current_USD'].values,
            'scenario_gdp_per_capita': scenario_features['GDP_per_capita_current_USD'].values,
            'actual_co2_per_capita': baseline_co2.values,
            'predicted_baseline_co2': baseline_pred,
            'predicted_scenario_co2': scenario_pred,
            'co2_change_percent': co2_change_percent,
            'gdp_level': pd.cut(baseline_features['GDP_per_capita_current_USD'], 
                              bins=[0, 5000, 15000, 50000, np.inf], 
                              labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High'])
        })
        
        # Summary statistics
        scenario_summary = {
            'mean_co2_change': co2_change_percent.mean(),
            'median_co2_change': np.median(co2_change_percent),
            'std_co2_change': co2_change_percent.std(),
            'min_co2_change': co2_change_percent.min(),
            'max_co2_change': co2_change_percent.max(),
            'countries_analyzed': len(co2_change_percent)
        }
        
        print(f"\nScenario Analysis Results ({gdp_increase_percent}% GDP increase):")
        print(f"Mean CO2 change: {scenario_summary['mean_co2_change']:.2f}%")
        print(f"Median CO2 change: {scenario_summary['median_co2_change']:.2f}%")
        print(f"Standard deviation: {scenario_summary['std_co2_change']:.2f}%")
        print(f"Range: {scenario_summary['min_co2_change']:.2f}% to {scenario_summary['max_co2_change']:.2f}%")
        
        # Analysis by income group
        income_analysis = scenario_results.groupby('gdp_level')['co2_change_percent'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        print(f"\nCO2 Change by Income Level:")
        print(income_analysis.to_string())
        
        return scenario_results, scenario_summary, income_analysis
        
    def create_visualizations(self, output_dir, scenario_results=None):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Model Performance Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cross-validation scores
        models = list(self.results.keys())
        cv_scores = [self.results[model]['cv_mean'] for model in models]
        cv_stds = [self.results[model]['cv_std'] for model in models]
        
        ax1.bar(range(len(models)), cv_scores, yerr=cv_stds, capsize=5, 
                color='skyblue', alpha=0.7, edgecolor='darkblue')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Cross-Validation R²')
        ax1.set_title('Model Performance Comparison (Cross-Validation)')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Test R² vs Train R² (overfitting check)
        train_r2 = [self.results[model]['train_r2'] for model in models]
        test_r2 = [self.results[model]['test_r2'] for model in models]
        
        ax2.scatter(train_r2, test_r2, s=100, alpha=0.7, color='coral')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('Training R²')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Training vs Test Performance')
        ax2.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(models):
            ax2.annotate(model, (train_r2[i], test_r2[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Feature importance (top 15)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            ax3.barh(range(len(top_features)), top_features['importance'], 
                    color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'], fontsize=8)
            ax3.set_xlabel('Feature Importance')
            ax3.set_title(f'Top 15 Feature Importance ({self.best_model_name})')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # Prediction vs Actual scatter plot
        test_pred = self.best_model.predict(self.X_test_scaled)
        ax4.scatter(self.y_test, test_pred, alpha=0.6, color='purple')
        ax4.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax4.set_xlabel('Actual CO2 Emissions per Capita')
        ax4.set_ylabel('Predicted CO2 Emissions per Capita')
        ax4.set_title(f'Predictions vs Actual ({self.best_model_name})')
        ax4.grid(True, alpha=0.3)
        
        # Add R² to the plot
        test_r2_best = self.results[self.best_model_name]['test_r2']
        ax4.text(0.05, 0.95, f'R² = {test_r2_best:.3f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residuals Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        residuals = test_pred - self.y_test
        
        # Residuals vs Predicted
        ax1.scatter(test_pred, residuals, alpha=0.6, color='orange')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for residual normality
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residuals_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scenario Analysis Visualizations
        if scenario_results is not None:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Distribution of CO2 change percentages
            ax1.hist(scenario_results['co2_change_percent'], bins=30, alpha=0.7, 
                    color='lightblue', edgecolor='darkblue')
            ax1.axvline(scenario_results['co2_change_percent'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {scenario_results["co2_change_percent"].mean():.2f}%')
            ax1.set_xlabel('CO2 Emissions Change (%)')
            ax1.set_ylabel('Number of Countries')
            ax1.set_title('Distribution of CO2 Emissions Change (10% GDP Increase)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot by income group
            scenario_results.boxplot(column='co2_change_percent', by='gdp_level', ax=ax2)
            ax2.set_xlabel('GDP Level')
            ax2.set_ylabel('CO2 Emissions Change (%)')
            ax2.set_title('CO2 Change by Income Level')
            ax2.grid(True, alpha=0.3)
            plt.suptitle('')  # Remove default title
            
            # CO2 change vs baseline GDP
            ax3.scatter(scenario_results['baseline_gdp_per_capita'], 
                       scenario_results['co2_change_percent'], alpha=0.6, color='green')
            ax3.set_xscale('log')
            ax3.set_xlabel('Baseline GDP per Capita (USD, log scale)')
            ax3.set_ylabel('CO2 Emissions Change (%)')
            ax3.set_title('CO2 Sensitivity vs Economic Development Level')
            ax3.grid(True, alpha=0.3)
            
            # Top 10 countries with highest and lowest CO2 sensitivity
            top_sensitive = scenario_results.nlargest(10, 'co2_change_percent')
            bottom_sensitive = scenario_results.nsmallest(10, 'co2_change_percent')
            
            combined = pd.concat([bottom_sensitive, top_sensitive])
            colors = ['red'] * 10 + ['blue'] * 10
            
            ax4.barh(range(len(combined)), combined['co2_change_percent'], color=colors, alpha=0.7)
            ax4.set_yticks(range(len(combined)))
            ax4.set_yticklabels(combined['country'], fontsize=8)
            ax4.set_xlabel('CO2 Emissions Change (%)')
            ax4.set_title('Countries with Highest/Lowest CO2 Sensitivity')
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'scenario_analysis_results.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Visualizations saved to output directory")
        
    def save_results(self, output_dir, scenario_results=None, scenario_summary=None, income_analysis=None):
        """Save model and results"""
        print("Saving results...")
        
        # Save the best model
        model_path = os.path.join(output_dir, f'best_model_{self.best_model_name.replace(" ", "_").lower()}.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Save the scaler
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save model performance results
        performance_results = {}
        for model_name, results in self.results.items():
            performance_results[model_name] = {
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'train_r2': results['train_r2'],
                'test_r2': results['test_r2'],
                'test_rmse': results['test_rmse'],
                'test_mae': results['test_mae']
            }
        
        # Save comprehensive results
        results_summary = {
            'best_model': self.best_model_name,
            'model_performance': performance_results,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'training_data_shape': self.X_train.shape,
            'test_data_shape': self.X_test.shape,
            'features_used': list(self.X_train.columns)
        }
        
        if scenario_summary is not None:
            results_summary['scenario_analysis'] = {
                'gdp_increase_percent': 10,
                'summary_statistics': scenario_summary,
                'income_level_analysis': income_analysis.to_dict() if income_analysis is not None else None
            }
        
        # Save to JSON
        results_path = os.path.join(output_dir, 'modeling_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save scenario results to CSV if available
        if scenario_results is not None:
            scenario_path = os.path.join(output_dir, 'gdp_scenario_results.csv')
            scenario_results.to_csv(scenario_path, index=False)
        
        # Save feature importance to CSV
        if self.feature_importance is not None:
            importance_path = os.path.join(output_dir, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
        
        print("Results saved successfully")


def main():
    """Main execution function"""
    print("=" * 70)
    print("CO2 EMISSIONS PREDICTIVE MODELING & SCENARIO ANALYSIS")
    print("=" * 70)
    
    # Initialize the modeling class
    data_path = 'outputs/processed_world_bank_data.csv'
    predictor = CO2PredictiveModel(data_path)
    
    # Create output directories
    output_dir = 'outputs/question_2'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and prepare data
        data = predictor.load_and_prepare_data()
        
        # Prepare features
        X, y, feature_columns = predictor.prepare_features()
        
        # Split and scale data
        X_train_scaled, X_test_scaled = predictor.split_and_scale_data(X, y)
        
        # Train models
        model_results = predictor.train_models()
        
        # Analyze feature importance
        feature_importance = predictor.analyze_feature_importance()
        
        # Perform GDP scenario analysis
        scenario_results, scenario_summary, income_analysis = predictor.gdp_scenario_analysis(gdp_increase_percent=10)
        
        # Create visualizations
        predictor.create_visualizations(output_dir, scenario_results)
        
        # Save results
        predictor.save_results(output_dir, scenario_results, scenario_summary, income_analysis)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Best model: {predictor.best_model_name}")
        print(f"Test R²: {predictor.results[predictor.best_model_name]['test_r2']:.3f}")
        print(f"Mean CO2 change (10% GDP increase): {scenario_summary['mean_co2_change']:.2f}%")
        print(f"Countries analyzed: {scenario_summary['countries_analyzed']}")
        print(f"Outputs saved to: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
