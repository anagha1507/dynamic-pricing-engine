"""
Random Forest Model for Dynamic Pricing
Includes: Training, Hyperparameter Tuning, SHAP Analysis, Feature Importance
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

class RandomForestPricingModel:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.shap_values = None
        
    def load_data(self):
        """Load processed data"""
        print("📂 Loading processed data...")
        
        processed_path = os.path.join(self.project_root, 'data', 'processed')
        
        train_df = pd.read_csv(os.path.join(processed_path, 'train_data.csv'))
        val_df = pd.read_csv(os.path.join(processed_path, 'val_data.csv'))
        test_df = pd.read_csv(os.path.join(processed_path, 'test_data.csv'))
        
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_val = val_df.drop('target', axis=1)
        y_val = val_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        print(f"✅ Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_base_model(self, X_train, y_train):
        """Train a base Random Forest model"""
        print("\n" + "=" * 60)
        print("🌲 TRAINING BASE RANDOM FOREST MODEL")
        print("=" * 60)
        
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("Training in progress...")
        self.model.fit(X_train, y_train)
        print("✅ Base model trained successfully!")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        print("\n" + "=" * 60)
        print("🔧 HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5],
            'bootstrap': [True, False]
        }
        
        # Use RandomizedSearchCV for efficiency
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter combinations to try
            cv=3,  # 3-fold cross-validation
            verbose=2,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_absolute_error'
        )
        
        print("Searching for best parameters (this may take a while)...")
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        
        print("\n✅ Best parameters found:")
        for param, value in self.best_params.items():
            print(f"  • {param}: {value}")
        
        return self.model, self.best_params
    
    def evaluate_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate model performance on all datasets"""
        print("\n" + "=" * 60)
        print("📊 MODEL EVALUATION")
        print("=" * 60)
        
        datasets = {
            'Train': (X_train, y_train),
            'Validation': (X_val, y_val),
            'Test': (X_test, y_test)
        }
        
        results = {}
        
        for name, (X, y) in datasets.items():
            predictions = self.model.predict(X)
            
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y - predictions) / y)) * 100
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
            print(f"\n{name} Set Results:")
            print(f"  • MAE:  {mae:.4f}")
            print(f"  • RMSE: {rmse:.4f}")
            print(f"  • R²:   {r2:.4f}")
            print(f"  • MAPE: {mape:.2f}%")
        
        return results
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        print("\n" + "=" * 60)
        print("📈 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Get feature importance
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        # Create DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
        bars = plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
        
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
            plt.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Top 20 Most Important Features:")
        print(top_features.to_string(index=False))
        
        return self.feature_importance
    
    def shap_analysis(self, X_train, sample_size=500):
        """Perform SHAP analysis for model interpretability"""
        print("\n" + "=" * 60)
        print("🔍 SHAP ANALYSIS")
        print("=" * 60)
        
        # Use a sample for SHAP (faster computation)
        if len(X_train) > sample_size:
            X_sample = X_train.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_train
        
        print(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        self.shap_values = explainer.shap_values(X_sample)
        
        print("✅ SHAP values computed!")
        
        # Plot 1: Summary Plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(self.shap_values, X_sample, show=False)
        plt.title('SHAP Feature Impact on Price Predictions')
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_root, 'models', 'shap_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Feature Importance (SHAP)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_root, 'models', 'shap_importance.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Dependence plots for top 3 features
        top_3_features = self.feature_importance.head(3)['feature'].values
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, feature in enumerate(top_3_features):
            if feature in X_sample.columns:
                shap.dependence_plot(feature, self.shap_values, X_sample, 
                                   ax=axes[i], show=False)
                axes[i].set_title(f'SHAP Dependence: {feature}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_root, 'models', 'shap_dependence.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ SHAP plots saved to models/")
        
        return self.shap_values
    
    def plot_predictions_vs_actual(self, X_test, y_test):
        """Plot predicted vs actual values"""
        print("\n📊 Plotting Predictions vs Actual...")
        
        predictions = self.model.predict(X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(y_test, predictions, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Predicted vs Actual Prices')
        
        # Residuals plot
        residuals = y_test - predictions
        axes[1].scatter(predictions, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Price')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_root, 'models', 'predictions_vs_actual.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Plots saved to models/")
    
    def save_model(self):
        """Save the trained model"""
        models_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
        joblib.dump(self.model, models_path)
        print(f"\n💾 Model saved to: {models_path}")
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Hyperparameter tuning
        self.model, self.best_params = self.hyperparameter_tuning(X_train, y_train)
        
        # Evaluate model
        results = self.evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Feature importance
        self.plot_feature_importance(top_n=20)
        
        # SHAP analysis
        self.shap_analysis(X_train, sample_size=500)
        
        # Predictions vs actual
        self.plot_predictions_vs_actual(X_test, y_test)
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("✅ RANDOM FOREST PIPELINE COMPLETED!")
        print("=" * 60)
        
        return self.model, results

if __name__ == "__main__":
    # Initialize and run
    rf_model = RandomForestPricingModel()
    model, results = rf_model.run_full_training()
    
    print("\n🎯 Final Model Performance on Test Set:")
    print(f"  • R² Score: {results['Test']['R2']:.4f}")
    print(f"  • MAE: ${results['Test']['MAE']:.2f}")
    print(f"  • MAPE: {results['Test']['MAPE']:.2f}%")