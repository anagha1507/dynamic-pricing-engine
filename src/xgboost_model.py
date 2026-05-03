"""
XGBoost Model for Dynamic Pricing
Includes: Training, Hyperparameter Tuning, Cost-Sensitive Learning, Feature Importance
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class XGBoostPricingModel:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
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
    
    def create_cost_sensitive_weights(self, y_train):
        """Create sample weights for cost-sensitive learning
        
        In pricing: Overpricing (predicting higher than actual) is worse than underpricing.
        Overpricing loses customers, underpricing just reduces margin slightly.
        """
        print("\n💰 Creating cost-sensitive learning weights...")
        
        # Calculate price distribution
        median_price = np.median(y_train)
        
        # Higher weights for higher-priced items (more revenue at stake)
        sample_weights = np.ones(len(y_train))
        
        # Weight based on price percentile
        price_percentile = pd.Series(y_train).rank(pct=True)
        
        # Higher weight for higher prices (more risk)
        sample_weights = 1.0 + (price_percentile * 2)
        
        print(f"  • Weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
        print(f"  • Median weight: {np.median(sample_weights):.2f}")
        
        return sample_weights
    
    def train_base_model(self, X_train, y_train, sample_weights=None):
        """Train a base XGBoost model"""
        print("\n" + "=" * 60)
        print("⚡ TRAINING BASE XGBOOST MODEL")
        print("=" * 60)
        
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        print("Training with cost-sensitive weights...")
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        print("✅ Base model trained successfully!")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, sample_weights=None):
        """Perform hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("🔧 HYPERPARAMETER TUNING - XGBOOST")
        print("=" * 60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 1.5]
        }
        
        # Use RandomizedSearchCV
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=15,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_absolute_error'
        )
        
        print("Searching for best parameters...")
        
        # Fit with sample weights if provided
        fit_params = {'sample_weight': sample_weights} if sample_weights is not None else {}
        random_search.fit(X_train, y_train, **fit_params)
        
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        
        print("\n✅ Best parameters found:")
        for param, value in self.best_params.items():
            print(f"  • {param}: {value}")
        
        return self.model, self.best_params
    
    def evaluate_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "=" * 60)
        print("📊 MODEL EVALUATION - XGBOOST")
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
            
            # MAPE
            mape = np.mean(np.abs((y - predictions) / y)) * 100
            
            # Custom overpricing penalty metric
            overprice_mask = predictions > y
            if overprice_mask.sum() > 0:
                overprice_error = np.mean(predictions[overprice_mask] - y[overprice_mask])
                overprice_pct = overprice_mask.sum() / len(y) * 100
            else:
                overprice_error = 0
                overprice_pct = 0
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Overpricing_Error': overprice_error,
                'Overpricing_%': overprice_pct
            }
            
            print(f"\n{name} Set Results:")
            print(f"  • MAE:  {mae:.4f}")
            print(f"  • RMSE: {rmse:.4f}")
            print(f"  • R²:   {r2:.4f}")
            print(f"  • MAPE: {mape:.2f}%")
            print(f"  • Overpricing Error: ${overprice_error:.2f}")
            print(f"  • Overpricing Rate: {overprice_pct:.1f}%")
        
        return results
    
    def plot_feature_importance(self, top_n=20):
        """Plot XGBoost feature importance"""
        print("\n" + "=" * 60)
        print("📈 XGBOOST FEATURE IMPORTANCE")
        print("=" * 60)
        
        # Get feature importance (multiple types available in XGBoost)
        importance_types = ['weight', 'gain', 'cover']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        for idx, imp_type in enumerate(importance_types):
            importance = self.model.get_booster().get_score(importance_type=imp_type)
            
            # Convert to DataFrame
            imp_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp_df)))
            axes[idx].barh(range(len(imp_df)), imp_df['importance'].values, color=colors)
            axes[idx].set_yticks(range(len(imp_df)))
            axes[idx].set_yticklabels(imp_df['feature'].values, fontsize=8)
            axes[idx].set_xlabel('Score')
            axes[idx].set_title(f'Feature Importance ({imp_type})')
            axes[idx].invert_yaxis()
        
        plt.suptitle('XGBoost Feature Importance - Multiple Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'xgboost_feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print top features by 'gain' (most important for performance)
        importance_gain = self.model.get_booster().get_score(importance_type='gain')
        gain_df = pd.DataFrame({
            'feature': list(importance_gain.keys()),
            'gain': list(importance_gain.values())
        }).sort_values('gain', ascending=False).head(20)
        
        print("\nTop 20 Features by Gain:")
        print(gain_df.to_string(index=False))
        
        self.feature_importance = gain_df
        
        return gain_df
    
    def plot_learning_curve(self):
        """Plot learning curve from training history"""
        print("\n📈 Plotting Learning Curve...")
        
        results = self.model.evals_result()
        
        if results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # RMSE
            epochs = len(results['validation_0']['rmse'])
            x_axis = range(0, epochs)
            
            axes[0].plot(x_axis, results['validation_0']['rmse'], label='Train')
            if 'validation_1' in results:
                axes[0].plot(x_axis, results['validation_1']['rmse'], label='Validation')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('XGBoost Learning Curve - RMSE')
            axes[0].legend()
            
            # MAE
            axes[1].plot(x_axis, results['validation_0']['mae'], label='Train')
            if 'validation_1' in results:
                axes[1].plot(x_axis, results['validation_1']['mae'], label='Validation')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('XGBoost Learning Curve - MAE')
            axes[1].legend()
            
            plt.tight_layout()
            models_dir = os.path.join(self.project_root, 'models')
            plt.savefig(os.path.join(models_dir, 'xgboost_learning_curve.png'), dpi=150, bbox_inches='tight')
            plt.show()
    
    def compare_with_random_forest(self, X_test, y_test):
        """Compare XGBoost with Random Forest if available"""
        rf_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
        
        if os.path.exists(rf_path):
            print("\n" + "=" * 60)
            print("🔄 COMPARING XGBOOST vs RANDOM FOREST")
            print("=" * 60)
            
            rf_model = joblib.load(rf_path)
            
            # Predictions
            xgb_pred = self.model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            
            # Metrics
            metrics = {
                'XGBoost': {
                    'MAE': mean_absolute_error(y_test, xgb_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                    'R2': r2_score(y_test, xgb_pred),
                    'MAPE': np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
                },
                'Random Forest': {
                    'MAE': mean_absolute_error(y_test, rf_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                    'R2': r2_score(y_test, rf_pred),
                    'MAPE': np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
                }
            }
            
            # Comparison table
            comparison_df = pd.DataFrame(metrics).round(4)
            print("\n📊 Model Comparison on Test Set:")
            print(comparison_df)
            
            # Visual comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter plot
            axes[0].scatter(y_test, xgb_pred, alpha=0.5, label='XGBoost', edgecolors='k', linewidth=0.3)
            axes[0].scatter(y_test, rf_pred, alpha=0.5, label='Random Forest', edgecolors='k', linewidth=0.3)
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Price')
            axes[0].set_ylabel('Predicted Price')
            axes[0].set_title('XGBoost vs Random Forest Predictions')
            axes[0].legend()
            
            # Error distribution
            xgb_errors = np.abs(y_test - xgb_pred)
            rf_errors = np.abs(y_test - rf_pred)
            
            axes[1].hist(xgb_errors, bins=50, alpha=0.6, label='XGBoost', color='blue')
            axes[1].hist(rf_errors, bins=50, alpha=0.6, label='Random Forest', color='green')
            axes[1].set_xlabel('Absolute Error')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Error Distribution Comparison')
            axes[1].legend()
            
            plt.tight_layout()
            models_dir = os.path.join(self.project_root, 'models')
            plt.savefig(os.path.join(models_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
            return comparison_df
        else:
            print("\n⚠️  Random Forest model not found. Skipping comparison.")
            return None
    
    def save_model(self):
        """Save the trained model"""
        models_path = os.path.join(self.project_root, 'models', 'xgboost_model.pkl')
        joblib.dump(self.model, models_path)
        print(f"\n💾 Model saved to: {models_path}")
    
    def plot_predictions_vs_actual(self, X_test, y_test):
        """Plot predictions vs actual with cost-sensitive zones"""
        print("\n📊 Plotting Predictions vs Actual (with Cost Zones)...")
        
        predictions = self.model.predict(X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter with cost zones
        axes[0].scatter(y_test, predictions, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2, label='Perfect')
        axes[0].fill_between([y_test.min(), y_test.max()], 
                             [y_test.min(), y_test.max()],
                             [y_test.min()*0.9, y_test.max()*0.9], 
                             alpha=0.2, color='blue', label='Underpricing Zone')
        axes[0].fill_between([y_test.min(), y_test.max()], 
                             [y_test.min(), y_test.max()],
                             [y_test.min()*1.1, y_test.max()*1.1], 
                             alpha=0.2, color='red', label='Overpricing Zone (Costly!)')
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Predictions with Cost-Sensitive Zones')
        axes[0].legend()
        
        # Residuals
        residuals = y_test - predictions
        colors = ['red' if r < 0 else 'blue' for r in residuals]
        axes[1].scatter(predictions, residuals, c=colors, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='green', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Price')
        axes[1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1].set_title('Residuals: Red=Overpriced, Blue=Underpriced')
        
        plt.tight_layout()
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'xgboost_predictions_cost.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_full_pipeline(self):
        """Run complete XGBoost pipeline"""
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Create cost-sensitive weights
        sample_weights = self.create_cost_sensitive_weights(y_train)
        
        # Train base model
        self.model = self.train_base_model(X_train, y_train, sample_weights)
        
        # Evaluate base model
        self.evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Hyperparameter tuning
        self.model, self.best_params = self.hyperparameter_tuning(X_train, y_train, sample_weights)
        
        # Final evaluation
        results = self.evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Feature importance
        self.plot_feature_importance(top_n=20)
        
        # Predictions vs actual
        self.plot_predictions_vs_actual(X_test, y_test)
        
        # Compare with Random Forest
        self.compare_with_random_forest(X_test, y_test)
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("✅ XGBOOST PIPELINE COMPLETED!")
        print("=" * 60)
        
        return self.model, results

if __name__ == "__main__":
    xgb_model = XGBoostPricingModel()
    model, results = xgb_model.run_full_pipeline()
    
    print("\n🎯 Final XGBoost Performance on Test Set:")
    print(f"  • R² Score: {results['Test']['R2']:.4f}")
    print(f"  • MAE: ${results['Test']['MAE']:.2f}")
    print(f"  • MAPE: {results['Test']['MAPE']:.2f}%")
    print(f"  • Overpricing Rate: {results['Test']['Overpricing_%']:.1f}%")