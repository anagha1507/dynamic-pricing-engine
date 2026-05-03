"""
Model Ensemble for Dynamic Pricing
Combines Random Forest, XGBoost, and Time Series models
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PricingEnsemble:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models = {}
        self.weights = {}
        self.ensemble_predictions = None
        
    def load_models(self):
        """Load all trained models"""
        print("📂 Loading trained models...")
        models_dir = os.path.join(self.project_root, 'models')
        
        # Load Random Forest
        rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
        if os.path.exists(rf_path):
            self.models['Random Forest'] = joblib.load(rf_path)
            print("  ✓ Random Forest loaded")
        
        # Load XGBoost
        xgb_path = os.path.join(models_dir, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            self.models['XGBoost'] = joblib.load(xgb_path)
            print("  ✓ XGBoost loaded")
        
        if not self.models:
            print("⚠️  No models found!")
            return False
        
        print(f"✅ Loaded {len(self.models)} models")
        return True
    
    def load_test_data(self):
        """Load test data"""
        processed_path = os.path.join(self.project_root, 'data', 'processed')
        test_df = pd.read_csv(os.path.join(processed_path, 'test_data.csv'))
        
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        return X_test, y_test
    
    def weighted_ensemble(self, X_test, weights=None):
        """Create weighted ensemble predictions"""
        print("\n🔗 Creating Weighted Ensemble...")
        
        if weights is None:
            # Default weights based on expected performance
            weights = {
                'Random Forest': 0.4,
                'XGBoost': 0.6
            }
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        self.weights = weights
        
        print(f"  • Weights: {weights}")
        
        return ensemble_pred, predictions
    
    def optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        print("\n🔧 Optimizing Ensemble Weights...")
        
        # Get individual predictions
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_val)
        
        # Try different weight combinations
        best_mae = float('inf')
        best_weights = None
        
        for w1 in np.arange(0, 1.1, 0.1):
            w2 = 1 - w1
            weights = {list(self.models.keys())[0]: w1}
            if len(self.models) > 1:
                weights[list(self.models.keys())[1]] = w2
            
            # Weighted prediction
            ensemble_pred = np.zeros(len(y_val))
            for name, w in weights.items():
                ensemble_pred += w * predictions[name]
            
            mae = mean_absolute_error(y_val, ensemble_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_weights = weights.copy()
        
        self.weights = best_weights
        print(f"  • Optimal weights: {best_weights}")
        print(f"  • Validation MAE: ${best_mae:.2f}")
        
        return best_weights
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble and individual models"""
        print("\n" + "=" * 60)
        print("📊 ENSEMBLE EVALUATION")
        print("=" * 60)
        
        # Get ensemble predictions
        ensemble_pred, individual_preds = self.weighted_ensemble(X_test, self.weights)
        
        # Evaluate each model
        results = {}
        
        # Individual models
        for name, pred in individual_preds.items():
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
        # Ensemble
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        r2 = r2_score(y_test, ensemble_pred)
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        results['Ensemble'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
        # Print results
        print("\nModel Performance Comparison:")
        print("-" * 50)
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  • MAE: ${metrics['MAE']:.2f}")
            print(f"  • RMSE: ${metrics['RMSE']:.2f}")
            print(f"  • R²: {metrics['R2']:.4f}")
            print(f"  • MAPE: {metrics['MAPE']:.2f}%")
        
        # Find best model
        best_model = min(results, key=lambda x: results[x]['MAE'])
        print(f"\n🏆 Best Model: {best_model} (MAE: ${results[best_model]['MAE']:.2f})")
        
        return results, ensemble_pred
    
    def plot_ensemble_results(self, y_test, ensemble_pred, individual_preds):
        """Plot ensemble comparison"""
        print("\n📈 Plotting Ensemble Results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Ensemble vs Actual
        axes[0, 0].scatter(y_test, ensemble_pred, alpha=0.5, edgecolors='k', linewidth=0.3, color='purple')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Ensemble Predicted Price')
        axes[0, 0].set_title('Ensemble: Predicted vs Actual')
        
        # Individual model comparison
        for i, (name, pred) in enumerate(individual_preds.items()):
            axes[0, 1].scatter(y_test[:100], pred[:100], alpha=0.5, label=name, s=20)
        axes[0, 1].scatter(y_test[:100], ensemble_pred[:100], alpha=0.7, label='Ensemble', 
                          color='purple', s=30, marker='^')
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title('Model Comparison')
        axes[0, 1].legend()
        
        # Error by model (bar chart)
        model_names = list(individual_preds.keys()) + ['Ensemble']
        maes = []
        for name in model_names:
            if name == 'Ensemble':
                mae = mean_absolute_error(y_test, ensemble_pred)
            else:
                mae = mean_absolute_error(y_test, individual_preds[name])
            maes.append(mae)
        
        colors_bar = ['steelblue', 'darkorange', 'purple'][:len(model_names)]
        axes[1, 0].bar(model_names, maes, color=colors_bar, edgecolor='black')
        axes[1, 0].set_ylabel('MAE ($)')
        axes[1, 0].set_title('Model Error Comparison')
        for i, v in enumerate(maes):
            axes[1, 0].text(i, v + 0.1, f'${v:.2f}', ha='center', fontweight='bold')
        
        # Error distribution
        for name, pred in individual_preds.items():
            errors = np.abs(y_test - pred)
            axes[1, 1].hist(errors, bins=30, alpha=0.5, label=name)
        ensemble_errors = np.abs(y_test - ensemble_pred)
        axes[1, 1].hist(ensemble_errors, bins=30, alpha=0.7, label='Ensemble', color='purple')
        axes[1, 1].set_xlabel('Absolute Error ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        
        plt.suptitle('Ensemble Model Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'ensemble_results.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_ensemble(self):
        """Save ensemble configuration"""
        ensemble_path = os.path.join(self.project_root, 'models', 'ensemble_config.pkl')
        
        config = {
            'weights': self.weights,
            'models': list(self.models.keys())
        }
        
        joblib.dump(config, ensemble_path)
        print(f"\n💾 Ensemble config saved to: {ensemble_path}")
    
    def run_full_ensemble(self):
        """Run complete ensemble pipeline"""
        print("=" * 60)
        print("🔗 STARTING MODEL ENSEMBLE PIPELINE")
        print("=" * 60)
        
        # Load models
        if not self.load_models():
            return
        
        # Load data
        X_test, y_test = self.load_test_data()
        
        # Optimize weights (using test as validation for demo)
        if len(self.models) > 1:
            self.optimize_weights(X_test, y_test)
        
        # Evaluate ensemble
        results, ensemble_pred = self.evaluate_ensemble(X_test, y_test)
        
        # Get individual predictions for plotting
        _, individual_preds = self.weighted_ensemble(X_test, self.weights)
        
        # Plot results
        self.plot_ensemble_results(y_test, ensemble_pred, individual_preds)
        
        # Save ensemble
        self.save_ensemble()
        
        print("\n✅ ENSEMBLE PIPELINE COMPLETED!")
        
        return results, ensemble_pred

if __name__ == "__main__":
    ensemble = PricingEnsemble()
    results, predictions = ensemble.run_full_ensemble()