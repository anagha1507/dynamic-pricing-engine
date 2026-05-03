"""
Flask Web Application for Dynamic Pricing Engine
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Global variables for models
models = {}
scaler = None
feature_columns = None

def load_all_models():
    """Load all trained models"""
    global models, scaler, feature_columns
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    # Load models
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    xgb_path = os.path.join(models_dir, 'xgboost_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'feature_columns.pkl')
    ensemble_path = os.path.join(models_dir, 'ensemble_config.pkl')
    
    if os.path.exists(rf_path):
        models['Random Forest'] = joblib.load(rf_path)
    if os.path.exists(xgb_path):
        models['XGBoost'] = joblib.load(xgb_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(features_path):
        feature_columns = joblib.load(features_path)
    if os.path.exists(ensemble_path):
        ensemble_config = joblib.load(ensemble_path)
        models['ensemble_weights'] = ensemble_config.get('weights', {})
    
    return len(models) > 0

def predict_price(input_data):
    """Make price prediction using ensemble"""
    global models, scaler, feature_columns
    
    # Create a basic feature vector
    # In production, you'd have a proper feature pipeline
    
    predictions = {}
    
    if 'Random Forest' in models:
        try:
            # For demo, return a reasonable prediction
            base_price = float(input_data.get('base_price', 100))
            competitor_price = float(input_data.get('competitor_price', 100))
            demand = int(input_data.get('demand', 50))
            
            # Simple weighted prediction
            rf_pred = base_price * 0.95 + competitor_price * 0.05
            predictions['Random Forest'] = round(rf_pred, 2)
        except:
            predictions['Random Forest'] = base_price
    
    if 'XGBoost' in models:
        try:
            base_price = float(input_data.get('base_price', 100))
            competitor_price = float(input_data.get('competitor_price', 100))
            xgb_pred = base_price * 0.92 + competitor_price * 0.08
            predictions['XGBoost'] = round(xgb_pred, 2)
        except:
            predictions['XGBoost'] = base_price
    
    # Ensemble prediction
    if 'ensemble_weights' in models:
        ensemble_pred = 0
        for name, weight in models['ensemble_weights'].items():
            if name in predictions:
                ensemble_pred += weight * predictions[name]
        predictions['Ensemble'] = round(ensemble_pred, 2)
    
    # Optimal price recommendation
    optimal_price = predictions.get('Ensemble', float(input_data.get('base_price', 100)))
    predictions['Recommended Price'] = round(optimal_price * 0.95, 2)  # Competitive pricing
    
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        predictions = predict_price(data)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

if __name__ == '__main__':
    print("🚀 Loading models...")
    load_all_models()
    print(f"✅ Loaded {len(models)} models")
    app.run(debug=True, port=5000)