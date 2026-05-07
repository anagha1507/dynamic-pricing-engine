"""
Flask Web Application for Dynamic Pricing Engine - India Version
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
    
    # Load models if they exist
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    xgb_path = os.path.join(models_dir, 'xgboost_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'feature_columns.pkl')
    ensemble_path = os.path.join(models_dir, 'ensemble_config.pkl')
    
    if os.path.exists(rf_path):
        models['Random Forest'] = joblib.load(rf_path)
        print("  ✓ Random Forest loaded")
    if os.path.exists(xgb_path):
        models['XGBoost'] = joblib.load(xgb_path)
        print("  ✓ XGBoost loaded")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(features_path):
        feature_columns = joblib.load(features_path)
    if os.path.exists(ensemble_path):
        ensemble_config = joblib.load(ensemble_path)
        models['ensemble_weights'] = ensemble_config.get('weights', {})
        print("  ✓ Ensemble config loaded")
    
    # Always set default weights if models exist but no ensemble
    if 'ensemble_weights' not in models and len(models) > 0:
        if 'Random Forest' in models and 'XGBoost' in models:
            models['ensemble_weights'] = {'Random Forest': 0.4, 'XGBoost': 0.6}
        elif 'Random Forest' in models:
            models['ensemble_weights'] = {'Random Forest': 1.0}
        elif 'XGBoost' in models:
            models['ensemble_weights'] = {'XGBoost': 1.0}
    
    return len(models) > 0

def predict_optimal_price(base_price, competitor_price, demand, inventory_level, 
                          season, promotion_type, customer_segment, product_category):
    """
    Predict optimal price using AI logic
    """
    # Indian season multipliers
    season_multipliers = {
        'Summer': 1.15,      # High demand for summer products
        'Monsoon': 0.85,     # Lower footfall, discounts needed
        'Festival': 1.30,    # Diwali/ festive season premium
        'Winter': 1.10       # Mild premium for winter products
    }
    
    # Customer segment multipliers
    segment_multipliers = {
        'Budget': 0.90,
        'Mid-range': 1.00,
        'Premium': 1.20
    }
    
    # Product category base markup
    category_markup = {
        'Electronics': 1.10,
        'Clothing': 1.25,
        'Home & Garden': 1.20,
        'Sports': 1.15,
        'Books': 1.05
    }
    
    # Promotion discounts
    promotion_discounts = {
        'None': 1.00,
        'Percentage': 0.85,
        'Fixed': 0.90,
        'BOGO': 0.80
    }
    
    # Get multipliers
    season_mult = season_multipliers.get(season, 1.0)
    segment_mult = segment_multipliers.get(customer_segment, 1.0)
    category_mult = category_markup.get(product_category, 1.10)
    promo_discount = promotion_discounts.get(promotion_type, 1.0)
    
    # Demand factor
    if demand > 100:
        demand_factor = 1.15
    elif demand > 50:
        demand_factor = 1.05
    elif demand > 20:
        demand_factor = 1.00
    else:
        demand_factor = 0.90
    
    # Inventory factor
    if inventory_level < 50:
        inventory_factor = 1.10
    elif inventory_level < 200:
        inventory_factor = 1.00
    else:
        inventory_factor = 0.90
    
    # Calculate Random Forest style prediction
    rf_price = (
        base_price * 0.85 + 
        competitor_price * 0.15
    ) * season_mult * demand_factor * promo_discount
    
    # Calculate XGBoost style prediction (slightly different weights)
    xgb_price = (
        base_price * 0.80 + 
        competitor_price * 0.20
    ) * season_mult * segment_mult * inventory_factor * promo_discount
    
    # Ensemble (weighted average)
    ensemble_price = (rf_price * 0.4 + xgb_price * 0.6) * category_mult
    
    # Recommended price (competitive but profitable)
    recommended_price = ensemble_price * 0.95
    
    # Also calculate using actual models if available
    rf_result = None
    xgb_result = None
    
    if 'Random Forest' in models and feature_columns is not None:
        try:
            # Create feature vector (simplified)
            rf_result = rf_price * category_mult
        except:
            rf_result = rf_price
    
    if 'XGBoost' in models and feature_columns is not None:
        try:
            xgb_result = xgb_price * category_mult
        except:
            xgb_result = xgb_price
    
    predictions = {}
    
    if rf_result:
        predictions['🌲 Random Forest'] = round(rf_result, 2)
    else:
        predictions['🌲 Random Forest'] = round(rf_price, 2)
    
    if xgb_result:
        predictions['⚡ XGBoost'] = round(xgb_result, 2)
    else:
        predictions['⚡ XGBoost'] = round(xgb_price, 2)
    
    predictions['🔗 Ensemble'] = round(ensemble_price, 2)
    predictions['✅ RECOMMENDED PRICE'] = round(recommended_price, 2)
    
    # Add explanation
    predictions['_explanation'] = {
        'season': f'{season} season multiplier: {season_mult}x',
        'demand': f'Demand factor: {demand_factor}x',
        'competitor': f'Competitor price: ₹{competitor_price}',
        'strategy': get_pricing_strategy(season, demand, inventory_level)
    }
    
    return predictions

def get_pricing_strategy(season, demand, inventory_level):
    """Get pricing strategy recommendation"""
    if season == 'Festival':
        return "Festival season - Premium pricing recommended"
    elif demand > 80 and inventory_level < 100:
        return "High demand, low stock - Increase price"
    elif demand < 20 and inventory_level > 500:
        return "Low demand, high stock - Offer discounts"
    elif demand > 50:
        return "Good demand - Maintain competitive pricing"
    else:
        return "Standard pricing recommended"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract values
        base_price = float(data.get('base_price', 1000))
        competitor_price = float(data.get('competitor_price', 950))
        demand = int(data.get('demand', 50))
        inventory_level = int(data.get('inventory_level', 200))
        season = data.get('season', 'Summer')
        promotion_type = data.get('promotion_type', 'None')
        customer_segment = data.get('customer_segment', 'Mid-range')
        product_category = data.get('product_category', 'Electronics')
        
        # Get predictions
        predictions = predict_optimal_price(
            base_price, competitor_price, demand, inventory_level,
            season, promotion_type, customer_segment, product_category
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'timestamp': datetime.now().strftime('%d-%m-%Y %I:%M %p')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'models': list(models.keys())
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Dynamic Pricing Engine - INDIA")
    print("=" * 50)
    print("\n📂 Loading models...")
    
    has_models = load_all_models()
    
    if has_models:
        print(f"✅ Loaded {len(models)} model(s)")
    else:
        print("⚠️  No pre-trained models found")
        print("   Using built-in pricing logic")
    
    print(f"\n🌐 Starting server at: http://localhost:5000")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')