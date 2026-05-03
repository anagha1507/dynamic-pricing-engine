import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_ecommerce_data(n_samples=10000):
    """Generate synthetic e-commerce data for dynamic pricing"""
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Product categories and base prices
    categories = {
        'Electronics': {'base_price': 500, 'price_std': 150},
        'Clothing': {'base_price': 50, 'price_std': 20},
        'Home & Garden': {'base_price': 100, 'price_std': 40},
        'Sports': {'base_price': 75, 'price_std': 30},
        'Books': {'base_price': 25, 'price_std': 10}
    }
    
    data = []
    
    for i, date in enumerate(dates):
        category = np.random.choice(list(categories.keys()))
        cat_info = categories[category]
        
        # Generate base price
        base_price = np.random.normal(cat_info['base_price'], cat_info['price_std'])
        base_price = max(10, base_price)
        
        # Competitor price (usually within 20% of our price)
        competitor_price = base_price * np.random.normal(1.0, 0.15)
        
        # Seasonal factors
        month = date.month
        hour = date.hour
        day_of_week = date.weekday()
        
        # Season multiplier
        if month in [11, 12]:  # Holiday season
            season_mult = np.random.normal(1.3, 0.1)
        elif month in [6, 7, 8]:  # Summer
            season_mult = np.random.normal(0.9, 0.1)
        else:
            season_mult = 1.0
        
        # Time of day multiplier
        if 9 <= hour <= 17:  # Business hours
            time_mult = np.random.normal(1.2, 0.1)
        elif 18 <= hour <= 22:  # Evening peak
            time_mult = np.random.normal(1.4, 0.15)
        else:  # Night
            time_mult = np.random.normal(0.6, 0.1)
        
        # Day of week multiplier
        if day_of_week >= 5:  # Weekend
            day_mult = np.random.normal(1.3, 0.1)
        else:
            day_mult = np.random.normal(1.0, 0.1)
        
        # Generate demand
        price_elasticity = -1.5
        demand = (
            np.random.poisson(100) * 
            (base_price ** price_elasticity) * 
            season_mult * time_mult * day_mult *
            np.random.normal(1, 0.1)
        )
        demand = max(0, int(demand))
        
        # Inventory level
        inventory_level = np.random.randint(0, 1000)
        
        # Customer metrics
        customer_rating = np.random.normal(4.2, 0.5)
        customer_rating = max(1, min(5, customer_rating))
        
        # Promotion
        promotion_type = np.random.choice(
            ['None', 'Percentage', 'Fixed', 'BOGO'],
            p=[0.5, 0.2, 0.2, 0.1]
        )
        
        # Optimal price
        optimal_price = competitor_price * np.random.normal(1.0, 0.1)
        
        # Historical price (renamed to 'price' to match preprocessing)
        price = optimal_price * np.random.normal(1.0, 0.05)
        
        data.append({
            'timestamp': date,
            'product_category': category,
            'base_price': base_price,
            'competitor_price': competitor_price,
            'price': price,  # Changed from 'historical_price' to 'price'
            'optimal_price': optimal_price,
            'demand': demand,
            'inventory_level': inventory_level,
            'season': get_season(month),
            'day_of_week': day_of_week,
            'hour_of_day': hour,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_holiday_season': 1 if month in [11, 12] else 0,
            'promotion_type': promotion_type,
            'customer_rating': customer_rating,
            'shipping_time': np.random.normal(3, 1),
            'conversion_rate': np.random.beta(2, 5),
            'customer_segment': np.random.choice(
                ['Budget', 'Mid-range', 'Premium'],
                p=[0.3, 0.5, 0.2]
            )
        })
    
    df = pd.DataFrame(data)
    return df

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

if __name__ == "__main__":
    # Get script directory and go to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Change to project root
    os.chdir(project_root)
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate data
    print("🔄 Generating synthetic e-commerce data...")
    df = generate_ecommerce_data(10000)
    
    # Save to correct path
    save_path = 'data/raw/dynamic_pricing_data.csv'
    df.to_csv(save_path, index=False)
    
    print(f"✅ Generated {len(df)} samples")
    print(f"📁 Saved to: {save_path}")
    print(f"📂 Current directory: {os.getcwd()}")
    print("\n📊 Data Preview:")
    print(df.head())
    print("\n📋 Data Info:")
    print(df.info())