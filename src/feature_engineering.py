"""
Advanced Feature Engineering for Dynamic Pricing
Creates specialized features for each model type
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.selected_features = None
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        
    def create_rf_features(self, df):
        """Create features optimized for Random Forest"""
        print("🌲 Creating Random Forest specific features...")
        
        # Tree-based features
        if 'product_category' in df.columns:
            df['price_percentile'] = df.groupby('product_category')['price'].transform(
                lambda x: x.rank(pct=True)
            )
            df['demand_percentile'] = df.groupby('product_category')['demand'].transform(
                lambda x: x.rank(pct=True)
            )
        else:
            # Use one-hot encoded columns
            df['price_percentile'] = df['price'].rank(pct=True)
            df['demand_percentile'] = df['demand'].rank(pct=True)
        
        # Binned features (Random Forest handles these well)
        try:
            df['price_bin'] = pd.qcut(df['price'], q=10, labels=False, duplicates='drop')
            df['demand_bin'] = pd.qcut(df['demand'], q=10, labels=False, duplicates='drop')
        except:
            pass
        
        if 'hour_of_day' in df.columns:
            df['hour_bin'] = pd.cut(df['hour_of_day'], bins=[0,6,12,18,24], labels=False)
        
        return df
    
    def create_xgboost_features(self, df):
        """Create features optimized for XGBoost"""
        print("⚡ Creating XGBoost specific features...")
        
        # Gradient boosting loves these
        df['price_demand_interaction'] = df['price'] * df['demand']
        df['price_competitor_ratio'] = df['price'] / (df['competitor_price'] + 0.01)
        df['demand_supply_ratio'] = df['demand'] / (df['inventory_level'] + 1)
        
        # Polynomial features for non-linearity
        df['price_squared'] = df['price'] ** 2
        df['demand_squared'] = df['demand'] ** 2
        
        return df
    
    def create_rl_features(self, df):
        """Create state features for Reinforcement Learning"""
        print("🤖 Creating RL state features...")
        
        # State representation features
        df['inventory_health'] = df['inventory_level'] / (df['demand'].rolling(24).mean() + 1)
        df['price_competitiveness'] = (df['competitor_price'] - df['price']) / df['price']
        df['demand_momentum'] = df['demand'].diff(3).fillna(0)
        df['price_momentum'] = df['price'].diff(3).fillna(0)
        
        # Normalized state features (0-1 range)
        denom = df['inventory_level'].max() - df['inventory_level'].min() + 1
        df['norm_inventory'] = (df['inventory_level'] - df['inventory_level'].min()) / denom
        
        denom2 = df['demand'].max() - df['demand'].min() + 1
        df['norm_demand'] = (df['demand'] - df['demand'].min()) / denom2
        
        return df
    
    def create_timeseries_features(self, df):
        """Create features for Time Series models"""
        print("📈 Creating Time Series specific features...")
        
        # Exponential moving averages
        if 'product_category' in df.columns:
            df['ema_price_6'] = df.groupby('product_category')['price'].transform(
                lambda x: x.ewm(span=6, adjust=False).mean()
            )
            df['ema_price_12'] = df.groupby('product_category')['price'].transform(
                lambda x: x.ewm(span=12, adjust=False).mean()
            )
            df['ema_price_24'] = df.groupby('product_category')['price'].transform(
                lambda x: x.ewm(span=24, adjust=False).mean()
            )
        else:
            df['ema_price_6'] = df['price'].ewm(span=6, adjust=False).mean()
            df['ema_price_12'] = df['price'].ewm(span=12, adjust=False).mean()
            df['ema_price_24'] = df['price'].ewm(span=24, adjust=False).mean()
        
        # MACD-like features
        df['price_macd'] = df['ema_price_12'] - df['ema_price_24']
        df['price_macd_signal'] = df['price_macd'].ewm(span=9, adjust=False).mean()
        
        # Rate of change features
        for period in [1, 6, 12, 24]:
            df[f'price_roc_{period}'] = df['price'].pct_change(periods=period).fillna(0)
            df[f'demand_roc_{period}'] = df['demand'].pct_change(periods=period).fillna(0)
        
        return df
    
    def select_best_features(self, X, y, k=50):
        """Select top k features using mutual information"""
        print(f"\n🎯 Selecting top {k} features using Mutual Information...")
        
        # Handle any remaining NaN or infinity
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        y = y.fillna(y.mean())
        
        k = min(k, X.shape[1])
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X, y)
        
        # Get selected feature names
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selected_features = feature_scores.head(k)['feature'].tolist()
        
        print("Top 10 features by Mutual Information:")
        print(feature_scores.head(10).to_string(index=False))
        
        return self.selected_features
    
    def apply_pca(self, X, n_components=None):
        """Apply PCA for dimensionality reduction"""
        print(f"\n🔍 Applying PCA (keeping 95% variance)...")
        
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X_pca = self.pca.fit_transform(X)
        
        print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} features")
        print(f"Explained variance ratio sum: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca
    
    def create_all_specialized_features(self, df):
        """Create all specialized features"""
        print("\n" + "=" * 60)
        print("🔧 CREATING SPECIALIZED FEATURES")
        print("=" * 60)
        
        df = self.create_rf_features(df)
        df = self.create_xgboost_features(df)
        df = self.create_rl_features(df)
        df = self.create_timeseries_features(df)
        
        print("\n✅ All specialized features created!")
        return df

if __name__ == "__main__":
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load processed training data
    train_path = os.path.join(project_root, 'data', 'processed', 'train_data.csv')
    train_df = pd.read_csv(train_path)
    
    print(f"📂 Loaded training data: {train_df.shape}")
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Create specialized features
    train_df = engineer.create_all_specialized_features(train_df)
    
    # Separate features and target
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    # Select best features
    selected = engineer.select_best_features(X, y, k=50)
    
    print(f"\n✅ Selected {len(selected)} best features")
    print("\n📋 Selected features:")
    for i, feat in enumerate(selected[:20], 1):
        print(f"  {i}. {feat}")