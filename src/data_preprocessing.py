"""
Data Preprocessing Pipeline for Dynamic Pricing Engine
Handles: Missing values, outliers, encoding, scaling, feature creation
"""

import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        # Get project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # If config_path is not absolute, make it relative to project root
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.project_root, config_path)
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_columns = None
        
    def load_data(self):
        """Load raw data"""
        print("📂 Loading data...")
        raw_path = os.path.join(self.project_root, self.config['data']['raw_path'])
        df = pd.read_csv(raw_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    
    def check_missing_values(self, df):
        """Check and report missing values"""
        print("\n🔍 Checking missing values...")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print("⚠️  Missing values found:")
            print(missing_df)
        else:
            print("✅ No missing values found!")
        return missing_df
    
    def handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        print("\n🔧 Handling missing values...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numerical columns with median
        if len(numerical_cols) > 0:
            df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        print("✅ Missing values handled!")
        return df
    
    def detect_outliers(self, df, column, method='iqr'):
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def handle_outliers(self, df, columns=None):
        """Handle outliers using capping method"""
        print("\n📊 Handling outliers...")
        
        if columns is None:
            columns = ['price', 'demand', 'competitor_price']
        
        for col in columns:
            if col in df.columns:
                _, lower, upper = self.detect_outliers(df, col)
                original_outliers = len(df[(df[col] < lower) | (df[col] > upper)])
                
                # Cap the outliers
                df[col] = df[col].clip(lower, upper)
                
                print(f"  • {col}: Capped {original_outliers} outliers (range: {lower:.2f} - {upper:.2f})")
        
        print("✅ Outliers handled!")
        return df
    
    def create_time_features(self, df):
        """Create time-based features for time series analysis"""
        print("\n🕐 Creating time features...")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        print("✅ Time features created with cyclical encoding!")
        return df
    
    def create_lag_features(self, df, target_col='price', lags=[1, 3, 6, 12, 24]):
        """Create lag features for time series forecasting"""
        print("\n⏮️  Creating lag features...")
        
        df = df.sort_values('timestamp')
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby('product_category')[target_col].shift(lag)
            df[f'demand_lag_{lag}'] = df.groupby('product_category')['demand'].shift(lag)
        
        # Fill NaN values created by lagging
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✅ Created lag features for lags: {lags}")
        return df
    
    def create_rolling_features(self, df, windows=[6, 12, 24]):
        """Create rolling window features"""
        print("\n📈 Creating rolling features...")
        
        df = df.sort_values('timestamp')
        
        for window in windows:
            # Rolling mean
            df[f'price_rolling_mean_{window}h'] = df.groupby('product_category')['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'demand_rolling_mean_{window}h'] = df.groupby('product_category')['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'price_rolling_std_{window}h'] = df.groupby('product_category')['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min and max
            df[f'price_rolling_min_{window}h'] = df.groupby('product_category')['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'price_rolling_max_{window}h'] = df.groupby('product_category')['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        # Fill NaN values
        df = df.fillna(0)
        
        print(f"✅ Created rolling features for windows: {windows}")
        return df
    
    def create_price_features(self, df):
        """Create price-related features"""
        print("\n💰 Creating price features...")
        
        # Price differences
        df['price_vs_competitor'] = df['price'] - df['competitor_price']
        df['price_vs_competitor_pct'] = (df['price'] - df['competitor_price']) / df['competitor_price'] * 100
        
        # Price margin
        df['price_margin'] = df['price'] - df['base_price']
        df['price_margin_pct'] = (df['price'] - df['base_price']) / df['base_price'] * 100
        
        # Price changes
        df['price_change'] = df.groupby('product_category')['price'].diff()
        df['price_change_pct'] = df.groupby('product_category')['price'].pct_change() * 100
        df['price_change'] = df['price_change'].fillna(0)
        df['price_change_pct'] = df['price_change_pct'].fillna(0)
        
        # Price volatility
        df['price_volatility'] = df.groupby('product_category')['price_change'].transform('std')
        df['price_volatility'] = df['price_volatility'].fillna(0)
        
        print("✅ Price features created!")
        return df
    
    def create_demand_features(self, df):
        """Create demand-related features"""
        print("\n📊 Creating demand features...")
        
        # Demand intensity
        df['demand_intensity'] = df['demand'] / (df['inventory_level'] + 1)
        
        # Supply-demand ratio
        df['supply_demand_ratio'] = df['inventory_level'] / (df['demand'] + 1)
        
        # Demand trend
        df['demand_trend'] = df.groupby('product_category')['demand'].diff()
        df['demand_trend'] = df['demand_trend'].fillna(0)
        
        # Demand acceleration
        df['demand_acceleration'] = df.groupby('product_category')['demand_trend'].diff()
        df['demand_acceleration'] = df['demand_acceleration'].fillna(0)
        
        print("✅ Demand features created!")
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        print("\n🔗 Creating interaction features...")
        
        # Price and demand interactions
        df['price_demand_ratio'] = df['price'] / (df['demand'] + 1)
        df['revenue_potential'] = df['price'] * df['demand']
        
        # Time and price interactions
        df['peak_hour_price'] = df['price'] * df['is_weekend']
        df['holiday_season_price'] = df['price'] * df['is_holiday_season']
        
        # Customer and price interactions
        df['customer_value'] = df['customer_rating'] * df['conversion_rate']
        
        print("✅ Interaction features created!")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        print("\n🔄 Encoding categorical features...")
        
        # Label encoding for ordinal categories
        label_encode_cols = ['season', 'customer_segment']
        for col in label_encode_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                print(f"  • Label encoded: {col}")
        
        # One-hot encoding for nominal categories
        onehot_cols = ['product_category', 'promotion_type']
        if len(onehot_cols) > 0:
            onehot_data = self.onehot_encoder.fit_transform(df[onehot_cols])
            onehot_columns = self.onehot_encoder.get_feature_names_out(onehot_cols)
            onehot_df = pd.DataFrame(onehot_data, columns=onehot_columns, index=df.index)
            df = pd.concat([df, onehot_df], axis=1)
            print(f"  • One-hot encoded: {onehot_cols}")
        
        print("✅ Categorical features encoded!")
        return df
    
    def scale_numerical_features(self, df, columns=None):
        """Scale numerical features"""
        print("\n📏 Scaling numerical features...")
        
        if columns is None:
            columns = ['price', 'competitor_price', 'demand', 'inventory_level',
                      'customer_rating', 'shipping_time', 'conversion_rate',
                      'price_vs_competitor', 'price_margin', 'price_volatility']
        
        # Filter only existing columns
        scale_cols = [col for col in columns if col in df.columns]
        
        if len(scale_cols) > 0:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
            print(f"  • Scaled {len(scale_cols)} features")
        
        # Save scaler
        models_path = os.path.join(self.project_root, 'models', 'scaler.pkl')
        joblib.dump(self.scaler, models_path)
        print("✅ Features scaled and scaler saved to models/scaler.pkl")
        return df
    
    def select_features_for_modeling(self, df, target='optimal_price'):
        """Select relevant features for modeling"""
        print("\n🎯 Selecting features for modeling...")
        
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'optimal_price', 'base_price']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove original categorical columns (keep encoded versions)
        original_cats = ['product_category', 'promotion_type', 'season', 'customer_segment']
        feature_cols = [col for col in feature_cols if col not in original_cats]
        
        print(f"  • Selected {len(feature_cols)} features")
        print(f"  • Target variable: {target}")
        
        self.feature_columns = feature_cols
        
        # Save feature columns
        models_path = os.path.join(self.project_root, 'models', 'feature_columns.pkl')
        joblib.dump(feature_cols, models_path)
        
        return df, feature_cols, target
    
    def split_data(self, df, feature_cols, target):
        """Split data into train, validation, and test sets"""
        print("\n✂️  Splitting data...")
        
        X = df[feature_cols]
        y = df[target]
        
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        print(f"  • Train set: {X_train.shape[0]} samples")
        print(f"  • Validation set: {X_val.shape[0]} samples")
        print(f"  • Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed data"""
        print("\n💾 Saving processed data...")
        
        processed_path = os.path.join(self.project_root, 'data', 'processed')
        os.makedirs(processed_path, exist_ok=True)
        
        # Create DataFrames for saving
        train_df = X_train.copy()
        train_df['target'] = y_train
        
        val_df = X_val.copy()
        val_df['target'] = y_val
        
        test_df = X_test.copy()
        test_df['target'] = y_test
        
        # Save to CSV
        train_df.to_csv(os.path.join(processed_path, 'train_data.csv'), index=False)
        val_df.to_csv(os.path.join(processed_path, 'val_data.csv'), index=False)
        test_df.to_csv(os.path.join(processed_path, 'test_data.csv'), index=False)
        
        print("✅ Data saved to data/processed/")
        
    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("=" * 60)
        print("🚀 STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Check missing values
        self.check_missing_values(df)
        
        # 3. Handle missing values (if any)
        df = self.handle_missing_values(df)
        
        # 4. Handle outliers
        df = self.handle_outliers(df)
        
        # 5. Create all features
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_price_features(df)
        df = self.create_demand_features(df)
        df = self.create_interaction_features(df)
        
        # 6. Encode categorical features
        df = self.encode_categorical_features(df)
        
        # 7. Handle any remaining missing values from feature creation
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # 8. Scale numerical features
        df = self.scale_numerical_features(df)
        
        # 9. Select features
        df, feature_cols, target = self.select_features_for_modeling(df)
        
        # 10. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df, feature_cols, target
        )
        
        # 11. Save processed data
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run full pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_full_pipeline()
    
    print("\n📊 Final Dataset Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"\nFeature count: {X_train.shape[1]}")