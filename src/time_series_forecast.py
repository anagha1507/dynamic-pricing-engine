"""
Time Series Forecasting for Dynamic Pricing
Uses LSTM and Temporal Fusion Transformer (TFT) for demand/price prediction
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class TimeSeriesForecaster:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 24  # 24 hours lookback
        self.forecast_horizon = 6  # Forecast next 6 hours
        
    def load_data(self):
        """Load raw data for time series"""
        print("📂 Loading time series data...")
        
        raw_path = os.path.join(self.project_root, 'data', 'raw', 'dynamic_pricing_data.csv')
        df = pd.read_csv(raw_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"✅ Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def prepare_sequences(self, data, sequence_length=24, forecast_horizon=6):
        """Prepare sequences for time series models"""
        print(f"\n🔄 Preparing sequences (lookback={sequence_length}h, forecast={forecast_horizon}h)...")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - forecast_horizon + 1):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i:i+forecast_horizon].flatten())
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/val/test
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"✅ Sequences created: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_lstm_model(self, input_shape, output_length):
        """Build LSTM model for time series forecasting"""
        print("\n🏗️  Building LSTM Model...")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_length, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ LSTM Model built!")
        model.summary()
        
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train LSTM model"""
        print("\n" + "=" * 60)
        print("🧠 TRAINING LSTM MODEL")
        print("=" * 60)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.lstm_model = self.build_lstm_model(input_shape, y_train.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n✅ LSTM Training completed!")
        
        # Plot training history
        self.plot_training_history(history)
        
        return self.lstm_model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'lstm_training_history.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def evaluate_lstm(self, X_test, y_test):
        """Evaluate LSTM model"""
        print("\n" + "=" * 60)
        print("📊 EVALUATING LSTM MODEL")
        print("=" * 60)
        
        # Predict
        predictions = self.lstm_model.predict(X_test, verbose=0)
        
        # Inverse transform
        predictions_orig = self.scaler.inverse_transform(predictions)
        y_test_orig = self.scaler.inverse_transform(y_test)
        
        # Calculate metrics for each forecast step
        print("\nMetrics per forecast hour:")
        for i in range(predictions.shape[1]):
            mae = mean_absolute_error(y_test_orig[:, i], predictions_orig[:, i])
            rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], predictions_orig[:, i]))
            print(f"  • Hour {i+1}: MAE=${mae:.2f}, RMSE=${rmse:.2f}")
        
        # Overall metrics
        mae = mean_absolute_error(y_test_orig.flatten(), predictions_orig.flatten())
        rmse = np.sqrt(mean_squared_error(y_test_orig.flatten(), predictions_orig.flatten()))
        r2 = r2_score(y_test_orig.flatten(), predictions_orig.flatten())
        
        print(f"\nOverall Metrics:")
        print(f"  • MAE: ${mae:.2f}")
        print(f"  • RMSE: ${rmse:.2f}")
        print(f"  • R²: {r2:.4f}")
        
        return predictions_orig, y_test_orig
    
    def plot_forecasts(self, y_test, predictions, n_samples=200):
        """Plot forecast vs actual"""
        print("\n📈 Plotting Forecasts vs Actual...")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot full comparison
        time_steps = range(min(n_samples, len(y_test)))
        
        # Plot actual vs predicted for first forecast step
        axes[0].plot(time_steps, y_test[time_steps, 0], 'b-', label='Actual (t+1h)', alpha=0.7, linewidth=1.5)
        axes[0].plot(time_steps, predictions[time_steps, 0], 'r--', label='Predicted (t+1h)', alpha=0.7, linewidth=1.5)
        axes[0].fill_between(time_steps, y_test[time_steps, 0], predictions[time_steps, 0], 
                            alpha=0.3, color='gray', label='Error')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title('LSTM Forecast: 1-Hour Ahead Prediction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot multi-step forecast for a single sample
        sample_idx = len(y_test) // 2  # Middle sample
        steps = range(1, y_test.shape[1] + 1)
        
        axes[1].plot(steps, y_test[sample_idx], 'bo-', label='Actual', linewidth=2, markersize=8)
        axes[1].plot(steps, predictions[sample_idx], 'rs--', label='Predicted', linewidth=2, markersize=8)
        axes[1].fill_between(steps, y_test[sample_idx], predictions[sample_idx], 
                            alpha=0.3, color='gray')
        axes[1].set_xlabel('Forecast Horizon (Hours)')
        axes[1].set_ylabel('Price ($)')
        axes[1].set_title(f'Multi-Step Forecast Comparison (Sample {sample_idx})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(steps)
        
        plt.tight_layout()
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'lstm_forecasts.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_error_distribution(self, y_test, predictions):
        """Plot error distribution"""
        errors = (predictions - y_test).flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean Error: ${np.mean(errors):.2f}')
        axes[0].set_xlabel('Prediction Error ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Forecast Error Distribution')
        axes[0].legend()
        
        # Q-Q plot (simplified)
        sorted_errors = np.sort(errors)
        theoretical = np.random.normal(0, np.std(errors), len(errors))
        theoretical = np.sort(theoretical)
        axes[1].scatter(theoretical, sorted_errors, alpha=0.5, edgecolors='k', linewidth=0.3)
        axes[1].plot([theoretical.min(), theoretical.max()], 
                    [theoretical.min(), theoretical.max()], 'r--', linewidth=2)
        axes[1].set_xlabel('Theoretical Quantiles')
        axes[1].set_ylabel('Sample Quantiles')
        axes[1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'lstm_error_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the LSTM model"""
        models_path = os.path.join(self.project_root, 'models', 'lstm_model.h5')
        self.lstm_model.save(models_path)
        
        # Save scaler
        scaler_path = os.path.join(self.project_root, 'models', 'ts_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n💾 Models saved:")
        print(f"  • LSTM: {models_path}")
        print(f"  • Scaler: {scaler_path}")
    
    def run_full_pipeline(self):
        """Run complete time series pipeline"""
        print("=" * 60)
        print("📈 STARTING TIME SERIES FORECASTING PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Use price data for forecasting
        price_data = df['price'].values
        
        # Prepare sequences
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_sequences(
            price_data, self.sequence_length, self.forecast_horizon
        )
        
        # Train LSTM
        self.lstm_model, history = self.train_lstm(X_train, y_train, X_val, y_val, epochs=30)
        
        # Evaluate
        predictions, actual = self.evaluate_lstm(X_test, y_test)
        
        # Plot results
        self.plot_forecasts(actual, predictions, n_samples=200)
        self.plot_error_distribution(actual, predictions)
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("✅ TIME SERIES PIPELINE COMPLETED!")
        print("=" * 60)
        
        return self.lstm_model, predictions, actual

if __name__ == "__main__":
    forecaster = TimeSeriesForecaster()
    model, pred, actual = forecaster.run_full_pipeline()
    
    print("\n🎯 Time Series Forecasting Complete!")
    print(f"Forecast Horizon: {forecaster.forecast_horizon} hours")
    print(f"Lookback Window: {forecaster.sequence_length} hours")