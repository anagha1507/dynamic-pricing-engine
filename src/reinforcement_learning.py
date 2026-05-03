"""
Reinforcement Learning for Dynamic Pricing
Uses PPO (Proximal Policy Optimization) to learn optimal pricing strategies
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
warnings.filterwarnings('ignore')

class PricingEnvironment(gym.Env):
    """Custom Gym Environment for Dynamic Pricing"""
    
    def __init__(self, data, feature_cols, price_col='price', demand_col='demand', 
                 competitor_price_col='competitor_price', optimal_price_col='optimal_price'):
        super(PricingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.demand_col = demand_col
        self.competitor_price_col = competitor_price_col
        self.optimal_price_col = optimal_price_col
        
        # Action space: Price adjustment percentage (-20% to +20%)
        self.action_space = spaces.Box(
            low=-0.20,  # Max 20% decrease
            high=0.20,  # Max 20% increase
            shape=(1,),
            dtype=np.float32
        )
        
        # State space: All features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(feature_cols),),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(data) - 1
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        state = self.data.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        return state, {}
    
    def step(self, action):
        """Take an action and return next state, reward, done, truncated, info"""
        
        # Get current row
        current_data = self.data.iloc[self.current_step]
        current_price = current_data[self.price_col]
        competitor_price = current_data[self.competitor_price_col]
        demand = current_data[self.demand_col]
        optimal_price = current_data[self.optimal_price_col]
        
        # Apply price adjustment
        price_adjustment = float(action[0])
        new_price = current_price * (1 + price_adjustment)
        
        # Simulate demand response to price change
        price_elasticity = -1.5  # Negative: higher price = lower demand
        demand_change = price_elasticity * price_adjustment
        new_demand = demand * (1 + demand_change)
        new_demand = max(0, new_demand)
        
        # Calculate reward
        reward = self._calculate_reward(new_price, competitor_price, new_demand, 
                                        current_price, optimal_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Get next state
        if not done:
            next_state = self.data.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        else:
            next_state = np.zeros(len(self.feature_cols), dtype=np.float32)
        
        info = {
            'current_price': current_price,
            'new_price': new_price,
            'price_adjustment': price_adjustment,
            'demand': demand,
            'new_demand': new_demand,
            'reward': reward
        }
        
        return next_state, reward, done, truncated, info
    
    def _calculate_reward(self, new_price, competitor_price, new_demand, 
                          current_price, optimal_price):
        """Calculate reward based on multiple business objectives"""
        
        # 1. Revenue component
        revenue = new_price * new_demand
        
        # 2. Profit margin component
        profit_margin = (new_price - current_price * 0.7) / new_price  # Assuming 30% cost
        
        # 3. Competitiveness component
        if competitor_price > 0:
            price_ratio = new_price / competitor_price
            if 0.9 <= price_ratio <= 1.1:  # Within 10% of competitor
                competitiveness_bonus = 1.0
            elif price_ratio < 0.9:  # Much cheaper (losing profit)
                competitiveness_bonus = 0.5
            else:  # Much more expensive (losing customers)
                competitiveness_bonus = -0.5
        else:
            competitiveness_bonus = 0
        
        # 4. Demand impact
        demand_bonus = np.log1p(new_demand) - np.log1p(new_demand * 0.8)
        
        # 5. Deviation from optimal price penalty
        if optimal_price > 0:
            price_deviation = abs(new_price - optimal_price) / optimal_price
            optimality_penalty = -2.0 * price_deviation
        else:
            optimality_penalty = 0
        
        # Combined reward
        reward = (
            0.3 * np.log1p(revenue) +  # Revenue (log scale to prevent domination)
            0.2 * profit_margin +       # Profit margin
            0.2 * competitiveness_bonus + # Competitiveness
            0.15 * demand_bonus +       # Demand
            0.15 * optimality_penalty   # Proximity to optimal
        )
        
        return float(reward)

class RLPricingModel:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model = None
        self.env = None
        self.state_features = None
        self.training_history = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for RL"""
        print("📂 Loading data for RL...")
        
        processed_path = os.path.join(self.project_root, 'data', 'processed')
        train_df = pd.read_csv(os.path.join(processed_path, 'train_data.csv'))
        
        # Add target column and rename to optimal_price for RL env
        train_df['optimal_price'] = train_df['target']
        
        # Select key features for state representation (excluding price-related cols)
        feature_candidates = [
            'inventory_level', 'customer_rating', 'conversion_rate', 
            'is_weekend', 'is_holiday_season', 'price_vs_competitor', 
            'price_margin', 'demand_intensity', 'supply_demand_ratio',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'price_change', 'demand_trend'
        ]
        
        # Filter only available columns
        self.state_features = [col for col in feature_candidates if col in train_df.columns]
        
        print(f"  • Available state features: {len(self.state_features)}")
        print(f"  • First 10 features: {self.state_features[:10]}")
        
        # Keep only needed columns for environment
        needed_cols = self.state_features + ['target', 'optimal_price', 'price', 'demand', 'competitor_price']
        needed_cols = list(set(needed_cols))
        needed_cols = [col for col in needed_cols if col in train_df.columns]
        
        self.train_data = train_df[needed_cols].copy()
        
        print(f"✅ Prepared {len(self.train_data)} training samples")
        print(f"  • State features for RL: {len(self.state_features)}")
        
        return self.train_data
    
    def create_environment(self):
        """Create the pricing environment"""
        print("\n🎮 Creating Pricing Environment...")
        
        self.env = PricingEnvironment(
            data=self.train_data,
            feature_cols=self.state_features
        )
        
        print("✅ Environment created!")
        print(f"  • Action space: {self.env.action_space}")
        print(f"  • Observation space: {self.env.observation_space.shape}")
        
        return self.env
    
    def train_ppo_model(self, total_timesteps=50000):
        """Train PPO model"""
        print("\n" + "=" * 60)
        print("🤖 TRAINING PPO MODEL FOR DYNAMIC PRICING")
        print("=" * 60)
        
        # Wrap environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=None,
            policy_kwargs=dict(net_arch=[128, 128, 64]),
            verbose=1,
            seed=42,
            device='cpu'
        )
        
        print(f"\nTraining for {total_timesteps} timesteps...")
        print("This may take a few minutes ⏳")
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        print("\n✅ PPO Model trained successfully!")
        
        return self.model
    
    def evaluate_rl_model(self, n_episodes=10):
        """Evaluate the RL model"""
        print("\n" + "=" * 60)
        print("📊 EVALUATING RL MODEL")
        print("=" * 60)
        
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            vec_env, 
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        
        print(f"  • Mean Reward: {mean_reward:.2f}")
        print(f"  • Std Reward: {std_reward:.2f}")
        
        return mean_reward, std_reward
    
    def simulate_pricing_strategy(self, n_steps=100):
        """Simulate pricing strategy and visualize"""
        print("\n📈 Simulating Pricing Strategy...")
        
        vec_env = DummyVecEnv([lambda: self.env])
        obs = vec_env.reset()
        
        prices = []
        adjustments = []
        rewards = []
        
        for i in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            prices.append(info[0].get('new_price', 0))
            adjustments.append(info[0].get('price_adjustment', 0) * 100)
            rewards.append(info[0].get('reward', 0))
            
            if done:
                break
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price evolution
        axes[0, 0].plot(prices, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price Evolution Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price adjustments
        bar_colors = ['g' if a >= 0 else 'r' for a in adjustments]
        axes[0, 1].bar(range(len(adjustments)), adjustments, color=bar_colors)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Adjustment (%)')
        axes[0, 1].set_title('Price Adjustments (Green=Increase, Red=Decrease)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative reward
        cum_rewards = np.cumsum(rewards)
        axes[1, 0].plot(cum_rewards, 'g-', linewidth=2)
        axes[1, 0].fill_between(range(len(cum_rewards)), 0, cum_rewards, alpha=0.3)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].set_title('Cumulative Reward Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[1, 1].hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].axvline(x=np.mean(rewards), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].legend()
        
        plt.suptitle('Reinforcement Learning Pricing Strategy Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        models_dir = os.path.join(self.project_root, 'models')
        plt.savefig(os.path.join(models_dir, 'rl_pricing_strategy.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n📊 RL Strategy Summary:")
        print(f"  • Average Price: ${np.mean(prices):.2f}")
        print(f"  • Average Adjustment: {np.mean(adjustments):.2f}%")
        print(f"  • Total Reward: {sum(rewards):.2f}")
        print(f"  • Average Reward: {np.mean(rewards):.2f}")
        
        return prices, adjustments, rewards
    
    def compare_with_traditional_models(self, X_test, y_test):
        """Compare RL pricing with traditional ML models"""
        print("\n" + "=" * 60)
        print("🔄 COMPARING RL WITH TRADITIONAL MODELS")
        print("=" * 60)
        
        # Load traditional models if available
        models_dir = os.path.join(self.project_root, 'models')
        rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
        xgb_path = os.path.join(models_dir, 'xgboost_model.pkl')
        
        models = {}
        if os.path.exists(rf_path):
            models['Random Forest'] = joblib.load(rf_path)
        if os.path.exists(xgb_path):
            models['XGBoost'] = joblib.load(xgb_path)
        
        if not models:
            print("⚠️  No traditional models found for comparison.")
            return
        
        # Get RL predictions on test set
        rl_prices = []
        for i in range(min(100, len(X_test))):
            # Use state_features for RL prediction
            available_features = [f for f in self.state_features if f in X_test.columns]
            state = X_test.iloc[i][available_features].values.astype(np.float32)
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            action, _ = self.model.predict(state, deterministic=True)
            current_price = X_test.iloc[i]['price'] if 'price' in X_test.columns else y_test.iloc[i]
            rl_price = current_price * (1 + float(action[0]))
            rl_prices.append(rl_price)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = {'Random Forest': 'green', 'XGBoost': 'blue', 'RL (PPO)': 'red'}
        
        # Scatter comparison
        axes[0].scatter(y_test[:100], y_test[:100], alpha=0.3, label='Optimal', color='black', s=30)
        for name, model in models.items():
            pred = model.predict(X_test[:100])
            axes[0].scatter(y_test[:100], pred, alpha=0.5, label=name, color=colors.get(name, 'gray'), s=30)
        axes[0].scatter(y_test[:100], rl_prices, alpha=0.7, label='RL (PPO)', color='red', marker='^', s=50)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Optimal Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Model Comparison: RL vs Traditional ML')
        axes[0].legend()
        
        # Error comparison
        error_data = {}
        for name, model in models.items():
            pred = model.predict(X_test[:100])
            error_data[name] = np.abs(y_test[:100] - pred)
        error_data['RL (PPO)'] = np.abs(y_test[:100] - rl_prices)
        
        axes[1].boxplot(error_data.values(), labels=error_data.keys())
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title('Error Distribution Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, 'rl_vs_traditional.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print metrics
        print("\n📊 Pricing Performance Comparison:")
        for name, model in models.items():
            pred = model.predict(X_test[:100])
            mae = np.mean(np.abs(y_test[:100] - pred))
            print(f"  • {name} MAE: ${mae:.2f}")
        rl_mae = np.mean(np.abs(y_test[:100] - rl_prices))
        print(f"  • RL (PPO) MAE: ${rl_mae:.2f}")
    
    def save_model(self):
        """Save the RL model"""
        models_path = os.path.join(self.project_root, 'models', 'rl_pricing_model')
        self.model.save(models_path)
        print(f"\n💾 Model saved to: {models_path}")
    
    def run_full_pipeline(self):
        """Run complete RL pipeline"""
        # Load data
        self.load_and_prepare_data()
        
        # Create environment
        self.create_environment()
        
        # Train PPO
        self.train_ppo_model(total_timesteps=50000)
        
        # Evaluate
        self.evaluate_rl_model(n_episodes=10)
        
        # Simulate pricing strategy
        self.simulate_pricing_strategy(n_steps=100)
        
        # Load test data for comparison
        processed_path = os.path.join(self.project_root, 'data', 'processed')
        test_df = pd.read_csv(os.path.join(processed_path, 'test_data.csv'))
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        # Compare with traditional models
        self.compare_with_traditional_models(X_test, y_test)
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("✅ RL PIPELINE COMPLETED!")
        print("=" * 60)

if __name__ == "__main__":
    rl_model = RLPricingModel()
    rl_model.run_full_pipeline()