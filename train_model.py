import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FILE = "gold_training_data_v2.csv"
MODEL_FILE = "gold_trend_model_v2.joblib"
CONFIDENCE_THRESHOLD = 0.60 # Strict Filter: Only trade if > 60% sure

def train_brain():
    print("📂 Loading Advanced Data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("❌ Error: Data file not found. Run data_collector_v2.py first!")
        return

    # Clean infinite values/NaNs
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # FEATURES LIST (Must match what the Bot calculates later)
    features = [
        'RSI', 'RSI_Lag1', 'RSI_Lag2',  # Momentum History
        'Dist_EMA_50', 'Dist_EMA_200',  # Trend Context
        'ADX', 'ATR',                   # Volatility
        'Hour', 'DayOfWeek',            # Time Context
        'ROC_1', 'ROC_3', 'ROC_5',      # Velocity
        'Rel_Body', 'Rel_Upper_Wick', 'Rel_Lower_Wick' # Candle Shape
    ]

    # X = Inputs, y = Output (Target Direction)
    X = df[features]
    y = df['Target_Direction']

    # Split Data (80% Train, 20% Test)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📚 Training on {len(X_train)} candles...")

    # Initialize Optimized Model
    # Increased trees (300) and depth (15) for complex patterns
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=15, 
        min_samples_leaf=5, 
        random_state=42,
        n_jobs=-1 # Use all CPU cores
    )

    # Train the Model
    model.fit(X_train, y_train)

    # Evaluate Performance
    print("🧪 Testing Advanced Knowledge...")
    probs = model.predict_proba(X_test)
    
    # Filter by High Confidence
    # probs looks like [[prob_0, prob_1], ...]
    max_probs = np.max(probs, axis=1)
    confident_indices = max_probs > CONFIDENCE_THRESHOLD
    
    y_test_filtered = y_test[confident_indices]
    preds_filtered = model.predict(X_test)[confident_indices]

    print("\n" + "="*30)
    print(f"📊 ADVANCED AI RESULTS")
    print("="*30)
    
    if len(y_test_filtered) > 0:
        acc = accuracy_score(y_test_filtered, preds_filtered)
        print(f"✅ High Confidence Accuracy (> {CONFIDENCE_THRESHOLD:.0%}): {acc:.2%}")
        print(f"📉 Trades Taken: {len(y_test_filtered)} / {len(y_test)} ({len(y_test_filtered)/len(y_test):.1%})")
        
        if acc > 0.53:
            joblib.dump(model, MODEL_FILE)
            print(f"\n💾 Super-Brain saved: {MODEL_FILE}")
            print("⚠️ NOTE: Make sure your bot is set to use 'gold_trend_model_v2.joblib'")
        else:
            print("Accuracy is too low to be profitable. Try collecting more data.")
    else:
        print("No trades met the 60% confidence threshold.")

if __name__ == "__main__":
    train_brain()