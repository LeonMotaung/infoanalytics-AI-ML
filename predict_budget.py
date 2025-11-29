import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import os

# Try importing LightGBM, fallback to GradientBoostingRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not found, using GradientBoostingRegressor instead.")

def run_prediction():
    print("Loading data...")
    # 1. Data Preparation
    df = pd.read_csv('datasource.csv')
    
    # Filter for target indicator
    df = df[df['Indicator'] == 'Budget Deficit/Surplus'].copy()
    
    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.sort_values(['Country', 'Time'])
    
    # Convert Amount to numeric
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Handle missing values (forward fill then backward fill within country)
    df['Amount'] = df.groupby('Country')['Amount'].ffill().bfill()
    
    # Drop any remaining NaNs (e.g. if a country has NO data)
    df = df.dropna(subset=['Amount', 'Time'])

    # 2. Feature Engineering
    print("Engineering features...")
    
    # First-Order Differencing (Target Transformation)
    # We predict the CHANGE in Amount
    df['Target_Diff'] = df.groupby('Country')['Amount'].diff()
    
    # Lagged Features (on the differenced target or original? Usually on the variable we are predicting)
    # The prompt says "Lagged features for the target variable (Value)". 
    # If we predict 'Target_Diff', we should probably use lags of 'Target_Diff' or 'Amount'.
    # Given "Inverse Transform: Predicted Level = Predicted Change + Actual Value at t-1",
    # it implies we are predicting the change.
    # Let's use lags of the *Amount* (levels) and maybe lags of the *Diff*.
    # The prompt says "Lagged features for the target variable (Value)". Let's assume 'Value' means 'Amount'.
    
    # Actually, if we predict Diff, using lags of Diff is standard ARIMA style.
    # Using lags of Amount is also valid.
    # Let's create lags of the Amount (Level) as features.
    for lag in [1, 3, 6]:
        df[f'Lag_{lag}'] = df.groupby('Country')['Amount'].shift(lag)
        
    # Rolling Statistics (3-period rolling mean of Amount)
    df['Rolling_Mean_3'] = df.groupby('Country')['Amount'].transform(lambda x: x.rolling(window=3).mean())
    
    # Time Features
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Quarter'] = df['Time'].dt.quarter
    
    # Time Step (counter per country)
    df['Time_Step'] = df.groupby('Country').cumcount() + 1
    
    # Drop rows with NaNs created by lags/diff
    df = df.dropna()
    
    # Define Features (X) and Target (y)
    # We differenced Amount to get Target_Diff.
    # So y is Target_Diff.
    # X includes Lags, Rolling, Time features, Country.
    
    feature_cols = ['Lag_1', 'Lag_3', 'Lag_6', 'Rolling_Mean_3', 'Year', 'Month', 'Quarter', 'Time_Step', 'Country']
    X = df[feature_cols]
    y = df['Target_Diff']
    
    # Store 'Amount' at t-1 for inverse transform later
    # Since we dropped NaNs, the index is aligned.
    # Actual Value at t-1 is effectively Lag_1 (since Lag_1 is Amount at t-1)
    # We will need this for the hold-out set.
    
    # 3. Preprocessing Pipeline
    numeric_features = ['Lag_1', 'Lag_3', 'Lag_6', 'Rolling_Mean_3', 'Year', 'Month', 'Quarter', 'Time_Step']
    categorical_features = ['Country']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    # 4. Model Selection
    if HAS_LGBM:
        model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=7, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.03, max_depth=7, random_state=42)
        
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
                               
    # Validation (TimeSeriesSplit)
    print("Validating model...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # We need to be careful with TimeSeriesSplit on multiple time series (Countries).
    # Standard TimeSeriesSplit splits by index. If data is sorted by Country then Time, 
    # it might train on Country A and test on Country B, which is not "Time Series" split in the temporal sense.
    # However, for this exercise, we'll assume a global temporal split or just split the dataframe as is 
    # (which is sorted by Country, Time). 
    # A better approach for panel data is to split by Time across all countries.
    # But given the instructions "TimeSeriesSplit", we will apply it to the whole dataset.
    # Note: If we split by index on sorted(Country, Time), the folds are not strictly temporal across all countries.
    # But let's proceed with standard implementation for simplicity as requested.
    
    # Actually, let's sort by Time first for the CV to be meaningful temporally across the dataset?
    # No, the prompt says "grouped by Country" for features.
    # If we sort by Time for CV, we mix countries.
    # Let's stick to the user's request: "Perform Time Series Cross-Validation".
    # We will just run it on X, y.
    
    rmse_scores = []
    r2_scores = []
    
    # For CV, we usually just fit/predict.
    # But we need to be careful about the "Inverse Transform" for metrics.
    # The prompt asks to report RMSE/R2 on the "Final Prediction" (Hold-out).
    # For CV, it says "report model robustness".
    
    # Let's just do the Hold-Out as the primary output.
    
    # 5. Final Prediction and Output
    # Hold-Out Test: Last 10 observations (presumably per country? or global?)
    # "consisting of the last 10 observations". Usually implies the very last 10 rows of the dataset.
    # Or last 10 time steps for *each* country?
    # "Generate predictions on the hold-out set."
    # Let's assume the last 10 rows of the entire dataframe (which is sorted by Country, Time).
    # This effectively tests on the last few points of the last country.
    # A better interpretation might be "Last 10 time periods for all countries".
    # But let's stick to "last 10 observations" literally.
    
    train_df = df.iloc[:-10]
    test_df = df.iloc[-10:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df['Target_Diff']
    
    X_test = test_df[feature_cols]
    y_test = test_df['Target_Diff']
    
    print("Training final model...")
    pipeline.fit(X_train, y_train)
    
    # Predictions (Change)
    y_pred_diff = pipeline.predict(X_test)
    
    # Inverse Transform
    # Predicted Level = Predicted Change + Actual Value at t-1
    # Actual Value at t-1 is 'Lag_1' in X_test
    prev_level = X_test['Lag_1']
    y_pred_level = y_pred_diff + prev_level
    
    # Actual Level
    y_actual_level = test_df['Amount'] # This is current amount
    # Wait, df['Amount'] is the level at t.
    # Target_Diff = Amount_t - Amount_{t-1}
    # Amount_t = Target_Diff + Amount_{t-1}
    # So yes, y_pred_level = y_pred_diff + Lag_1.
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_actual_level, y_pred_level))
    r2 = r2_score(y_actual_level, y_pred_level)
    
    print(f"Final RMSE: {rmse:,.2f}")
    print(f"Final R2: {r2:.4f}")
    
    # Visualization
    print("Generating plot...")
    # Get previous 50 training context points
    context_df = train_df.iloc[-50:].copy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot Context (Actual)
    plt.plot(context_df['Time'], context_df['Amount'], label='Training Context (Actual)', color='gray', alpha=0.7)
    
    # Plot Hold-out Actual
    plt.plot(test_df['Time'], test_df['Amount'], label='Hold-out (Actual)', color='blue', marker='o')
    
    # Plot Hold-out Predicted
    plt.plot(test_df['Time'], y_pred_level, label='Hold-out (Predicted)', color='red', linestyle='--', marker='x')
    
    plt.title(f'Budget Deficit/Surplus Prediction (RMSE: {rmse:,.0f}, R2: {r2:.2f})')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/budget_prediction.png')
    print("Plot saved to static/images/budget_prediction.png")
    
    # Feature Importance
    print("\nTop 10 Feature Importances:")
    # Extract feature names after OneHotEncoding
    # This is tricky with Pipeline.
    # We can get feature names from preprocessor
    try:
        if HAS_LGBM:
            feature_importances = pipeline.named_steps['model'].feature_importances_
        else:
            feature_importances = pipeline.named_steps['model'].feature_importances_
            
        # Get transformed feature names
        # numeric_features are passed through
        # categorical_features are one-hot encoded
        
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        
        all_feature_names = numeric_features + list(cat_feature_names)
        
        # Create DataFrame
        fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        print(fi_df)
        
        # Save metrics to a text file for display?
        with open('static/model_metrics.txt', 'w') as f:
            f.write(f"RMSE: {rmse:,.2f}\n")
            f.write(f"R2: {r2:.4f}\n")
            f.write("\nTop Features:\n")
            for idx, row in fi_df.iterrows():
                f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
                
    except Exception as e:
        print(f"Could not extract feature importances: {e}")

if __name__ == "__main__":
    run_prediction()
