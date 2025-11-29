import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_process_data(filepath):
    """
    Loads the CSV, pivots it to wide format so we can use ALL indicators as features.
    Respects 'Time' and 'Amount' columns as requested.
    """
    print(f"Loading data from {filepath}...")
    
    # Load data
    # Trying multiple encodings just in case
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')

    print(f"Columns found: {df.columns.tolist()}")
    
    # Ensure we have the required columns
    # User specified: Country,Indicator,Source,Unit,Currency,Frequency,Country Code,Time,Amount
    required_cols = ['Time', 'Indicator', 'Amount']
    for col in required_cols:
        if col not in df.columns:
            # Fallback check if case sensitivity is an issue
            found = False
            for existing_col in df.columns:
                if existing_col.lower() == col.lower():
                    df = df.rename(columns={existing_col: col})
                    found = True
                    break
            if not found:
                raise ValueError(f"Missing required column: {col}. Found: {df.columns.tolist()}")

    # Parse Time column
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    
    # Drop rows with invalid Time
    if df['Time'].isna().any():
        print(f"Dropping {df['Time'].isna().sum()} rows with invalid Time.")
        df = df.dropna(subset=['Time'])
    
    # Ensure Amount is numeric
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Pivot: Time as index, Indicator as columns, Amount as values
    # We aggregate by mean in case of duplicates for the same month
    df_pivot = df.pivot_table(index='Time', columns='Indicator', values='Amount', aggfunc='mean')
    
    # Sort by Time
    df_pivot = df_pivot.sort_index()
    
    # Handle missing values
    # 1. Forward fill (propagate last known value - useful for Yearly data in a Monthly series)
    df_pivot = df_pivot.ffill()
    # 2. Backward fill (fill initial missing values)
    df_pivot = df_pivot.bfill()
    
    # Drop columns that are completely empty
    df_pivot = df_pivot.dropna(axis=1, how='all')
    
    return df_pivot

def create_features(df, target_col, lags=[1, 3, 6]):
    """
    Creates lag features for ALL available indicators to predict the target.
    """
    df_features = df.copy()
    
    # Create lag features for every column (including the target itself)
    # This allows the model to learn that "Inflation 3 months ago affects Budget Deficit today"
    for col in df.columns:
        for lag in lags:
            df_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    # Add time-based features (Seasonality)
    df_features['Month'] = df_features.index.month
    df_features['Quarter'] = df_features.index.quarter
    df_features['Year'] = df_features.index.year
    
    # Drop rows with NaN values created by shifting
    df_features = df_features.dropna()
    
    return df_features

def train_model(filepath, target_indicator='Budget Deficit/Surplus'):
    # 1. Load and Pivot Data
    try:
        df = load_and_process_data(filepath)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    if target_indicator not in df.columns:
        print(f"Target '{target_indicator}' not found in dataset.")
        print("Available indicators:", df.columns.tolist())
        # Try to find a close match
        for col in df.columns:
            if target_indicator.lower() in col.lower():
                print(f"Did you mean '{col}'?")
                target_indicator = col
                break
        else:
            return

    print(f"Target Variable: {target_indicator}")
    print(f"Data shape (Time steps, Indicators): {df.shape}")
    
    # 2. Feature Engineering
    print("Generating features from ALL indicators (Multivariate approach)...")
    df_processed = create_features(df, target_indicator)
    print(f"Shape after feature engineering: {df_processed.shape}")
    
    # 3. Prepare X and y
    y = df_processed[target_indicator]
    
    # X should contain all LAGGED features + Time features.
    # We drop the current timestamp's values for ALL indicators to prevent data leakage.
    original_cols = df.columns.tolist()
    X = df_processed.drop(columns=original_cols)
    
    # 4. Train/Test Split (Time Series)
    # Hold out the last 12 periods (e.g., 1 year if monthly) for testing
    test_size = 12 
    if len(X) < 20:
        test_size = 2 # Small dataset fallback
        
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # 5. Model Training
    print(f"Training Gradient Boosting Regressor on {len(X_train)} samples...")
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"\nTest Set Metrics (Last {test_size} periods):")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # 7. Feature Importance
    print("\nTop Predictors (Drivers of Budget Deficit):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', marker='o')
    plt.plot(y_test.index, predictions, label='Predicted', marker='x', linestyle='--')
    plt.title(f'Forecast: {target_indicator}')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forecast_plot.png')
    print("\nForecast plot saved to forecast_plot.png")

if __name__ == "__main__":
    train_model('datasource.csv', target_indicator='Budget Deficit/Surplus')