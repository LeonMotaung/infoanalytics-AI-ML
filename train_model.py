import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')

    # Rename columns if needed to match standard names
    # User specified: Country,Indicator,Source,Unit,Currency,Frequency,Country Code,Time,Amount
    col_map = {'Time': 'Date', 'Amount': 'Value'}
    df = df.rename(columns=col_map)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Ensure Value is numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Pivot to wide format (Date x Indicators)
    df_pivot = df.pivot_table(index='Date', columns='Indicator', values='Value', aggfunc='mean')
    df_pivot = df_pivot.sort_index()
    
    # Fill missing values
    df_pivot = df_pivot.ffill().bfill()
    df_pivot = df_pivot.dropna(axis=1, how='all')
    
    return df_pivot

def create_classification_target(df, target_indicator):
    """
    Converts a continuous target into a binary classification target.
    We will predict: Will the value INCREASE next month? (1 = Yes, 0 = No)
    """
    if target_indicator not in df.columns:
        raise ValueError(f"Target '{target_indicator}' not found.")
        
    # Calculate the change from the previous period
    # If Change > 0, Class = 1 (Improved/Increased), else 0
    target_series = df[target_indicator]
    
    # Create binary target: 1 if value increased, 0 if decreased/same
    # We shift(-1) because we want to predict the NEXT month's direction using CURRENT data
    y = (target_series.shift(-1) > target_series).astype(int)
    
    # Drop the last row because we don't have a "next month" for it
    y = y.iloc[:-1]
    
    return y

def train_svm(filepath, target_indicator='Budget Deficit/Surplus'):
    # 1. Load Data
    try:
        df = load_and_process_data(filepath)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Data Loaded. Shape: {df.shape}")
    
    # 2. Create Target (Classification)
    print(f"Target: Predicting direction of '{target_indicator}' (Up/Down)")
    y = create_classification_target(df, target_indicator)
    
    # 3. Create Features
    X = df.iloc[:-1].copy()
    X['Target_Lag1'] = df[target_indicator].shift(1).iloc[:-1]
    X['Target_Diff'] = df[target_indicator].diff().iloc[:-1]
    
    valid_indices = ~X.isna().any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    
    # 4. Train/Test Split
    test_size = 12
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # 5. Scale Data (CRITICAL for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Hyperparameter Tuning (Grid Search)
    print("Tuning SVM Hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Use TimeSeriesSplit for CV to prevent data leakage
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        SVC(random_state=42), 
        param_grid, 
        cv=tscv, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.2%}")
    
    # 7. Evaluate Best Model
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest Set Accuracy (Best Model): {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Decrease', 'Increase'], zero_division=0))

if __name__ == "__main__":
    train_svm('datasource.csv', target_indicator='Budget Deficit/Surplus')
