import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_advanced_insights():
    print("Loading data...")
    df = pd.read_csv('datasource.csv')
    
    # Ensure static/images exists
    os.makedirs('static/images', exist_ok=True)
    
    # Set style
    plt.style.use('dark_background')
    
    # Preprocessing
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Time', 'Amount'])
    df['Year'] = df['Time'].dt.year
    
    # --- 1. CORRELATION HEATMAP ---
    print("Generating Correlation Heatmap...")
    # Pivot data: Index=Time/Country, Columns=Indicator, Values=Amount
    # We need to align data by Country and Year to see correlations
    pivot_df = df.pivot_table(index=['Country', 'Year'], columns='Indicator', values='Amount')
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Filter for interesting indicators if too many
    # Let's keep top 10 most populated indicators
    top_indicators = df['Indicator'].value_counts().head(10).index
    corr_matrix_top = pivot_df[top_indicators].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix_top, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5, linecolor='black')
    plt.title('Macroeconomic Indicator Correlations', fontsize=16, color='white', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/images/chart_correlation.png', transparent=True)
    plt.close()

    # --- 2. VOLATILITY ANALYSIS (Risk Ranking) ---
    print("Generating Volatility Chart...")
    # Calculate Coefficient of Variation (CV) = StdDev / Mean
    # This normalizes volatility so we can compare different scales
    # We'll use 'Budget Deficit/Surplus' or 'Inflation' if available. 
    # Let's use 'Budget Deficit/Surplus' as it's our main theme.
    
    target_indicator = 'Budget Deficit/Surplus'
    vol_df = df[df['Indicator'] == target_indicator].copy()
    
    # Calculate Std Dev per country
    volatility = vol_df.groupby('Country')['Amount'].std().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=volatility.values, y=volatility.index, palette="magma")
    plt.title(f'Economic Volatility Ranking ({target_indicator})', fontsize=16, color='white')
    plt.xlabel('Standard Deviation (Volatility)', fontsize=12, color='gray')
    plt.ylabel('Country', fontsize=12, color='gray')
    plt.grid(True, axis='x', alpha=0.1)
    plt.tight_layout()
    plt.savefig('static/images/chart_volatility.png', transparent=True)
    plt.close()

    print("Advanced insights generated successfully.")

if __name__ == "__main__":
    generate_advanced_insights()
