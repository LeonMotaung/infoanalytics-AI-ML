import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_extra_insights():
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
    
    # --- 1. CRISIS WATCH: INFLATION ANOMALIES ---
    print("Generating Anomalies Chart...")
    target_indicator = 'Inflation Rate'
    # Handle potential variations in naming
    if target_indicator not in df['Indicator'].unique():
        # Try finding it
        matches = [i for i in df['Indicator'].unique() if 'Inflation' in i]
        if matches:
            target_indicator = matches[0]
            
    anom_df = df[df['Indicator'] == target_indicator].copy()
    
    # Calculate Mean and Std Dev per country
    stats = anom_df.groupby('Country')['Amount'].agg(['mean', 'std'])
    
    anomalies = []
    for country in anom_df['Country'].unique():
        country_data = anom_df[anom_df['Country'] == country]
        mean = stats.loc[country, 'mean']
        std = stats.loc[country, 'std']
        
        # Define anomaly as > 2 std devs from mean
        country_anoms = country_data[np.abs(country_data['Amount'] - mean) > (2 * std)]
        anomalies.append(country_anoms)
        
    anomalies_df = pd.concat(anomalies)
    
    plt.figure(figsize=(12, 6))
    # Plot all points faintly
    sns.lineplot(data=anom_df, x='Time', y='Amount', hue='Country', alpha=0.3, legend=False, palette='viridis')
    # Plot anomalies boldly
    sns.scatterplot(data=anomalies_df, x='Time', y='Amount', hue='Country', s=100, marker='X', palette='viridis')
    
    plt.title(f'Crisis Watch: {target_indicator} Anomalies (>2 Std Dev)', fontsize=16, color='white')
    plt.xlabel('Year', fontsize=12, color='gray')
    plt.ylabel('Rate', fontsize=12, color='gray')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig('static/images/chart_anomalies.png', transparent=True)
    plt.close()

    # --- 2. FISCAL HEALTH: DEBT-TO-GDP RATIO ---
    print("Generating Debt-to-GDP Chart...")
    # We need Government Debt and Nominal GDP
    debt_indicator = 'Government Debt'
    gdp_indicator = 'Nominal GDP'
    
    if debt_indicator in df['Indicator'].unique() and gdp_indicator in df['Indicator'].unique():
        # Filter for latest year available for each country
        latest_year = df['Year'].max()
        
        # Aggregate by mean to handle potential duplicates for the same country/year
        debt_df = df[(df['Indicator'] == debt_indicator) & (df['Year'] == latest_year)].groupby('Country')['Amount'].mean()
        gdp_df = df[(df['Indicator'] == gdp_indicator) & (df['Year'] == latest_year)].groupby('Country')['Amount'].mean()
        
        # Join (pandas Series join aligns on index automatically)
        ratio_df = pd.DataFrame({'Debt': debt_df, 'GDP': gdp_df})
        
        # Calculate Ratio
        ratio_df['Debt_to_GDP'] = (ratio_df['Debt'] / ratio_df['GDP']) * 100
        ratio_df = ratio_df.dropna().sort_values('Debt_to_GDP', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=ratio_df.index, y=ratio_df['Debt_to_GDP'], palette='Reds_r')
        plt.title(f'Debt-to-GDP Ratio ({latest_year})', fontsize=16, color='white')
        plt.ylabel('Ratio (%)', fontsize=12, color='gray')
        plt.xlabel('Country', fontsize=12, color='gray')
        plt.axhline(y=60, color='yellow', linestyle='--', alpha=0.5, label='Prudent Limit (60%)')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.1)
        plt.tight_layout()
        plt.savefig('static/images/chart_debt_to_gdp.png', transparent=True)
        plt.close()
    else:
        print("Skipping Debt-to-GDP: Indicators not found.")

    print("Extra insights generated successfully.")

if __name__ == "__main__":
    generate_extra_insights()
