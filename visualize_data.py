import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_full_history(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')

    # Rename columns
    col_map = {'Time': 'Date', 'Amount': 'Value'}
    df = df.rename(columns=col_map)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    
    # Ensure Value is numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value'])
    
    # Filter for the specific range if needed, but user asked for "whole"
    # The user mentioned 23785, which is roughly 65 years (1960-2025)
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    print(f"Visualizing data from {start_date.date()} to {end_date.date()}")
    
    # Pivot to get indicators as columns
    df_pivot = df.pivot_table(index='Date', columns='Indicator', values='Value', aggfunc='mean')
    
    # Select key indicators to plot (to avoid overcrowding)
    # We'll try to pick the most populated ones or specific ones of interest
    key_indicators = ['Budget Deficit/Surplus', 'Nominal GDP', 'Inflation Rate', 'Revenue', 'Expenditure']
    available_indicators = [col for col in key_indicators if col in df_pivot.columns]
    
    if not available_indicators:
        # Fallback to top 5 most populated columns
        available_indicators = df_pivot.count().sort_values(ascending=False).head(5).index.tolist()
    
    print(f"Plotting indicators: {available_indicators}")
    
    # Plotting
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Create subplots for each indicator to handle different scales
    for i, indicator in enumerate(available_indicators):
        plt.subplot(len(available_indicators), 1, i+1)
        
        # Get data for this indicator
        data = df_pivot[indicator].dropna()
        
        plt.plot(data.index, data.values, label=indicator, linewidth=2)
        plt.title(indicator, fontsize=12, loc='left')
        plt.legend(loc='upper left')
        
        # Format x-axis only on the bottom plot
        if i < len(available_indicators) - 1:
            plt.xticks([])
        else:
            plt.xlabel('Year', fontsize=12)
            
    plt.suptitle(f'Economic Indicators History ({start_date.year} - {end_date.year})', fontsize=16)
    plt.tight_layout()
    
    output_file = 'full_history_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    visualize_full_history('datasource.csv')
