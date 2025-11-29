import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_sector_insights():
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
    
    # --- 1. TRADE BALANCE (Net Exports) ---
    print("Generating Trade Balance Chart...")
    trade_indicators = ['Exports', 'Imports']
    
    # Check if indicators exist
    available_trade = [i for i in trade_indicators if i in df['Indicator'].unique()]
    
    if len(available_trade) == 2:
        trade_df = df[df['Indicator'].isin(trade_indicators)].pivot_table(
            index=['Country', 'Year'], 
            columns='Indicator', 
            values='Amount'
        ).reset_index()
        
        trade_df['Net Trade'] = trade_df['Exports'] - trade_df['Imports']
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=trade_df, x='Year', y='Net Trade', hue='Country', marker='o', palette='GnBu_r')
        plt.axhline(0, color='white', linestyle='--', alpha=0.5)
        plt.title('Trade Balance Trends (Exports - Imports)', fontsize=16, color='white')
        plt.ylabel('Net Amount (Currency)', fontsize=12, color='gray')
        plt.xlabel('Year', fontsize=12, color='gray')
        plt.grid(True, alpha=0.1)
        plt.legend(title='Country')
        plt.tight_layout()
        plt.savefig('static/images/chart_trade_balance.png', transparent=True)
        plt.close()
    else:
        print("Skipping Trade Balance: Exports/Imports not fully available.")

    # --- 2. SPENDING PRIORITIES (Education vs Health vs Defence) ---
    print("Generating Spending Priorities Chart...")
    spending_indicators = ['Education Expenditure', 'Health Expenditure', 'Defence Expenditure']
    
    # Filter available
    available_spending = [i for i in spending_indicators if i in df['Indicator'].unique()]
    
    if available_spending:
        # Get latest year
        latest_year = df['Year'].max()
        spend_df = df[(df['Indicator'].isin(available_spending)) & (df['Year'] == latest_year)]
        
        # Pivot for stacked bar
        spend_pivot = spend_df.pivot_table(index='Country', columns='Indicator', values='Amount')
        
        if not spend_pivot.empty:
            # Normalize to 100% to show priorities relative to each other
            spend_pct = spend_pivot.div(spend_pivot.sum(axis=1), axis=0) * 100
            
            ax = spend_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
        plt.title(f'Government Spending Priorities ({latest_year})', fontsize=16, color='white')
        plt.ylabel('Share of Selected Spending (%)', fontsize=12, color='gray')
        plt.xlabel('Country', fontsize=12, color='gray')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.1)
        plt.tight_layout()
        plt.savefig('static/images/chart_spending_priorities.png', transparent=True)
        plt.close()
    else:
        print("Skipping Spending Priorities: Indicators not found.")

    print("Sector insights generated successfully.")

if __name__ == "__main__":
    generate_sector_insights()
