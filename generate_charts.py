import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_charts():
    print("Loading data...")
    df = pd.read_csv('datasource.csv')
    
    # Ensure static/images exists
    os.makedirs('static/images', exist_ok=True)
    
    # Set style
    plt.style.use('dark_background')
    sns.set_palette("viridis")
    
    # 1. PIE CHART: Data Distribution by Country
    print("Generating Pie Chart...")
    country_counts = df['Country'].value_counts()
    
    plt.figure(figsize=(10, 10))
    plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(country_counts)))
    plt.title('Data Distribution by Country', fontsize=16, color='white')
    plt.savefig('static/images/chart_pie_country_dist.png', transparent=True)
    plt.close()
    
    # 2. LINE CHART: Budget Deficit Trends (Top 5 Countries by data volume)
    print("Generating Line Chart...")
    # Filter for Budget Deficit
    bd_df = df[df['Indicator'] == 'Budget Deficit/Surplus'].copy()
    bd_df['Time'] = pd.to_datetime(bd_df['Time'], errors='coerce')
    bd_df['Amount'] = pd.to_numeric(bd_df['Amount'], errors='coerce')
    bd_df = bd_df.dropna(subset=['Time', 'Amount'])
    
    # Get top 5 countries
    top_countries = bd_df['Country'].value_counts().head(5).index
    bd_df_top = bd_df[bd_df['Country'].isin(top_countries)]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=bd_df_top, x='Time', y='Amount', hue='Country', linewidth=2.5)
    plt.title('Budget Deficit Trends (Top 5 Countries)', fontsize=16, color='white')
    plt.xlabel('Year', fontsize=12, color='gray')
    plt.ylabel('Amount', fontsize=12, color='gray')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig('static/images/chart_line_trends.png', transparent=True)
    plt.close()
    
    # 3. BAR CHART: Average Budget Deficit by Country
    print("Generating Bar Chart...")
    avg_bd = bd_df.groupby('Country')['Amount'].mean().sort_values()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=avg_bd.values, y=avg_bd.index, palette="mako")
    plt.title('Average Budget Deficit/Surplus by Country', fontsize=16, color='white')
    plt.xlabel('Average Amount', fontsize=12, color='gray')
    plt.ylabel('Country', fontsize=12, color='gray')
    plt.grid(True, axis='x', alpha=0.1)
    plt.tight_layout()
    plt.savefig('static/images/chart_bar_ranking.png', transparent=True)
    plt.close()

    print("Charts generated successfully in static/images/")

if __name__ == "__main__":
    generate_charts()
