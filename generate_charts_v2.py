import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import pi

def generate_charts():
    print("Loading data...")
    df = pd.read_csv('datasource.csv')
    
    # Ensure static/images exists
    os.makedirs('static/images', exist_ok=True)
    
    # Set style
    plt.style.use('dark_background')
    # Custom colors
    colors = ['#ccff00', '#00ff9d', '#00ccff', '#ff00ff', '#ffff00', '#ff5500']
    sns.set_palette(sns.color_palette(colors))
    
    # Preprocessing
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Time', 'Amount'])
    df['Year'] = df['Time'].dt.year
    
    # --- 1. CLUSTERED COLUMN CHART ---
    # Compare 'Tax Revenue' vs 'Budget Deficit/Surplus' (Absolute) for Top Countries (Last 5 Years Avg)
    print("Generating Clustered Column Chart...")
    indicators = ['Tax Revenue', 'Budget Deficit/Surplus']
    subset = df[df['Indicator'].isin(indicators)].copy()
    
    # Filter for last 5 years
    max_year = subset['Year'].max()
    subset = subset[subset['Year'] >= max_year - 5]
    
    # Pivot to get indicators as columns
    pivot_df = subset.groupby(['Country', 'Indicator'])['Amount'].mean().unstack()
    
    # Select top 5 countries by Tax Revenue
    if 'Tax Revenue' in pivot_df.columns:
        top_countries = pivot_df.sort_values('Tax Revenue', ascending=False).head(5).index
        plot_data = pivot_df.loc[top_countries]
        
        # Make Budget Deficit positive for comparison if needed, or keep as is. 
        # Usually deficits are negative. Let's plot raw values.
        
        plot_data.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title('Fiscal Performance (5-Year Avg)', fontsize=16, color='white')
        plt.ylabel('Amount', fontsize=12, color='gray')
        plt.xlabel('Country', fontsize=12, color='gray')
        plt.legend(title='Indicator')
        plt.grid(True, axis='y', alpha=0.1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/images/chart_clustered_column.png', transparent=True)
        plt.close()
    else:
        print("Tax Revenue not found for Clustered Column")

    # --- 2. STACKED AREA CHART ---
    # Tax Revenue Trend over time (Top 5 Countries)
    print("Generating Stacked Area Chart...")
    tax_df = df[df['Indicator'] == 'Tax Revenue'].copy()
    if not tax_df.empty:
        # Top 5 countries
        top_tax_countries = tax_df.groupby('Country')['Amount'].sum().sort_values(ascending=False).head(5).index
        tax_df = tax_df[tax_df['Country'].isin(top_tax_countries)]
        
        # Pivot for stacking
        pivot_tax = tax_df.pivot_table(index='Year', columns='Country', values='Amount', aggfunc='sum').fillna(0)
        
        plt.figure(figsize=(12, 6))
        plt.stackplot(pivot_tax.index, pivot_tax.T, labels=pivot_tax.columns, alpha=0.7, colors=colors[:5])
        plt.title('Tax Revenue Trends (Stacked Area)', fontsize=16, color='white')
        plt.xlabel('Year', fontsize=12, color='gray')
        plt.ylabel('Total Revenue', fontsize=12, color='gray')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        plt.savefig('static/images/chart_area_stacked.png', transparent=True)
        plt.close()

    # --- 3. PIE CHART ---
    # Share of Total Tax Revenue (Most Recent Year)
    print("Generating Pie Chart...")
    if not tax_df.empty:
        recent_year = tax_df['Year'].max()
        recent_tax = tax_df[tax_df['Year'] == recent_year].groupby('Country')['Amount'].sum().sort_values(ascending=False)
        
        # Group small slices into 'Other'
        top_slices = recent_tax.head(5)
        other = pd.Series({'Other': recent_tax.iloc[5:].sum()}) if len(recent_tax) > 5 else pd.Series()
        final_pie = pd.concat([top_slices, other])
        
        plt.figure(figsize=(10, 10))
        plt.pie(final_pie, labels=final_pie.index, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title(f'Tax Revenue Share ({recent_year})', fontsize=16, color='white')
        
        # Add a hole for Donut chart look (modern)
        centre_circle = plt.Circle((0,0),0.70,fc='#050505')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.tight_layout()
        plt.savefig('static/images/chart_pie_donut.png', transparent=True)
        plt.close()

    # --- 4. RADAR CHART ---
    # Compare Normalized Indicators for Top 3 Countries
    print("Generating Radar Chart...")
    # Select indicators
    radar_indicators = ['Tax Revenue', 'Budget Deficit/Surplus', 'Unemployment Rate']
    radar_df = df[df['Indicator'].isin(radar_indicators)].copy()
    
    # Filter for recent year
    max_radar_year = radar_df['Year'].max()
    radar_df = radar_df[radar_df['Year'] == max_radar_year]
    
    # Pivot
    radar_pivot = radar_df.pivot_table(index='Country', columns='Indicator', values='Amount', aggfunc='mean')
    
    # Normalize (Min-Max Scaling)
    radar_norm = (radar_pivot - radar_pivot.min()) / (radar_pivot.max() - radar_pivot.min())
    radar_norm = radar_norm.fillna(0)
    
    # Select top 3 countries with most data
    countries_to_plot = radar_norm.dropna().head(3).index
    
    if len(countries_to_plot) >= 3:
        # Categories
        categories = list(radar_norm.columns)
        N = len(categories)
        
        # Angles
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # Draw one axe per variable + labels
        plt.xticks(angles[:-1], categories, color='white', size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="gray", size=7)
        plt.ylim(0, 1)
        
        # Plot each country
        for i, country in enumerate(countries_to_plot):
            values = radar_norm.loc[country].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=country, color=colors[i])
            ax.fill(angles, values, color=colors[i], alpha=0.2)
            
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Economic Health Comparison (Normalized)', size=16, color='white', y=1.1)
        plt.savefig('static/images/chart_radar.png', transparent=True)
        plt.close()
    else:
        print("Not enough data for Radar Chart")

    # --- 5. STACKED LINE CHART (Alternative to Area) ---
    # Just another visual for Unemployment if available
    print("Generating Stacked Line Chart...")
    unemp_df = df[df['Indicator'] == 'Unemployment Rate'].copy()
    if not unemp_df.empty:
        top_unemp = unemp_df.groupby('Country')['Amount'].mean().sort_values(ascending=False).head(5).index
        unemp_df = unemp_df[unemp_df['Country'].isin(top_unemp)]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=unemp_df, x='Year', y='Amount', hue='Country', linewidth=2.5, palette=colors[:5])
        plt.title('Unemployment Rate Trends', fontsize=16, color='white')
        plt.xlabel('Year', fontsize=12, color='gray')
        plt.ylabel('Rate (%)', fontsize=12, color='gray')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        plt.savefig('static/images/chart_line_stacked.png', transparent=True)
        plt.close()

    print("All charts generated successfully.")

if __name__ == "__main__":
    generate_charts()
