from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

# Country Code Mapping (ISO Alpha-3 to Alpha-2)
COUNTRY_MAPPING = {
    'EGY': 'EG', 'ETH': 'ET', 'GHA': 'GH', 'CIV': 'CI',
    'NGA': 'NG', 'KEN': 'KE', 'RWA': 'RW', 'DZA': 'DZ',
    'TGO': 'TG', 'SEN': 'SN', 'AGO': 'AO', 'BWA': 'BW',
    'ZAF': 'ZA', 'MAR': 'MA', 'TUN': 'TN', 'UGA': 'UG',
    'TZA': 'TZ'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/world')
def world():
    try:
        df = pd.read_csv('datasource.csv')
        
        # Filter for GDP Growth Rate
        target_indicator = 'GDP Growth Rate'
        
        map_data = []
        
        # Get latest data for each country
        if target_indicator in df['Indicator'].unique():
            countries = df['Country'].unique()
            for country in countries:
                # Filter safely
                country_df = df[(df['Country'] == country) & (df['Indicator'] == target_indicator)].copy()
                
                if not country_df.empty:
                    # Sort by Time and get latest
                    country_df['Time'] = pd.to_datetime(country_df['Time'])
                    latest = country_df.sort_values('Time').iloc[-1]
                    
                    code_3 = latest['Country Code']
                    code_2 = COUNTRY_MAPPING.get(code_3)
                    
                    if code_2:
                        map_data.append({
                            'id': code_2,
                            'name': country,
                            'value': float(latest['Amount']),
                            'year': int(latest['Time'].year)
                        })
        
        return render_template('world.html', map_data=map_data)
    except Exception as e:
        print(f"Error generating map data: {e}")
        return render_template('world.html', map_data=[])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
