import pandas as pd

def check_years(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')

    # Rename columns if needed
    col_map = {'Time': 'Date', 'Amount': 'Value'}
    df = df.rename(columns=col_map)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    with open('date_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"Lower Date (Earliest): {min_date.date()}\n")
        f.write(f"Upper Date (Latest):   {max_date.date()}\n")
        
        # Compute difference
        diff = max_date - min_date
        years = diff.days / 365.25
        
        f.write(f"Total Span: {years:.2f} years\n")
        f.write(f"Unique Years present: {sorted(df['Date'].dt.year.unique())}\n")
        
    print("Info written to date_info.txt")

if __name__ == "__main__":
    check_years('datasource.csv')
