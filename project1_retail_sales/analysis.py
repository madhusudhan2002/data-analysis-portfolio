import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    return df

def sales_summary(df):
    total_revenue = df['Sales'].sum()
    monthly_sales = df.resample('M', on='OrderDate')['Sales'].sum()
    category_sales = df.groupby('Category')['Sales'].sum()
    
    return total_revenue, monthly_sales, category_sales

def rfm_analysis(df):
    max_date = df['OrderDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'OrderDate': lambda x: (max_date - x.max()).days,
        'OrderID': 'nunique',
        'Sales': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm
