import pandas as pd
import numpy as np


# -------------------------------------------------
# 1️⃣ LOAD & BASIC CLEANING
# -------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)

    # Convert to datetime safely
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing numeric values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


# -------------------------------------------------
# 2️⃣ SALES SUMMARY
# -------------------------------------------------
def sales_summary(df):

    total_revenue = df['Sales'].sum()

    monthly_sales = df.resample('M', on='OrderDate')['Sales'].sum()

    category_sales = df.groupby('Category')['Sales'].sum()

    return total_revenue, monthly_sales, category_sales


# -------------------------------------------------
# 3️⃣ RFM ANALYSIS + SCORING
# -------------------------------------------------
def rfm_analysis(df):

    max_date = df['OrderDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'OrderDate': lambda x: (max_date - x.max()).days,
        'OrderID': 'nunique',
        'Sales': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # RFM scoring (1–5 scale)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=range(5, 0, -1))
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=range(1, 6))
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=range(1, 6))

    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(int) +
        rfm['F_Score'].astype(int) +
        rfm['M_Score'].astype(int)
    )

    return rfm


# -------------------------------------------------
# 4️⃣ ADVANCED FEATURE ENGINEERING
# -------------------------------------------------
def advanced_feature_engineering(df):

    df = df.sort_values('OrderDate')

    # Time-based features
    df['Year'] = df['OrderDate'].dt.year
    df['Month'] = df['OrderDate'].dt.month
    df['Day'] = df['OrderDate'].dt.day
    df['Weekday'] = df['OrderDate'].dt.weekday
    df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)

    # Rolling 7-day average per category
    df['Rolling_7Day_Sales'] = (
        df.groupby('Category')['Sales']
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )

    # Average price per unit
    df['AvgPrice'] = df['Sales'] / df['Quantity']

    return df


# -------------------------------------------------
# 5️⃣ PIVOT TABLE (ADVANCED PANDAS)
# -------------------------------------------------
def region_category_pivot(df):

    pivot = pd.pivot_table(
        df,
        values='Sales',
        index='Region',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    )

    return pivot


# -------------------------------------------------
# 6️⃣ CORRELATION MATRIX
# -------------------------------------------------
def correlation_matrix(df):

    return df.corr(numeric_only=True)


# -------------------------------------------------
# 7️⃣ EDA SUMMARY STATISTICS
# -------------------------------------------------
def eda_summary(df):

    summary = {
        "Shape": df.shape,
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.to_dict(),
        "Describe": df.describe().to_dict(),
        "Unique Customers": df['CustomerID'].nunique(),
        "Unique Products": df['Product'].nunique()
    }

    return summary


# -------------------------------------------------
# 8️⃣ OUTLIER REMOVAL (IQR METHOD)
# -------------------------------------------------
def remove_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]


# -------------------------------------------------
# 9️⃣ MONTHLY GROWTH RATE
# -------------------------------------------------
def calculate_monthly_growth(df):

    monthly_sales = df.resample('M', on='OrderDate')['Sales'].sum()
    growth = monthly_sales.pct_change() * 100

    return growth


# -------------------------------------------------
# 🔟 PARETO ANALYSIS (80/20 RULE)
# -------------------------------------------------
def pareto_analysis(df):

    customer_sales = df.groupby('CustomerID')['Sales'].sum().sort_values(ascending=False)

    cumulative_percentage = customer_sales.cumsum() / customer_sales.sum() * 100

    top_20_percent_customers = cumulative_percentage[cumulative_percentage <= 80].count()

    return {
        "top_customers_count": top_20_percent_customers,
        "total_customers": len(customer_sales)
    }
