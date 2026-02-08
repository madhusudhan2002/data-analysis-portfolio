import streamlit as st
from project1_retail_sales.analysis import load_data, sales_summary
import plotly.express as px

st.title("Retail Sales Dashboard")

df = load_data("project1_retail_sales/data/retail_sales.csv")

total_revenue, monthly_sales, category_sales = sales_summary(df)

st.metric("Total Revenue", f"₹{total_revenue:,.2f}")

fig1 = px.line(monthly_sales, title="Monthly Sales Trend")
st.plotly_chart(fig1)

fig2 = px.bar(x=category_sales.index, y=category_sales.values, title="Category Sales")
st.plotly_chart(fig2)
