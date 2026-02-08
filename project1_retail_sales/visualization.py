import matplotlib.pyplot as plt
import seaborn as sns

def plot_monthly_sales(monthly_sales):
    plt.figure(figsize=(8,5))
    monthly_sales.plot()
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

def plot_category_sales(category_sales):
    plt.figure(figsize=(8,5))
    sns.barplot(x=category_sales.index, y=category_sales.values)
    plt.title("Category Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
