import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay


# Global Style
sns.set_theme(style="whitegrid")


# -------------------------------------------------
# 1️⃣ MONTHLY SALES TREND
# -------------------------------------------------
def plot_monthly_sales(monthly_sales, save=False):
    plt.figure(figsize=(10, 5))
    monthly_sales.plot(marker='o')
    plt.title("Monthly Sales Trend", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()

    if save:
        plt.savefig("monthly_sales.png")

    plt.show()


# -------------------------------------------------
# 2️⃣ CATEGORY SALES BAR CHART
# -------------------------------------------------
def plot_category_sales(category_sales, save=False):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category_sales.index, y=category_sales.values)
    plt.title("Category Sales Distribution", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        plt.savefig("category_sales.png")

    plt.show()


# -------------------------------------------------
# 3️⃣ CORRELATION HEATMAP
# -------------------------------------------------
def plot_correlation_heatmap(df, save=False):
    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if save:
        plt.savefig("correlation_heatmap.png")

    plt.show()


# -------------------------------------------------
# 4️⃣ SALES DISTRIBUTION
# -------------------------------------------------
def plot_sales_distribution(df, save=False):
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Sales'], kde=True, bins=30)
    plt.title("Sales Distribution")
    plt.tight_layout()

    if save:
        plt.savefig("sales_distribution.png")

    plt.show()


# -------------------------------------------------
# 5️⃣ BOXPLOT (OUTLIER DETECTION)
# -------------------------------------------------
def plot_sales_boxplot(df, save=False):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['Sales'])
    plt.title("Outlier Detection - Sales")
    plt.tight_layout()

    if save:
        plt.savefig("sales_boxplot.png")

    plt.show()


# -------------------------------------------------
# 6️⃣ PAIRPLOT
# -------------------------------------------------
def plot_pairplot(df):
    sns.pairplot(df[['Sales', 'Quantity']])
    plt.show()


# -------------------------------------------------
# 7️⃣ PIVOT HEATMAP
# -------------------------------------------------
def plot_pivot_heatmap(pivot_table, save=False):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".0f")
    plt.title("Region vs Category Sales")
    plt.tight_layout()

    if save:
        plt.savefig("pivot_heatmap.png")

    plt.show()


# -------------------------------------------------
# 8️⃣ CONFUSION MATRIX
# -------------------------------------------------
def plot_confusion_matrix(model, X_test, y_test, save=False):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save:
        plt.savefig("confusion_matrix.png")

    plt.show()


# -------------------------------------------------
# 9️⃣ ROC CURVE
# -------------------------------------------------
def plot_roc_curve(y_test, y_probs, save=False):
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("roc_curve.png")

    plt.show()


# -------------------------------------------------
# 🔟 FEATURE IMPORTANCE
# -------------------------------------------------
def plot_feature_importance(feature_importance_dict, save=False):
    features = list(feature_importance_dict.keys())
    importance = list(feature_importance_dict.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance, y=features)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()

    if save:
        plt.savefig("feature_importance.png")

    plt.show()


# -------------------------------------------------
# 1️⃣1️⃣ ROLLING SALES TREND
# -------------------------------------------------
def plot_rolling_sales(df, save=False):
    plt.figure(figsize=(10, 5))
    df.groupby('OrderDate')['Rolling_7Day_Sales'].mean().plot()
    plt.title("7-Day Rolling Average Sales")
    plt.xlabel("Date")
    plt.ylabel("Rolling Avg Sales")
    plt.tight_layout()

    if save:
        plt.savefig("rolling_sales.png")

    plt.show()


# -------------------------------------------------
# 1️⃣2️⃣ REGRESSION RESIDUAL PLOT
# -------------------------------------------------
def plot_regression_residuals(y_test, predictions, save=False):
    residuals = y_test - predictions

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Regression Residual Plot")
    plt.tight_layout()

    if save:
        plt.savefig("regression_residuals.png")

    plt.show()


# -------------------------------------------------
# 1️⃣3️⃣ MONTHLY GROWTH PLOT
# -------------------------------------------------
def plot_monthly_growth(growth_series, save=False):
    plt.figure(figsize=(8, 4))
    growth_series.plot(marker='o')
    plt.title("Monthly Growth Rate (%)")
    plt.ylabel("Growth %")
    plt.tight_layout()

    if save:
        plt.savefig("monthly_growth.png")

    plt.show()
