from project1_retail_sales.analysis import (
    load_data,
    advanced_feature_engineering,
    region_category_pivot,
    rfm_analysis
)

from project1_retail_sales.statistics_analysis import t_test_weekend_sales

from project1_retail_sales.modeling import (
    churn_prediction,
    random_forest_churn,
    sales_regression_model,
    shap_analysis
)

from project1_retail_sales.visualization import (
    plot_monthly_sales,
    plot_category_sales,
    plot_correlation_heatmap,
    plot_sales_distribution,
    plot_sales_boxplot,
    plot_pivot_heatmap,
    plot_roc_curve,
    plot_feature_importance
)

from project1_retail_sales.executive_summary import generate_executive_summary


def main():

    print("\nLOADING DATA...")
    df = load_data("project1_retail_sales/data/retail_sales.csv")

    print("Performing Feature Engineering...")
    df = advanced_feature_engineering(df)

    # -------------------------------------------------
    # PIVOT TABLE
    # -------------------------------------------------
    pivot = region_category_pivot(df)
    print("\nRegion-Category Pivot Table:\n", pivot)

    # -------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------
    stats_results = t_test_weekend_sales(df)

    print("\nSTATISTICAL TEST RESULTS")
    print("=" * 40)

    if "error" in stats_results:
        print(stats_results["error"])
    else:
        for key, value in stats_results.items():
            print(f"{key}: {value}")

    # -------------------------------------------------
    # BASIC VISUALIZATIONS
    # -------------------------------------------------
    monthly_sales = df.resample('M', on='OrderDate')['Sales'].sum()
    category_sales = df.groupby('Category')['Sales'].sum()

    plot_monthly_sales(monthly_sales)
    plot_category_sales(category_sales)
    plot_correlation_heatmap(df)
    plot_sales_distribution(df)
    plot_sales_boxplot(df)
    plot_pivot_heatmap(pivot)

    # -------------------------------------------------
    # RFM ANALYSIS
    # -------------------------------------------------
    rfm = rfm_analysis(df)
    print("\nRFM Size:", len(rfm))

    # -------------------------------------------------
    # MACHINE LEARNING MODELS
    # -------------------------------------------------
    print("\nTRAINING LOGISTIC REGRESSION...")
    logistic_results = churn_prediction(rfm)

    print("\nTRAINING RANDOM FOREST...")
    rf_results = random_forest_churn(rfm)

    print("\nTRAINING REGRESSION MODEL...")
    reg_results = sales_regression_model(df)

    # -------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------
    print("\nMODEL COMPARISON")
    print("=" * 30)

    if "error" in logistic_results:
        print("Logistic Model Skipped:", logistic_results["error"])
    else:
        print(f"Logistic ROC-AUC: {logistic_results['roc_auc']:.3f}")

    if "error" in rf_results:
        print("Random Forest Model Skipped:", rf_results["error"])
    else:
        print(f"Random Forest ROC-AUC: {rf_results['roc_auc']:.3f}")

    if (
        "error" not in logistic_results and
        "error" not in rf_results
    ):
        if rf_results['roc_auc'] > logistic_results['roc_auc']:
            print("→ Random Forest performs better.")
        else:
            print("→ Logistic Regression performs better.")

    # -------------------------------------------------
    # ML VISUALIZATION
    # -------------------------------------------------
    if "error" not in logistic_results:
        plot_roc_curve(
            logistic_results["y_test"],
            logistic_results["y_probs"]
        )

    if "error" not in rf_results:
        plot_feature_importance(
            rf_results["feature_importance"]
        )

    # -------------------------------------------------
    # SHAP EXPLAINABILITY
    # -------------------------------------------------
    if "error" not in rf_results:
        print("\nRunning SHAP Analysis...")
        shap_analysis(
            rf_results["model"],
            rf_results["X_train"]
        )
    else:
        print("\nSHAP skipped due to insufficient data.")

    # -------------------------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------------------------
    generate_executive_summary(
        df,
        regression_results=reg_results,
        logistic_results=(
            None if "error" in logistic_results else logistic_results
        ),
        rf_results=(
            None if "error" in rf_results else rf_results
        )
    )


if __name__ == "__main__":
    main()
