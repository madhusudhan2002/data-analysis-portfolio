from project1_retail_sales.analysis import load_data, sales_summary
from project1_retail_sales.modeling import train_sales_model
from project1_retail_sales.executive_summary import generate_executive_summary

def main():
    path = "project1_retail_sales/data/retail_sales.csv"
    
    df = load_data(path)
    
    model, score = train_sales_model(df)
    
    generate_executive_summary(df, score)

if __name__ == "__main__":
    main()
