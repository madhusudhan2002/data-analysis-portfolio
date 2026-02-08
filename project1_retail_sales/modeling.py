from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def train_sales_model(df):
    df['Month'] = df['OrderDate'].dt.month
    
    X = df[['Month', 'Quantity']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    
    return model, score
