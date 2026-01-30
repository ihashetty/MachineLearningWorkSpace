import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("house_sales_data.csv")

print(df.isnull().sum())

targets = ["Price", "Sold_Within_Week"]
features = df.drop(columns=targets)
for col in features.columns:
    if df[col].dtype in ["float64", "int64"]:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    else:
        median_val = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
        df[col].fillna(median_val, inplace=True)

print(df.isnull().sum())


X = df.drop(columns=["Price","Sold_Within_Week"])
y_price = df["Price"]
y_sold = df["Sold_Within_Week"]

categorical = ["Location_Type"]
numerical = X.columns.drop(categorical)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

linreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.3, random_state=42)
linreg_pipeline.fit(X_train, y_train)
y_pred = linreg_pipeline.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, y_pred))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

logreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y_sold, test_size=0.3, random_state=42)
logreg_pipeline.fit(X_train, y_train)
y_pred = logreg_pipeline.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(linreg_pipeline, "price_predicter.pkl")

joblib.dump(logreg_pipeline, "sales_predicter.pkl")