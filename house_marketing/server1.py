from flask import Flask, request, jsonify
import joblib
import sqlite3
import pandas as pd

app = Flask(__name__)

# Load models
linreg_model = joblib.load("price_predicter.pkl")
logreg_model = joblib.load("sales_predicter.pkl")

# DB setup
def init_db():
    conn = sqlite3.connect("housing.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Square_Footage REAL,
            Bedrooms INTEGER,
            Bathrooms REAL,
            Age INTEGER,
            Garage_Spaces REAL,
            Lot_Size REAL,
            Floors INTEGER,
            Neighborhood_Rating INTEGER,
            Condition INTEGER,
            School_Rating REAL,
            Has_Pool INTEGER,
            Renovated INTEGER,
            Location_Type TEXT,
            Distance_To_Center_KM REAL,
            Days_On_Market REAL,
            Predicted_Price REAL,
            Sold_Within_Week INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])  # Convert input to DataFrame
    
    # Predict
    price_pred = linreg_model.predict(df)[0]
    sold_pred = logreg_model.predict(df)[0]
    
    # Save to DB
    conn = sqlite3.connect("housing.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            Square_Footage, Bedrooms, Bathrooms, Age, Garage_Spaces, Lot_Size, Floors,
            Neighborhood_Rating, Condition, School_Rating, Has_Pool, Renovated,
            Location_Type, Distance_To_Center_KM, Days_On_Market, Predicted_Price, Sold_Within_Week
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["Square_Footage"], data["Bedrooms"], data["Bathrooms"], data["Age"], data["Garage_Spaces"],
        data["Lot_Size"], data["Floors"], data["Neighborhood_Rating"], data["Condition"], data["School_Rating"],
        data["Has_Pool"], data["Renovated"], data["Location_Type"], data["Distance_To_Center_KM"],
        data["Days_On_Market"], price_pred, sold_pred
    ))
    conn.commit()
    conn.close()
    
    return jsonify({"Predicted_Price": price_pred, "Sold_Within_Week": int(sold_pred)})

if __name__ == "__main__":
    app.run(debug=True)