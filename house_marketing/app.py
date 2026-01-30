import streamlit as st
import requests
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

st.title("🏡 House Price & Sales Predictor")

# Input form
inputs = {
    "Square_Footage": st.number_input("Square Footage"),
    "Bedrooms": st.number_input("Bedrooms", step=1),
    "Bathrooms": st.number_input("Bathrooms"),
    "Age": st.number_input("Age", step=1),
    "Garage_Spaces": st.number_input("Garage Spaces"),
    "Lot_Size": st.number_input("Lot Size"),
    "Floors": st.number_input("Floors", step=1),
    "Neighborhood_Rating": st.number_input("Neighborhood Rating", step=1),
    "Condition": st.number_input("Condition", step=1),
    "School_Rating": st.number_input("School Rating"),
    "Has_Pool": st.selectbox("Has Pool", [0,1]),
    "Renovated": st.selectbox("Renovated", [0,1]),
    "Location_Type": st.selectbox("Location Type", ["Suburban","Urban","Rural"]),
    "Distance_To_Center_KM": st.number_input("Distance to Center (KM)"),
    "Days_On_Market": st.number_input("Days on Market")
}

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:5000/predict", json=inputs)
    result = response.json()
    st.success(f"Predicted Price: ${result['Predicted_Price']:.2f}")
    st.info(f"Sold Within Week: {'Yes' if result['Sold_Within_Week']==1 else 'No'}")

# Show saved records
conn = sqlite3.connect("housing.db")
df = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

st.subheader("📊 Saved Predictions")
st.dataframe(df)

# Graphs
st.subheader("Price vs Square Footage")
fig, ax = plt.subplots()
ax.scatter(df["Square_Footage"], df["Predicted_Price"])
ax.set_xlabel("Square Footage")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)