import streamlit as st
import joblib
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# -------------------
# Database Functions
# -------------------
def init_db():
    conn = sqlite3.connect("loan_applicants.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS applicants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Married TEXT,
                    Education TEXT,
                    Self_Employed TEXT,
                    ApplicantIncome INTEGER,
                    CoapplicantIncome INTEGER,
                    LoanAmount INTEGER,
                    Loan_Amount_Term INTEGER,
                    Credit_History INTEGER,
                    Property_Area TEXT,
                    Prediction INTEGER
                )''')
    conn.commit()
    conn.close()

def insert_applicant(data, prediction):
    conn = sqlite3.connect("loan_applicants.db")
    c = conn.cursor()
    c.execute('''INSERT INTO applicants 
                 (Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Prediction)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (data["Married"], data["Education"], data["Self_Employed"], data["ApplicantIncome"], 
               data["CoapplicantIncome"], data["LoanAmount"], data["Loan_Amount_Term"], 
               data["Credit_History"], data["Property_Area"], prediction))
    conn.commit()
    conn.close()

def fetch_applicants():
    conn = sqlite3.connect("loan_applicants.db")
    df = pd.read_sql_query("SELECT * FROM applicants", conn)
    conn.close()
    return df

def update_applicant(applicant_id, column, new_value):
    conn = sqlite3.connect("loan_applicants.db")
    c = conn.cursor()
    c.execute(f"UPDATE applicants SET {column} = ? WHERE id = ?", (new_value, applicant_id))
    conn.commit()
    conn.close()

    # Visual feedback on update (you already had this)
    st.snow()

# -------------------
# Simple CSS for big emoji
# -------------------
st.markdown("""
<style>
  .emoji-big {
    font-size: 64px;
    line-height: 1.1;
    filter: drop-shadow(0 6px 10px rgba(0,0,0,0.15));
    margin: 0.3rem 0 0.8rem 0;
  }
</style>
""", unsafe_allow_html=True)

# -------------------
# Streamlit UI
# -------------------
st.title("📊 Loan Prediction App with Database & Graphs")

model = joblib.load("loan_pretrained.pkl")
init_db()

st.sidebar.header("Enter Applicant Details")
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=200)
loan_term = st.sidebar.number_input("Loan Amount Term (months)", min_value=12, value=360)
credit_history = st.sidebar.selectbox("Credit History", [0, 1, 2, 3])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_data = {
    "Married": married,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

st.subheader("🔎 Applicant Data Preview")
st.write(pd.DataFrame([input_data]))

if st.button("Predict and Save"):
    prediction = int(model.predict(pd.DataFrame([input_data]))[0])
    insert_applicant(input_data, prediction)

    # --- Visual feedback added here ---
    if prediction == 1:
        st.success("✅ Loan Approved and Saved!")
        st.balloons()  # 🎈 balloons animation
        st.markdown('<div class="emoji-big">🥳</div>', unsafe_allow_html=True)  # big party emoji
    else:
        st.error("❌ Loan Not Approved but Saved!")
        st.markdown('<div class="emoji-big">😞</div>', unsafe_allow_html=True)  # big sad emoji
    # -------------------------------

# -------------------
# Display Database
# -------------------
st.subheader("📂 Saved Applicants")
applicants_df = fetch_applicants()
st.dataframe(applicants_df, use_container_width=True)

# -------------------
# Edit Option
# -------------------
st.subheader("✏️ Edit Applicant Record")
if not applicants_df.empty:
    applicant_id = st.selectbox("Select Applicant ID to Edit", applicants_df["id"])
    column_to_edit = st.selectbox("Select Column", applicants_df.columns[1:-1])  # exclude id and prediction
    new_value = st.text_input("Enter New Value")

    if st.button("Update Record"):
        update_applicant(applicant_id, column_to_edit, new_value)
        st.success(f"Updated Applicant {applicant_id}: {column_to_edit} → {new_value}")

# -------------------
# Graph-Based Insights
# -------------------
st.subheader("📈 Graph-Based Insights")

if not applicants_df.empty:
    # Loan Approval Distribution
    st.write("### Loan Approval Distribution")
    approval_counts = applicants_df["Prediction"].value_counts()
    st.bar_chart(approval_counts)

    # Average Income by Approval Status
    st.write("### Average Applicant Income by Loan Approval")
    income_by_status = applicants_df.groupby("Prediction")["ApplicantIncome"].mean()
    st.bar_chart(income_by_status)

    # Loan Amount vs Income Scatter Plot
    st.write("### Loan Amount vs Applicant Income")
    fig, ax = plt.subplots()
    colors = applicants_df["Prediction"].map({1: "green", 0: "red"})
    ax.scatter(applicants_df["ApplicantIncome"], applicants_df["LoanAmount"], c=colors)
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Loan Amount vs Income (Green=Approved, Red=Rejected)")
    st.pyplot(fig)

    # Property Area Distribution
    st.write("### Applicants by Property Area")
    st.bar_chart(applicants_df["Property_Area"].value_counts())
else:
    st.info("No applicants saved yet. Add entries to see graphs.")