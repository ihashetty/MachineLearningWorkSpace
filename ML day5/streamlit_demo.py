import math
import sqlite3
from contextlib import closing

import pandas as pd
import streamlit as st

# Try to load your pretrained model
MODEL = None
MODEL_ERR = None
try:
    import joblib
    MODEL = joblib.load("loan_pretrained.pkl")
except Exception as e:
    MODEL_ERR = str(e)

# -----------------------------
# Database Helpers
# -----------------------------
DB_PATH = "loan_applicants.db"

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS applicants (
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
            )
            """
        )
        conn.commit()

def insert_applicant(row: dict, prediction: int):
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as c:
        c.execute(
            """INSERT INTO applicants
               (Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
                Loan_Amount_Term, Credit_History, Property_Area, Prediction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["Married"],
                row["Education"],
                row["Self_Employed"],
                int(row["ApplicantIncome"]),
                int(row["CoapplicantIncome"]),
                int(row["LoanAmount"]),
                int(row["Loan_Amount_Term"]),
                int(row["Credit_History"]),
                row["Property_Area"],
                int(prediction),
            ),
        )
        conn.commit()

def fetch_applicants() -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query("SELECT * FROM applicants", conn)

def update_applicant(applicant_id: int, column: str, new_value):
    # Coerce the new_value to the correct type based on schema
    int_cols = {
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Prediction",
    }
    if column in int_cols:
        try:
            new_value = int(float(new_value))
        except ValueError:
            st.error(f"❌ '{column}' must be a number.")
            return False

    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as c:
        # Column name comes from a controlled selectbox → safe to format
        c.execute(f"UPDATE applicants SET {column} = ? WHERE id = ?", (new_value, applicant_id))
        conn.commit()
    return True

# -----------------------------
# Prediction Helper
# -----------------------------
INPUT_ORDER = [
    "Married",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

def predict_with_model(row: dict) -> int:
    """
    Returns 1/0 using loaded model. Raises a user-friendly error
    if model isn't available or schema mismatch occurs.
    """
    if MODEL is None:
        raise RuntimeError(
            "Model not found or failed to load. "
            f"Details: {MODEL_ERR or 'Unknown error'}"
        )
    X = pd.DataFrame([row], columns=INPUT_ORDER)
    try:
        pred = MODEL.predict(X)[0]
        return int(pred)
    except Exception as e:
        raise RuntimeError(
            "Prediction failed. Ensure your saved model can handle these columns "
            f"and categorical types: {INPUT_ORDER}. Error: {e}"
        )

# -----------------------------
# Streamlit Page Setup & Styles
# -----------------------------
st.set_page_config(
    page_title="📊 Loan Prediction App with Database",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 1.4rem; }
        .hero {
            background: linear-gradient(135deg, #2E86DE 0%, #6C63FF 100%);
            color: white; padding: 1rem 1.2rem; border-radius: 14px;
            margin-bottom: 1rem; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .nice-card {
            background: #ffffff; border: 1px solid #eaecef; border-radius: 12px;
            padding: 1rem 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }
        .emoji-big { font-size: 52px; filter: drop-shadow(0 6px 10px rgba(0,0,0,0.12)); }
        .scroll-box {
            max-height: 300px; overflow-y: auto; border: 1px solid #eef0f2;
            border-radius: 10px; padding: 0.9rem; background: #fafbfc;
        }
        .stButton > button {
            background-color: #2E86DE !important; color: white !important;
            font-weight: 600; border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h3 style="margin:0;">👋 Loan Prediction App with Database</h3>
        <p style="margin:6px 0 0;">Enter details in the sidebar, predict with your trained model, and save to SQLite. Balloons for approvals 🎈</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧾 Enter Applicant Details")

married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=200, step=10)
loan_term = st.sidebar.number_input("Loan Amount Term (months)", min_value=12, value=360, step=12)
credit_history = st.sidebar.selectbox("Credit History", [0, 1, 2, 3], index=1)
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
    "Property_Area": property_area,
}

# -----------------------------
# Init DB once per session
# -----------------------------
init_db()

# -----------------------------
# Preview & Predict
# -----------------------------
st.subheader("🔎 Applicant Data Preview")
st.write(pd.DataFrame([input_data]))

col_p1, col_p2 = st.columns([1, 2])
with col_p1:
    do_predict = st.button("🔮 Predict and Save")

if do_predict:
    try:
        prediction = predict_with_model(input_data)  # 1 = approved, 0 = not
        insert_applicant(input_data, prediction)

        if prediction == 1:
            st.success("✅ Loan Approved and Saved!")
            st.balloons()  # 🎈
            st.markdown('<div class="emoji-big">🥳</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Loan Not Approved but Saved!")
            st.markdown('<div class="emoji-big">😞</div>', unsafe_allow_html=True)

    except RuntimeError as e:
        st.error(str(e))

st.markdown("---")

# -----------------------------
# Display Saved Applicants
# -----------------------------
st.subheader("📂 Saved Applicants")
applicants_df = fetch_applicants()
st.dataframe(applicants_df, use_container_width=True)

# -----------------------------
# Edit Record (Update)
# -----------------------------
st.subheader("✏️ Edit Applicant Record")
if not applicants_df.empty:
    applicant_id = st.selectbox("Select Applicant ID to Edit", applicants_df["id"])

    # Exclude id and Prediction from editable list if you want to
    editable_cols = [c for c in applicants_df.columns if c not in ("id",)]
    column_to_edit = st.selectbox("Select Column", editable_cols)

    new_value = st.text_input("Enter New Value")
    if st.button("Update Record"):
        ok = update_applicant(int(applicant_id), column_to_edit, new_value)
        if ok:
            st.success(f"Updated Applicant {applicant_id}: {column_to_edit} → {new_value}")
            # Refresh view
            applicants_df = fetch_applicants()
            st.dataframe(applicants_df, use_container_width=True)
else:
    st.info("No applicants saved yet. Run a prediction to insert the first record.")