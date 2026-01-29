import streamlit as st

import joblib

import pandas as pd

import requests

import altair as alt
 
st.set_page_config(page_title="Loan Prediction App", page_icon="📊", layout="wide")

st.title("📊 Loan Prediction App (Frontend via Streamlit)")
 
# ---------- Load model ----------

@st.cache_resource

def load_model():

    return joblib.load("loan_pretrained.pkl")
 
model = load_model()
 
# ---------- Sidebar inputs ----------

st.sidebar.header("Enter Applicant Details")

married = st.sidebar.selectbox("Married", ["Yes", "No"])

education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])

self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)

coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=100)

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=200, step=10)

loan_term = st.sidebar.number_input("Loan Amount Term (months)", min_value=12, value=360, step=12)

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
 
# ---------- Predict & Save ----------

if st.button("Predict and Save"):

    try:

        prediction = int(model.predict(pd.DataFrame([input_data]))[0])

        input_data["Prediction"] = prediction

        res = requests.post("http://localhost:5000/applicants", json=input_data)

        if res.status_code in (200, 201):

            if prediction == 1:

                st.success("🎉 Loan Approved and Saved!")

            else:

                st.error("❌ Loan Not Approved but Saved!")

        else:

            st.error(f"Save failed: {res.status_code} - {res.text}")

    except Exception as e:

        st.error(f"Prediction/Save error: {e}")
 
# ---------- Fetch applicants ----------

st.subheader("📂 Saved Applicants")
 
def fetch_applicants(limit=1000):

    try:

        res = requests.get(f"http://localhost:5000/applicants?limit={limit}")

        if res.status_code == 200:

            return pd.DataFrame(res.json())

        else:

            st.error(f"Could not fetch applicants: {res.status_code}")

            return pd.DataFrame()

    except Exception as e:

        st.error(f"API error: {e}")

        return pd.DataFrame()
 
applicants_df = fetch_applicants()
 
if applicants_df.empty:

    st.info("No applicants found yet. Create one using the sidebar!")

else:

    # Ensure correct dtypes for charts

    int_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Prediction", "id"]

    for c in int_cols:

        if c in applicants_df.columns:

            applicants_df[c] = pd.to_numeric(applicants_df[c], errors="coerce").astype("Int64")
 
    # ---------- Filters for analytics ----------

    with st.expander("🔍 Filter Data"):

        col1, col2 = st.columns(2)

        with col1:

            area_filter = st.multiselect("Filter by Property Area", options=sorted(applicants_df["Property_Area"].dropna().unique().tolist()))

        with col2:

            prediction_filter = st.multiselect("Filter by Prediction (0 = Not Approved, 1 = Approved)", options=[0, 1])
 
    filtered_df = applicants_df.copy()

    if area_filter:

        filtered_df = filtered_df[filtered_df["Property_Area"].isin(area_filter)]

    if prediction_filter:

        filtered_df = filtered_df[filtered_df["Prediction"].isin(prediction_filter)]
 
    st.write("🗃️ Current filtered dataset:")

    st.dataframe(filtered_df, use_container_width=True)
 
    # ---------- KPI: Approval Rate ----------

    approved = int((filtered_df["Prediction"] == 1).sum())

    total = int(filtered_df["Prediction"].count())

    approval_rate = (approved / total * 100) if total > 0 else 0
 
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    with kpi_col1:

        st.metric("✅ Approved", approved)

    with kpi_col2:

        st.metric("📋 Total Applications", total)

    with kpi_col3:

        st.metric("📈 Approval Rate", f"{approval_rate:.1f}%")
 
    st.progress(int(approval_rate))
 
    st.caption("**Baby-level explanation:** Imagine you have 10 apples (applications). If 6 apples are green (approved), the approval rate is 6 out of 10 = 60%.")
 
    # ---------- Chart 1: Income distribution ----------

    st.markdown("### 💸 Income Distribution (Applicant & Coapplicant)")

    income_cols = ["ApplicantIncome", "CoapplicantIncome"]

    long_income = filtered_df[income_cols].melt(value_name="Income", var_name="Type").dropna()
 
    income_hist = alt.Chart(long_income).mark_bar(opacity=0.7).encode(

        x=alt.X("Income:Q", bin=alt.Bin(maxbins=30), title="Income"),

        y=alt.Y("count():Q", title="Number of Applicants"),

        color=alt.Color("Type:N", title="Income Type")

    ).properties(height=300)
 
    st.altair_chart(income_hist, use_container_width=True)

    st.caption("**Baby-level explanation:** This shows how many people earn certain amounts. Like counting how many candies are in each jar of different sizes.")
 
    # ---------- Chart 2: Property area breakdown ----------

    st.markdown("### 🌍 Property Area Breakdown")

    area_counts = filtered_df["Property_Area"].value_counts(dropna=True).reset_index()

    area_counts.columns = ["Property_Area", "Count"]
 
    pie = alt.Chart(area_counts).mark_arc(innerRadius=60).encode(

        theta=alt.Theta("Count:Q"),

        color=alt.Color("Property_Area:N", legend=alt.Legend(title="Area")),

        tooltip=["Property_Area", "Count"]

    ).properties(height=300)
 
    st.altair_chart(pie, use_container_width=True)

    st.caption("**Baby-level explanation:** Think of the circle as a pizza. Each slice shows how many people come from each area (Urban, Semiurban, Rural). Bigger slice = more people.")
 
    # ---------- Chart 3: Credit history vs approval rate ----------

    st.markdown("### 🧠 Credit History vs Approval Rate")

    if "Credit_History" in filtered_df.columns and "Prediction" in filtered_df.columns:

        group = (

            filtered_df.groupby("Credit_History")["Prediction"]

            .agg(total="count", approved=lambda s: int((s == 1).sum()))

            .reset_index()

        )

        group["ApprovalRate%"] = group.apply(

            lambda r: (r["approved"] / r["total"] * 100) if r["total"] > 0 else 0, axis=1

        )
 
        bar = alt.Chart(group).mark_bar().encode(

            x=alt.X("Credit_History:O", title="Credit History (higher is better)"),

            y=alt.Y("ApprovalRate%:Q", title="Approval Rate (%)"),

            tooltip=["Credit_History", "total", "approved", alt.Tooltip("ApprovalRate%:Q", format=".1f")]

        ).properties(height=300)
 
        st.altair_chart(bar, use_container_width=True)

        st.caption("**Baby-level explanation:** If you return borrowed toys (good credit), you’re more likely to get new toys (loan approval). Higher credit history → higher approval chance.")
 
    # ---------- Chart 4: Loan amount vs Approval (scatter) ----------

    st.markdown("### 🎯 Loan Amount vs Approval")

    if "LoanAmount" in filtered_df.columns:

        scatter = alt.Chart(filtered_df.dropna(subset=["LoanAmount"])).mark_circle(size=80, opacity=0.6).encode(

            x=alt.X("LoanAmount:Q", title="Loan Amount"),

            y=alt.Y("ApplicantIncome:Q", title="Applicant Income"),

            color=alt.Color("Prediction:N", title="Approved (1) or Not (0)"),

            tooltip=["id", "LoanAmount", "ApplicantIncome", "Prediction", "Property_Area", "Credit_History"]

        ).properties(height=300)

        st.altair_chart(scatter, use_container_width=True)

        st.caption("**Baby-level explanation:** Each dot is a person. Where the dot is placed shows the loan size and income. Color tells if they were approved (green) or not (red).")
 
# ---------- Edit option ----------

st.subheader("✏️ Edit Applicant Record")

applicant_id = st.number_input("Applicant ID", min_value=1, step=1)

column_to_edit = st.text_input("Column to Edit (e.g., LoanAmount, Prediction, Property_Area)")

new_value = st.text_input("New Value (numbers for numeric fields)")
 
# Helper to auto-cast new_value to correct type to match backend validation

def cast_value(column: str, value: str):

    int_fields = {"ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Prediction"}

    if column in int_fields:

        try:

            return int(value)

        except ValueError:

            st.error(f"'{column}' must be an integer. You entered: {value}")

            return None

    # Treat others as strings

    return value
 
if st.button("Update Record"):

    if not column_to_edit:

        st.error("Please enter a column name to edit.")

    else:

        casted_value = cast_value(column_to_edit, new_value)

        if casted_value is not None:

            try:

                res = requests.put(

                    f"http://localhost:5000/applicants/{applicant_id}",

                    json={"column": column_to_edit, "new_value": casted_value}

                )

                if res.status_code == 200:

                    st.success("Record Updated!")

                else:

                    st.error(f"Update Failed: {res.status_code} - {res.text}")

            except Exception as e:

                st.error(f"Update error: {e}")
 