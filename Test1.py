import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

MODEL_PATH = "ETO Prediction Model.pkl"

TEMPLATE_PATH = "Input Data Template.xlsx"

FEATURES = [
    "EmployeeID","Branch","Tenure","Salary","Department","JobSatisfaction",
    "WorkLifeBalance","CommuteDistance","MaritalStatus","Education",
    "PerformanceRating","TrainingHours","YearsSincePromotion",
    "EnvironmentSatisfaction"
]

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

def template_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

st.title("ETO Prediction Web Application")

st.subheader("Step 1: Download the Excel Template")
st.download_button(
    label="Download Input Excel Template",
    data=template_bytes(TEMPLATE_PATH),
    file_name="Input Data Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("### Step 2: Fill the template and upload it below")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# Stop until user uploads a file
if uploaded is None:
    st.info("Please upload the completed Excel file to continue")
    st.stop()

# Read file only after upload
df = pd.read_excel(uploaded)
st.subheader("Preview")
st.dataframe(df.head())

if st.button("Submit / Run Model"):
    # Check required columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Load model after submit
    model = joblib.load(MODEL_PATH)

    # Prepare features
    X = df[FEATURES].copy()

    # Align to model expected order (if available)
    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    # Encode non-numeric columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    status_map = {
      0: "Highly Likely to Churn",
      1: "Moderately Likely to Churn",
      2: "Slightly Likely to Churn"
    }

    # Predict + add output column
    df_out = df.drop(columns=["ChurnLikelihood"]).copy()
    df_out["Predictions"] = pd.Series(model.predict(X)).map(status_map)

    st.success("Done!")
    st.subheader("Output Preview")
    st.dataframe(df_out.head())

    st.download_button(
        "Download Output Excel",
        data=to_excel_bytes(df_out),
        file_name="Output Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

