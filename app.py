import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to estimate the chance they will leave.")

THRESHOLD_FILE = "main.pkl"  # change this if your threshold file has another name


def to_float(obj):
    """Turn whatever was saved (float, numpy value, list, dict) into one number."""
    if isinstance(obj, dict):
        obj = next(iter(obj.values()))
    return float(np.ravel(obj)[0])


# ---------- Load saved files ----------
@st.cache_resource
def load_files():
    model = joblib.load("lr.pkl")
    columns = list(joblib.load("columns.pkl"))
    threshold = to_float(joblib.load(THRESHOLD_FILE))
    return model, columns, threshold


model, columns, threshold = load_files()

yes_no = ["Yes", "No"]
yes_no_internet = ["Yes", "No", "No internet service"]

# ---------- Inputs ----------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", yes_no)
    dependents = st.selectbox("Dependents", yes_no)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", yes_no)
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", yes_no_internet)

with col2:
    backup = st.selectbox("Online Backup", yes_no_internet)
    protection = st.selectbox("Device Protection", yes_no_internet)
    support = st.selectbox("Tech Support", yes_no_internet)
    tv = st.selectbox("Streaming TV", yes_no_internet)
    movies = st.selectbox("Streaming Movies", yes_no_internet)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", yes_no)
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

total = st.number_input("Total Charges", 0.0, 10000.0, float(monthly * tenure))


# ---------- Build the model input ----------
def build_features():
    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )

    # One-hot encode, then force the exact training columns (missing -> 0).
    # Works whether or not drop_first=True was used in training.
    encoded = pd.get_dummies(row)
    encoded = encoded.reindex(columns=columns, fill_value=0).astype(float)
    return encoded


# ---------- Predict ----------
if st.button("Predict"):
    try:
        encoded = build_features()
        prob = model.predict_proba(encoded)[0][1]

        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(float(prob))
        st.caption(f"Decision threshold: {threshold:.2f}")

        if prob >= threshold:
            st.error("⚠️ High risk: this customer is likely to churn.")
        else:
            st.success("✅ Low risk: this customer is likely to stay.")

        with st.expander("Debug: features sent to the model"):
            nonzero = encoded.loc[:, (encoded != 0).any()]
            st.write(nonzero.T.rename(columns={0: "value"}))
            st.write(f"Total features: {encoded.shape[1]}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Common causes: scikit-learn version differs from training, "
            "or columns.pkl names don't match the inputs above."
        )
