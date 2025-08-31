import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/intrusion_model.pkl"
FEATURES = [
    "FlowDuration", "TotalFwdPackets", "TotalBackwardPackets",
    "FwdPacketLengthMean", "BwdPacketLengthMean",
    "FlowBytesPerSec", "FlowPacketsPerSec"
]

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ------------------------------
# Inject CSS styles
# ------------------------------
st.markdown(
    """
    <style>
    /* Background color */
    .reportview-container {
        background-color: #f0f2f6;
    }
    /* Sidebar styles */
    .sidebar .sidebar-content {
        background-color: #003366;
        color: white;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    /* Primary buttons */
    button[kind="primary"] {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Header image / logo
# ------------------------------
st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=120)

# ------------------------------
# App Title & intro text
# ------------------------------
st.title("📡 NIDS Demo - Intrusion Detection")
st.markdown("""
Predict whether a network flow is **Normal** or **Attack** using a trained model.

### 🔍 How to use this app:
- Use the sidebar to input flow data manually.
- Or upload a CSV file for batch predictions.
- Get quick predictions on network flow security status.

---
""")

# ------------------------------
# Sidebar: Single Flow Input
# ------------------------------
st.sidebar.header("Single Flow Input")
input_data = {}
for f in FEATURES:
    default_val = 0.0 if "Mean" in f or "Bytes" in f else 1
    # Add unique keys to fix Streamlit DuplicateElementId error
    input_data[f] = st.sidebar.number_input(f, value=default_val, key=f)

if st.sidebar.button("Predict Single Flow"):
    df_input = pd.DataFrame([input_data])
    with st.spinner("Predicting single flow..."):
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input).max()
    st.write(f"**Prediction:** {'Attack' if pred == 1 else 'Normal'} (Confidence: {proba:.2f})")

# ------------------------------
# Batch Prediction via CSV Upload
# ------------------------------
st.header("Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV with flow features", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure all features exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0
    df = df[FEATURES]

    with st.spinner("Predicting batch flows..."):
        preds = model.predict(df)

    df_result = df.copy()
    df_result["Prediction"] = ["Attack" if p == 1 else "Normal" for p in preds]
    st.write(df_result)

    # Download button for batch results
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

# ------------------------------
# Extra info section with expander
# ------------------------------
with st.expander("About this model"):
    st.write("""
    This is a Random Forest model trained on a CICIDS2017-inspired dataset.
    It predicts whether network flows are normal or attacks.
    The model was trained with synthetic data for demonstration.
    """)
