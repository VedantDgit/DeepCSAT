import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="DeepCSAT Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model + Scaler
# -----------------------------
model = joblib.load('csat_model_5features.pkl')
scaler = joblib.load('scaler_5features.pkl')
# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1E1E1E;
        text-align: center;
        font-size: 24px;
        color: #00FFAA;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("📊 DeepCSAT Score Prediction System")
st.write("Predict customer satisfaction score using selected order features.")

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    response_time = st.number_input("Response Time", min_value=0.0, step=1.0)
    connected_handling_time = st.number_input("Connected Handling Time", min_value=0.0, step=1.0)
    item_price = st.number_input("Item Price", min_value=0.0, step=1.0)

with col2:
    month_names = {
        1: "January", 2: "February", 3: "March",
        4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September",
        10: "October", 11: "November", 12: "December"
    }

    weekday_names = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
    }

    order_month = st.selectbox(
        "Order Month",
        list(month_names.keys()),
        format_func=lambda x: month_names[x]
    )

    order_weekday = st.selectbox(
        "Order Weekday",
        list(weekday_names.keys()),
        format_func=lambda x: weekday_names[x]
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict CSAT Score"):

    input_data = np.array([[
        response_time,
        connected_handling_time,
        item_price,
        order_month,
        order_weekday
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Result Display
    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted CSAT Score: {prediction[0]:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit | DeepCSAT Project")