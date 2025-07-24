import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def set_background():
    st.markdown(
        '''
        <style>
        .stApp {
            background-image: url("https://media.istockphoto.com/id/1449244963/photo/prospect-exchange-rate-amid-financial-crisis.jpg?s=2048x2048&w=is&k=20&c=I_Zi0BDaN33Rgy9rfhWdoW8I4qoxJWa3KedekGMowRs=");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        .stMarkdown, .stTextInput > label, .stNumberInput > label, .css-1cpxqw2, .css-qrbaxs, .css-10trblm {
            color: white !important;
        }

        h1.title-black {
            color: black !important;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

# Set background
set_background()

@st.cache_resource
def load_model():
    df = pd.read_csv("UCI_Credit_Card.csv")
    X = df.drop(columns=["ID", "default.payment.next.month"])
    y = df["default.payment.next.month"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

def predict(model, scaler, user_input):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]
    return prediction, probability

st.set_page_config(page_title="Credit Scoring Predictor", layout="centered")
st.markdown('<h1 style="color:white;">💳 Creditworthiness Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:white;">📥 Enter Financial Details</h3>', unsafe_allow_html=True)

model, scaler, feature_names = load_model()

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("🔮 Predict Creditworthiness"):
    pred, prob = predict(model, scaler, user_input)
    if pred == 1:
        st.error(f"❌ This person is likely to default! (Risk Score: {prob:.2f})")
    else:
        st.success(f"✅ This person is likely creditworthy! (Risk Score: {prob:.2f})")
