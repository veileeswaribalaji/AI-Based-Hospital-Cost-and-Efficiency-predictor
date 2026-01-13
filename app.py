import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


st.set_page_config(page_title="Hospital Cost and Efficiency AI", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #3f51b5 0%, #9aa3c7 45%, #e5e7ec 100%);
        color: #000000;
        font-family: Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #000000 !important;
    }

    .stButton>button {
        background-color: #d6d9e0;
        color: #000000;
        border-radius: 6px;
        border: 1px solid #6f78a8;
    }

    .stButton>button:hover {
        background-color: #c3c8d6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("AI-Based Hospital Cost and Efficiency predictor")

data_file = "Inpatient_Pat.csv"

try:
    df = pd.read_csv(data_file)
except FileNotFoundError:
    st.error("CSV file not found. Place 'Inpatient_Pat.csv' in the project folder.")
    st.stop()

df.columns = df.columns.str.strip()


st.subheader("Dataset Preview")
st.dataframe(df.head())


st.subheader("Select Prediction Target")

target = st.selectbox(
    "Choose the variable to predict",
    [
        "Average Covered Charges",
        "Average Total Payments",
        "Average Medicare Payments"
    ]
)


df_ml = df.copy()

le = LabelEncoder()
df_ml["DRG Definition"] = le.fit_transform(df_ml["DRG Definition"])

X = df_ml.drop(columns=["Patient_ID", target], errors="ignore")
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Treatment Cost Prediction")

user_input = []
for col in X.columns:
    value = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    user_input.append(value)

if st.button("Predict Cost"):
    prediction = model.predict([user_input])
    st.success(f"Predicted Cost: {round(prediction[0], 2)}")

st.subheader("DRG-Based Cost Analysis")

top_drg = (
    df.groupby("DRG Definition")[target]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig1, ax1 = plt.subplots()
top_drg.plot(kind="bar", ax=ax1)
ax1.set_xlabel("DRG Definition")
ax1.set_ylabel("Average Cost")
ax1.set_title("Top 10 High-Cost DRGs")
st.pyplot(fig1)

st.subheader("Hospital Efficiency Dashboard")

fig2, ax2 = plt.subplots()
ax2.scatter(df["Total Discharges"], df[target])
ax2.set_xlabel("Total Discharges")
ax2.set_ylabel(target)
ax2.set_title("Discharge Volume vs Cost")
st.pyplot(fig2)

st.subheader("Key Insights")

st.write(
    "- High-cost DRGs indicate treatments requiring improved cost control.\n"
    "- High discharge volume with low payments may indicate underfunding.\n"
    "- Predictive cost modeling supports effective hospital resource planning."
)

