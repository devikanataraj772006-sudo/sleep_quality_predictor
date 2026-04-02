import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# ----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("sleep_data.csv")
# -----------------------------
# SELECT IMPORTANT COLUMNS
# -----------------------------
data = data[[
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "BMI Category",
    "Quality of Sleep"
]]
# -----------------------------
# CONVERT TEXT TO NUMBER
# -----------------------------
data["BMI Category"] = data["BMI Category"].map({
    "Normal": 0,
    "Overweight": 1,
    "Obese": 2
})
# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = data.drop("Quality of Sleep", axis=1)
y = data["Quality of Sleep"]   # already numeric (1–10)
# Train model
model = RandomForestClassifier()
model.fit(X, y)
# -----------------------------
# UI
# -----------------------------
st.title("😴 Sleep Quality Predictor")
sleep = st.number_input("Sleep Duration (hours)", 0.0, 12.0, 6.0)
activity = st.number_input("Physical Activity Level (minutes)", 0, 120, 30)
stress = st.slider("Stress Level", 0, 10, 5)
heart = st.number_input("Heart Rate", 40, 120, 70)
steps = st.number_input("Daily Steps", 0, 20000, 5000)
bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
# Convert input
bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
# ----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Sleep Quality"):

    input_data = [[
        sleep,
        activity,
        stress,
        heart,
        steps,
        bmi_map[bmi]
    ]]
    prediction = model.predict(input_data)
    score = prediction[0]
    # Convert score → label
    if score >= 7:
        result = "Good 😴"
    elif score >= 5:
        result = "Average 😐"
    else:
        result = "Poor 😫"
    st.success(f"Sleep Score: {score}")
    st.subheader(f"Sleep Quality: {result}")




