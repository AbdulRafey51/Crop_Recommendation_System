import streamlit as st
st.set_page_config(page_title="Crop Recommendation System", layout="centered")  # 🔺 MUST BE FIRST

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from groq import Groq  # Ensure this is installed: pip install groq

# --- Load and preprocess the dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    df = df.drop(columns=["N", "P", "K", "ph"])  # Remove unused columns

    for col in ["temperature", "humidity", "rainfall"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        df[col] = np.where(df[col] > upper_limit, upper_limit,
                           np.where(df[col] < lower_limit, lower_limit, df[col]))

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    return df, label_encoder

df, label_encoder = load_data()
X = df[["temperature", "humidity", "rainfall"]]
y = df["label"]

# --- Train the model ---
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(max_depth=7, n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# --- Crop recommendation logic ---
def recommend_crops(temp, hum, rain):
    input_data = np.array([[temp, hum, rain]])
    probs = model.predict_proba(input_data)[0]
    top_indices = np.argsort(probs)[-5:][::-1]
    top_crops = label_encoder.inverse_transform(top_indices)
    return top_crops

# --- LLM-based Crop Insight using Groq ---
def get_crop_insight(crop_name, temp, hum, rain):
    client = Groq(api_key="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")  # Replace with your key

    prompt = f"""
You are an expert agricultural assistant. A user has provided these conditions:

- Temperature: {temp}°C
- Humidity: {hum}%
- Rainfall: {rain} mm

A top crop recommendation is: **{crop_name}**

Please explain:
1. Why is this crop suitable?
2. Estimated yield in kg/hectare.
3. Advice to improve results.
4. Important notes for farmers.

Keep the explanation friendly and useful.
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error from GroqCloud: {e}"

# --- Streamlit UI ---
st.title("🌾 Crop Recommendation System")
st.markdown("Enter your current environmental conditions to get the best crop suggestions.")

# --- User Inputs ---
temp = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=60.0, step=0.1)
hum = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
rain = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# --- Recommendation & LLM Explanation ---
if st.button("🔍 Get Crop Recommendations"):
    top_crops = recommend_crops(temp, hum, rain)
    st.success("✅ Top 5 Recommended Crops:")
    
    for crop in top_crops:
        st.markdown(f"### 🌱 {crop.capitalize()}")
        with st.spinner("Asking Groq LLM for expert insight..."):
            insight = get_crop_insight(crop, temp, hum, rain)
        st.markdown(f"🧠 **LLM Advice:**\n\n{insight}")
        st.markdown("---")

# --- Footer ---
st.markdown("""
---
📌 *Note: This app uses AI and ML models. While it provides helpful recommendations, consult local experts for final decisions.*
""")
