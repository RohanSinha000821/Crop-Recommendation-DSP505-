## webapp.py – Ensemble of Top 3 Models with Majority Voting and Priority Tie-Break

import streamlit as st
import numpy as np
import pickle
import os
from collections import Counter
from PIL import Image

# Try importing xgboost (needed to unpickle XGBoost model)
try:
    import xgboost  # noqa: F401
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ============================
# Optional Header Image
# ============================

if os.path.exists("crop.png"):
    img = Image.open("crop.png")
    st.image(img, use_column_width=True)

# ============================
# Load Models & Label Encoder
# ============================

models = {}

# 1. RandomForest (primary model)
if os.path.exists("RandomForest_encoded.pkl"):
    with open("RandomForest_encoded.pkl", "rb") as f:
        models["RandomForest"] = pickle.load(f)
else:
    st.error("RandomForest_encoded.pkl is missing.")
    st.stop()

# 2. KNN
if os.path.exists("KNeighborsClassifier.pkl"):
    with open("KNeighborsClassifier.pkl", "rb") as f:
        models["KNN"] = pickle.load(f)

# 3. XGBoost (if available)
if XGB_AVAILABLE and os.path.exists("XGBoost.pkl"):
    with open("XGBoost.pkl", "rb") as f:
        models["XGBoost"] = pickle.load(f)
else:
    # fallback to NaiveBayes if XGBoost not available
    if os.path.exists("NBClassifier.pkl"):
        with open("NBClassifier.pkl", "rb") as f:
            models["NaiveBayes"] = pickle.load(f)

# Load label encoder
if os.path.exists("label_encoder.pkl"):
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
else:
    st.error("label_encoder.pkl missing.")
    st.stop()

# Use only top 3 models
selected_models = list(models.items())[:3]

# ============================
# Recommended Ranges (from dataset)
# ============================

RECOMMENDED_RANGES = {
    "Nitrogen (N)": (0, 140),
    "Phosphorus (P)": (5, 145),
    "Potassium (K)": (5, 205),
    "Temperature (°C)": (8, 43),
    "Humidity (%)": (14, 100),
    "Soil pH": (3.5, 9.9),
    "Rainfall (mm)": (20, 300),
}

# ============================
# Majority Voting with Tie-Break
# ============================

MODEL_PRIORITY = ["RandomForest", "XGBoost", "KNN", "NaiveBayes"]

def predict_with_ensemble(features):
    """
    features: numpy array(1 × 7)
    returns:
        model_preds: {model_name: crop}
        final_crop: crop label after voting
    """

    model_preds = {}

    # Predict using each model
    for name, model in selected_models:
        pred_encoded = model.predict(features)[0]
        crop = label_encoder.inverse_transform([pred_encoded])[0]
        model_preds[name] = crop

    # Majority voting
    votes = list(model_preds.values())
    counts = Counter(votes)
    most_common = counts.most_common()

    # If clear majority → use it
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        final_crop = most_common[0][0]

    else:
        # Tie → use model priority
        for preferred_model in MODEL_PRIORITY:
            if preferred_model in model_preds:
                final_crop = model_preds[preferred_model]
                break

    return model_preds, final_crop

# ============================
# Streamlit UI
# ============================

def main():
    st.markdown(
        "<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS</h1>",
        unsafe_allow_html=True
    )

    st.sidebar.title("Enter Soil & Climate Parameters")
    st.sidebar.write("Recommended ranges based on dataset.")

    # Helper to get values + recommended ranges
    def get_input(label, rec_range, minv, maxv, default):
        return st.sidebar.number_input(
            f"{label} [Recommended: {rec_range[0]}–{rec_range[1]}]",
            min_value=minv,
            max_value=maxv,
            value=default,
            step=0.1,
        )

    nitrogen = get_input("Nitrogen (N)", RECOMMENDED_RANGES["Nitrogen (N)"], 0.0, 200.0, 50.0)
    phosphorus = get_input("Phosphorus (P)", RECOMMENDED_RANGES["Phosphorus (P)"], 0.0, 200.0, 50.0)
    potassium = get_input("Potassium (K)", RECOMMENDED_RANGES["Potassium (K)"], 0.0, 300.0, 50.0)
    temperature = get_input("Temperature (°C)", RECOMMENDED_RANGES["Temperature (°C)"], 0.0, 60.0, 25.0)
    humidity = get_input("Humidity (%)", RECOMMENDED_RANGES["Humidity (%)"], 0.0, 100.0, 60.0)
    ph = get_input("Soil pH", RECOMMENDED_RANGES["Soil pH"], 0.0, 14.0, 6.5)
    rainfall = get_input("Rainfall (mm)", RECOMMENDED_RANGES["Rainfall (mm)"], 0.0, 500.0, 100.0)

    # Prepare numpy input
    features = np.array([[nitrogen, phosphorus, potassium,
                          temperature, humidity, ph, rainfall]])

    st.sidebar.markdown("---")

    if st.sidebar.button("Predict Crop"):
        # Show warnings for out of range values
        warnings = []
        vals = {
            "Nitrogen (N)": nitrogen,
            "Phosphorus (P)": phosphorus,
            "Potassium (K)": potassium,
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "Soil pH": ph,
            "Rainfall (mm)": rainfall,
        }

        for key, value in vals.items():
            rmin, rmax = RECOMMENDED_RANGES[key]
            if not (rmin <= value <= rmax):
                warnings.append(f"- **{key}**: {value} (recommended {rmin}–{rmax})")

        if warnings:
            st.warning("These values are outside recommended ranges:\n\n" + "\n".join(warnings))

        # Model-wise predictions
        st.subheader("Model-wise Predictions")
        model_preds, final_crop = predict_with_ensemble(features)

        for name, crop in model_preds.items():
            st.write(f"**{name}** predicts: `{crop}`")

        st.markdown("---")
        st.success(f"🌾 Final recommended crop (majority voting): **{final_crop}**")

        st.markdown("**Models used:** " + ", ".join(model_preds.keys()))


if __name__ == "__main__":
    main()
