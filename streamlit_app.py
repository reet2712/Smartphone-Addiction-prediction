# -*- coding: utf-8 -*-
"""
Smartphone Addiction Prediction - Interactive Streamlit Web Application
College SI Lab Machine Learning Project
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smartphone Addiction Prediction",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-box {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
        background-color: #f1f5f9;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    .header-desc {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
        margin-bottom: 1.25rem;
    }
    .card-mild {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    .card-moderate {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    }
    .card-severe {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    }
    .result-title {
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.9;
        font-weight: 600;
    }
    .result-class {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .result-explanation {
        font-size: 1.05rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .disclaimer-box {
        margin-top: 2rem;
        padding: 0.85rem 1rem;
        background-color: #f8fafc;
        border-left: 4px solid #94a3b8;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Artifacts Safely
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "smartphone_addiction_best_model.pkl")
FEATURES_FILE = os.path.join(BASE_DIR, "smartphone_addiction_feature_columns.pkl")
CLASSES_FILE = os.path.join(BASE_DIR, "smartphone_addiction_target_classes.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "smartphone_addiction_model_metadata.pkl")

@st.cache_resource(show_spinner="Loading machine learning model...")
def load_artifacts():
    missing = []
    for filepath in [MODEL_FILE, FEATURES_FILE, CLASSES_FILE, METADATA_FILE]:
        if not os.path.exists(filepath):
            missing.append(os.path.basename(filepath))
    
    if missing:
        return None, None, None, None, f"Missing required file(s): {', '.join(missing)}"
    
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_FILE, "rb") as f:
            features = pickle.load(f)
        with open(CLASSES_FILE, "rb") as f:
            classes = pickle.load(f)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        return model, features, classes, metadata, None
    except Exception as e:
        return None, None, None, None, f"Failed to deserialize model artifacts: {str(e)}"

model, feature_columns, target_classes, model_metadata, load_error = load_artifacts()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">📱 Smartphone Addiction Prediction</h1>
    <p class="header-desc">An educational machine learning tool that assesses smartphone usage habits and lifestyle factors to predict addiction risk levels (Mild, Moderate, or Severe).</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ **Model Loading Error**: {load_error}")
    st.info("Please ensure that all required `.pkl` model artifact files are present in the application directory.")
    st.stop()

# ---------------------------------------------------------
# Input Form for Required 12 Features
# ---------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Enter Usage & Lifestyle Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=10,
            max_value=100,
            value=22,
            step=1,
            help="Your age in years."
        )
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male", "Other"],
            index=0,
            help="Select your gender (Female, Male, or Other as represented in dataset)."
        )
        daily_screen_time_hours = st.number_input(
            "Daily Screen Time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=5.5,
            step=0.25,
            help="Total active smartphone screen hours on typical weekdays."
        )
        social_media_hours = st.number_input(
            "Social Media Usage (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=2.5,
            step=0.25,
            help="Hours spent on platforms such as Instagram, TikTok, YouTube, X, etc."
        )
        gaming_hours = st.number_input(
            "Gaming Usage (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.25,
            help="Hours spent playing mobile games."
        )
        work_study_hours = st.number_input(
            "Work / Study Usage (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.25,
            help="Hours spent on academic lectures, study tools, or work productivity."
        )

    with col2:
        sleep_hours = st.number_input(
            "Nightly Sleep (hours)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.25,
            help="Average sleep duration per night."
        )
        notifications_per_day = st.number_input(
            "Notifications per day",
            min_value=0,
            max_value=1000,
            value=120,
            step=5,
            help="Estimated daily notification count received."
        )
        app_opens_per_day = st.number_input(
            "App Opens per day",
            min_value=0,
            max_value=1000,
            value=75,
            step=5,
            help="Estimated number of times phone is unlocked or checked."
        )
        weekend_screen_time = st.number_input(
            "Weekend Screen Time (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=6.5,
            step=0.25,
            help="Average daily smartphone screen hours during weekends."
        )
        stress_level = st.selectbox(
            "Stress Level",
            options=["Low", "Medium", "High"],
            index=1,
            help="Perceived level of mental stress."
        )
        academic_work_impact = st.selectbox(
            "Impact on Work / Studies",
            options=["No", "Yes"],
            index=0,
            help="Whether smartphone usage negatively affects your work or academic performance."
        )
        
    submitted = st.form_submit_button("Predict Addiction Level", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Prediction Execution & Rule Handling
# ---------------------------------------------------------
if submitted:
    # 1. Feature sanity warning (non-blocking)
    activity_sum = social_media_hours + gaming_hours + work_study_hours
    if activity_sum > daily_screen_time_hours:
        st.warning(
            f"ℹ️ Note: The sum of Social Media, Gaming, and Work/Study ({activity_sum:.2f}h) "
            f"exceeds total Daily Screen Time ({daily_screen_time_hours:.2f}h)."
        )

    # 2. Build input DataFrame strictly following expected feature columns & ordering
    input_dict = {
        "age": age,
        "gender": gender,
        "daily_screen_time_hours": daily_screen_time_hours,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact
    }
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # 3. Model inference vs explicit business rule handling
    is_rule_applied = False
    rule_note = None
    prediction = None
    prob_dict = None

    # Application rule for out-of-distribution screen time
    if daily_screen_time_hours >= 13.0:
        is_rule_applied = True
        prediction = "Severe"
        rule_note = (
            "⚠️ Application Rule Notice: Daily screen time is 13 hours or higher, which is outside the "
            "training range of the original survey dataset (3.0 to 12.0 hours). By project-specific rule, "
            "this usage is categorized as Severe. This is a heuristic business rule, not a learned statistical prediction."
        )
    else:
        try:
            raw_pred = model.predict(input_df)[0]
            prediction = str(raw_pred)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_dict = {cls_name: float(p) for cls_name, p in zip(model.classes_, probs)}
        except Exception as pred_err:
            st.error(f"Prediction Error: {str(pred_err)}")
            st.stop()

    # 4. Result Card Presentation
    card_class = {
        "Mild": "card-mild",
        "Moderate": "card-moderate",
        "Severe": "card-severe"
    }.get(prediction, "card-mild")

    explanation_map = {
        "Mild": "Low smartphone addiction risk. Device habits indicate balanced screen exposure with minimal disruption to sleep and daily activities.",
        "Moderate": "Moderate smartphone addiction risk. Screen usage and notification checking are noticeable and may warrant mindful boundary-setting.",
        "Severe": "Severe smartphone addiction risk. Usage patterns reflect intense screen exposure, frequent checking, or lifestyle displacement."
    }

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-title">Predicted Addiction Level</div>
        <div class="result-class">{prediction}</div>
        <div class="result-explanation">{explanation_map.get(prediction, '')}</div>
    </div>
    """, unsafe_allow_html=True)

    if is_rule_applied and rule_note:
        st.warning(rule_note)

    # 5. Optional Probability Breakdown
    if prob_dict is not None and not is_rule_applied:
        st.markdown("### 📊 Model Prediction Probabilities")
        prob_cols = st.columns(len(prob_dict))
        for col, (cls_name, prob_val) in zip(prob_cols, prob_dict.items()):
            col.metric(label=f"{cls_name}", value=f"{prob_val * 100:.1f}%")

        # Clean native Streamlit progress bars
        for cls_name, prob_val in prob_dict.items():
            st.write(f"**{cls_name}**: {prob_val * 100:.1f}%")
            st.progress(float(prob_val))

# ---------------------------------------------------------
# Educational Disclaimer
# ---------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    <strong>Educational Project Disclaimer:</strong> This application was developed as a college SI Lab Machine Learning classification project. 
    Predictions are generated for educational and analytical demonstration purposes only. It is not a clinical diagnostic tool or medical evaluation.
</div>
""", unsafe_allow_html=True)
