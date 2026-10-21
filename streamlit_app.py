# -*- coding: utf-8 -*-
"""
Smartphone Addiction Prediction - Interactive Streamlit Web Application
Author: Reet Advani
GitHub: https://github.com/reet2712/Smartphone-Addiction-prediction
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #C7D2FE !important;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 400;
    }
    .badge-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-right: 0.5rem;
        backdrop-filter: blur(4px);
    }
    
    /* Result Badges */
    .result-badge {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px -3px rgba(0,0,0,0.15);
    }
    .badge-mild {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    }
    .badge-moderate {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
    }
    .badge-severe {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
    }
    
    /* Metric pill */
    .metric-pill {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        border-left: 4px solid #4F46E5;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .metric-pill-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Artifacts and Data
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model_artifacts():
    try:
        with open(os.path.join(BASE_DIR, "smartphone_addiction_best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "smartphone_addiction_feature_columns.pkl"), "rb") as f:
            features = pickle.load(f)
        with open(os.path.join(BASE_DIR, "smartphone_addiction_target_classes.pkl"), "rb") as f:
            classes = pickle.load(f)
        with open(os.path.join(BASE_DIR, "smartphone_addiction_model_metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        return model, features, classes, metadata, None
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data(show_spinner="Loading usage dataset...")
def load_dataset():
    csv_path = os.path.join(BASE_DIR, "Smartphone_Usage_And_Addiction_Analysis.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

model, feature_columns, target_classes, model_metadata, load_error = load_model_artifacts()
dataset_df = load_dataset()

# ---------------------------------------------------------
# Sidebar Configuration & User Presets
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.title("⚙️ Control Panel")
    
    st.markdown("### ⚡ Quick Presets")
    st.caption("Select a representative user persona or calibrate custom values:")
    
    PRESETS = {
        "Manual Input": None,
        "Casual Social Scroller": {
            "age": 22, "gender": "Female", "daily_screen": 6.5, "social_media": 4.0,
            "gaming": 1.0, "work_study": 1.5, "sleep": 7.0, "notifications": 120,
            "app_opens": 85, "weekend_screen": 8.0, "stress": "Medium", "impact": "No"
        },
        "Exam-Stressed Student": {
            "age": 20, "gender": "Male", "daily_screen": 9.5, "social_media": 3.0,
            "gaming": 1.0, "work_study": 5.5, "sleep": 5.0, "notifications": 150,
            "app_opens": 110, "weekend_screen": 10.5, "stress": "High", "impact": "Yes"
        },
        "Hardcore Mobile Gamer": {
            "age": 24, "gender": "Male", "daily_screen": 11.5, "social_media": 2.5,
            "gaming": 5.0, "work_study": 2.0, "sleep": 4.5, "notifications": 210,
            "app_opens": 140, "weekend_screen": 14.0, "stress": "High", "impact": "Yes"
        },
        "Balanced Digital Minimalist": {
            "age": 28, "gender": "Other", "daily_screen": 3.5, "social_media": 1.0,
            "gaming": 0.5, "work_study": 2.0, "sleep": 8.0, "notifications": 40,
            "app_opens": 30, "weekend_screen": 4.0, "stress": "Low", "impact": "No"
        }
    }
    
    selected_preset = st.selectbox("Select Habit Persona", list(PRESETS.keys()), index=0)
    preset_data = PRESETS[selected_preset]
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if model_metadata:
        st.markdown(f"**Algorithm**: `{model_metadata.get('best_model_name', 'Random Forest')}`")
        st.markdown(f"**Features Evaluated**: `{len(feature_columns)}`")
        st.markdown(f"**Target Classes**: `{', '.join(target_classes)}`")
    if dataset_df is not None:
        st.markdown(f"**Training Cohort**: `{len(dataset_df):,} samples`")
        
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/reet2712/Smartphone-Addiction-prediction)")
    st.markdown("Developed by **Reet Advani**")

# ---------------------------------------------------------
# Top Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">📱 Smartphone Addiction & Digital Wellness Intelligence</h1>
    <p class="hero-subtitle">Machine Learning Assessment of Usage Habits, Psychological Stress, and Digital Dependency</p>
    <div style="margin-top: 10px;">
        <span class="badge-tag">🤖 Random Forest AI</span>
        <span class="badge-tag">📊 7,500 Survey Cohort</span>
        <span class="badge-tag">⚡ Real-Time Probability Scoring</span>
        <span class="badge-tag">🌱 Behavioral Optimization</span>
    </div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Error loading model artifacts: {load_error}. Please ensure `.pkl` files exist in project root.")
    st.stop()

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_analytics, tab_batch, tab_about = st.tabs([
    "🎯 Interactive Assessment",
    "📊 Dataset Exploratory Analytics",
    "📁 Batch CSV Prediction",
    "🧠 Model Architecture & Methodology"
])

# ---------------------------------------------------------
# TAB 1: Interactive Assessment
# ---------------------------------------------------------
with tab_predict:
    st.markdown("### 📋 Digital Habit Profile")
    st.caption("Adjust your demographic, behavioral, and psychological parameters below to compute your predicted addiction risk score.")
    
    def val(key, default):
        if preset_data and key in preset_data:
            return preset_data[key]
        return default

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics & Total Time")
        age = st.slider("Age (years)", min_value=16, max_value=80, value=int(val("age", 22)), step=1)
        gender_options = ["Female", "Male", "Other"]
        preset_gender = val("gender", "Female")
        gender_default_idx = gender_options.index(preset_gender) if preset_gender in gender_options else 0
        gender = st.selectbox("Gender", gender_options, index=gender_default_idx)
        daily_screen_time = st.slider(
            "Daily Screen Time (hours)",
            min_value=1.0, max_value=18.0,
            value=float(val("daily_screen", 6.5)), step=0.25,
            help="Total hours spent actively using smartphone per weekday"
        )
        weekend_screen_time = st.slider(
            "Weekend Screen Time (hours)",
            min_value=1.0, max_value=20.0,
            value=float(val("weekend_screen", 8.0)), step=0.25,
            help="Average daily hours on weekends"
        )
        
    with col2:
        st.markdown("#### ⏱️ Activity Breakdown")
        social_media = st.slider(
            "Social Media (hours)",
            min_value=0.0, max_value=12.0,
            value=float(val("social_media", 3.5)), step=0.25,
            help="Instagram, TikTok, YouTube Shorts, X, Snapchat, etc."
        )
        gaming = st.slider(
            "Gaming (hours)",
            min_value=0.0, max_value=10.0,
            value=float(val("gaming", 1.0)), step=0.25,
            help="Mobile gaming, competitive or casual"
        )
        work_study = st.slider(
            "Work & Study (hours)",
            min_value=0.0, max_value=12.0,
            value=float(val("work_study", 2.0)), step=0.25,
            help="Productive work, reading, emails, academic lectures"
        )
        
        sum_activities = social_media + gaming + work_study
        if sum_activities > daily_screen_time:
            st.warning(f"⚠️ Activity breakdown ({sum_activities:.1f}h) exceeds reported Daily Screen Time ({daily_screen_time:.1f}h).")
        else:
            other_time = daily_screen_time - sum_activities
            st.info(f"💡 Other Screen Time (Browsing/Utilities): **{other_time:.1f} hours**")

    with col3:
        st.markdown("#### 🧠 Device Habits & Well-Being")
        sleep_hours = st.slider(
            "Nightly Sleep (hours)",
            min_value=2.0, max_value=12.0,
            value=float(val("sleep", 7.0)), step=0.25
        )
        notifications = st.number_input(
            "Notifications per day",
            min_value=10, max_value=800,
            value=int(val("notifications", 120)), step=10,
            help="Average alerts and push notifications received daily"
        )
        app_opens = st.number_input(
            "App Opens per day",
            min_value=10, max_value=500,
            value=int(val("app_opens", 80)), step=5,
            help="How many times you pick up or unlock phone to check apps"
        )
        
        stress_options = ["Low", "Medium", "High"]
        preset_stress = val("stress", "Medium")
        stress_default_idx = stress_options.index(preset_stress) if preset_stress in stress_options else 1
        stress_level = st.selectbox("Perceived Stress Level", stress_options, index=stress_default_idx)
        
        impact_options = ["No", "Yes"]
        preset_impact = val("impact", "No")
        impact_default_idx = impact_options.index(preset_impact) if preset_impact in impact_options else 0
        academic_work_impact = st.selectbox(
            "Negative Impact on Work/Studies?",
            impact_options, index=impact_default_idx,
            help="Has phone usage caused missed deadlines, distraction, or lower performance?"
        )

    st.markdown("---")
    
    input_data = {
        "age": age,
        "gender": gender,
        "daily_screen_time_hours": daily_screen_time,
        "social_media_hours": social_media,
        "gaming_hours": gaming,
        "work_study_hours": work_study,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications,
        "app_opens_per_day": app_opens,
        "weekend_screen_time": weekend_screen_time,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact
    }
    
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns)
    
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        max_prob = max(probabilities) * 100
    except Exception as pred_err:
        st.error(f"Prediction Error: {pred_err}")
        st.stop()
        
    st.markdown("## 📊 Assessment Results")
    
    badge_class = {
        "Mild": "badge-mild",
        "Moderate": "badge-moderate",
        "Severe": "badge-severe"
    }.get(prediction, "badge-mild")
    
    icon_map = {"Mild": "🟢", "Moderate": "🟡", "Severe": "🔴"}
    summary_map = {
        "Mild": "Low Addiction Risk. Your smartphone interaction patterns demonstrate healthy boundaries, manageable screen time, and balanced sleep.",
        "Moderate": "Moderate Addiction Risk. Smartphone habits are noticeable, with elevated screentime or notification frequency starting to encroach on rest and attention.",
        "Severe": "Severe Addiction Risk. High screen exposure, frequent micro-checking, and sleep displacement indicate significant digital dependency."
    }
    
    st.markdown(f"""
    <div class="result-badge {badge_class}">
        <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9;">Predicted Addiction Level</div>
        <div style="font-size: 2.8rem; font-weight: 800; margin: 0.3rem 0;">{icon_map.get(prediction, '')} {prediction}</div>
        <div style="font-size: 1.1rem; max-width: 700px; margin: 0 auto; line-height: 1.5;">
            {summary_map.get(prediction, '')}
        </div>
        <div style="margin-top: 1rem; font-weight: 600; background: rgba(0,0,0,0.2); display: inline-block; padding: 6px 18px; border-radius: 20px;">
            Model Confidence: {max_prob:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    row1_left, row1_right = st.columns([1, 1])
    
    with row1_left:
        st.markdown("#### 🎛️ Addiction Severity Gauge")
        if "Mild" in prob_dict and "Moderate" in prob_dict and "Severe" in prob_dict:
            severity_score = (prob_dict["Mild"] * 15) + (prob_dict["Moderate"] * 55) + (prob_dict["Severe"] * 90)
        else:
            severity_score = 25 if prediction == "Mild" else (60 if prediction == "Moderate" else 90)
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=severity_score,
            number={'suffix': "/100", 'font': {'size': 32, 'color': '#1E293B'}},
            title={'text': "Composite Risk Index", 'font': {'size': 18, 'color': '#475569'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': "#1E1B4B", 'thickness': 0.28},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 35], 'color': '#D1FAE5'},
                    {'range': [35, 70], 'color': '#FEF3C7'},
                    {'range': [70, 100], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=290, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with row1_right:
        st.markdown("#### 📈 Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Addiction Level": list(prob_dict.keys()),
            "Probability (%)": [v * 100 for v in prob_dict.values()]
        })
        color_map = {"Mild": "#10B981", "Moderate": "#F59E0B", "Severe": "#EF4444"}
        fig_prob = px.bar(
            prob_df,
            x="Probability (%)",
            y="Addiction Level",
            orientation="h",
            text="Probability (%)",
            color="Addiction Level",
            color_discrete_map=color_map
        )
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prob.update_layout(
            height=290,
            xaxis=dict(range=[0, 115], title="Model Confidence (%)"),
            yaxis=dict(title=None, autorange="reversed"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    row2_left, row2_right = st.columns([1, 1])
    
    with row2_left:
        st.markdown("#### 🍩 Screen Time Distribution")
        other_time = max(0.0, daily_screen_time - (social_media + gaming + work_study))
        activity_df = pd.DataFrame({
            "Activity": ["Social Media", "Gaming", "Work/Study", "Other Apps & Browsing"],
            "Hours": [social_media, gaming, work_study, other_time]
        })
        fig_donut = px.pie(
            activity_df,
            names="Activity",
            values="Hours",
            hole=0.55,
            color="Activity",
            color_discrete_sequence=["#6366F1", "#EC4899", "#10B981", "#94A3B8"]
        )
        fig_donut.update_traces(textinfo="label+percent", hoverinfo="label+value+percent")
        fig_donut.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with row2_right:
        st.markdown("#### 🕸️ Benchmark vs Dataset Population Average")
        categories = ['Daily Screen (h)', 'Social Media (h)', 'Sleep Debt (10-h)', 'Notifications / 20', 'App Opens / 15']
        user_metrics = [
            daily_screen_time,
            social_media,
            max(0, 10 - sleep_hours),
            notifications / 20.0,
            app_opens / 15.0
        ]
        cohort_avg = [
            7.50,
            3.27,
            max(0, 10 - 6.74),
            134.26 / 20.0,
            97.83 / 15.0
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_metrics,
            theta=categories,
            fill='toself',
            name='Your Profile',
            line=dict(color='#4F46E5', width=2.5),
            fillcolor='rgba(79, 70, 229, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=cohort_avg,
            theta=categories,
            fill='toself',
            name='Cohort Average',
            line=dict(color='#94A3B8', width=1.5, dash='dash'),
            fillcolor='rgba(148, 163, 184, 0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 16])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=300,
            margin=dict(l=30, r=30, t=10, b=35),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### 💡 Tailored Behavioral & Wellness Plan")
    rec_cols = st.columns(3)
    
    with rec_cols[0]:
        st.markdown("""
        <div class="metric-pill">
            <div class="metric-pill-title">🌙 Sleep & Circadian Hygiene</div>
            <div style="margin-top: 8px; font-size: 0.95rem; color: #334155; line-height: 1.4;">
        """, unsafe_allow_html=True)
        if sleep_hours < 6.5:
            st.markdown("⚠️ **Critical Sleep Deficit**: You are sleeping under 6.5 hours. Implement a strict **no-screen rule 45 minutes before bedtime** to allow melatonin secretion.")
        else:
            st.markdown("✅ **Sufficient Rest**: Your sleep schedule is protective against cognitive burnout. Keep bedtime consistent on weekends.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with rec_cols[1]:
        st.markdown("""
        <div class="metric-pill">
            <div class="metric-pill-title">🔔 Notification Hygiene</div>
            <div style="margin-top: 8px; font-size: 0.95rem; color: #334155; line-height: 1.4;">
        """, unsafe_allow_html=True)
        if notifications > 120 or app_opens > 90:
            st.markdown(f"⚠️ **Attention Fragmentation**: With **{notifications} alerts** and **{app_opens} opens**, try batching non-urgent apps into scheduled morning and evening notification digests.")
        else:
            st.markdown("✅ **Healthy Boundaries**: Alert volume is within a controlled range, minimizing dopaminergic micro-checking.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with rec_cols[2]:
        st.markdown("""
        <div class="metric-pill">
            <div class="metric-pill-title">🧠 Stress & Focus Mitigation</div>
            <div style="margin-top: 8px; font-size: 0.95rem; color: #334155; line-height: 1.4;">
        """, unsafe_allow_html=True)
        if stress_level == "High" or academic_work_impact == "Yes":
            st.markdown("⚠️ **Displaced Focus**: High stress and reported work/academic disruption indicate compulsive phone use as a coping escape. Use the **Pomodoro technique (25 min focus / 5 min walk)**.")
        else:
            st.markdown("✅ **Stable Digital Balance**: Phone usage is not interfering with daily commitments or escalating psychological tension.")
        st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Dataset Exploratory Analytics
# ---------------------------------------------------------
with tab_analytics:
    st.markdown("### 🔍 Exploratory Data Analysis (EDA)")
    st.caption("Insights extracted from the 7,500 survey responses across demographic and digital behavior dimensions.")
    
    if dataset_df is not None:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Cohort Size", f"{len(dataset_df):,} users")
        kpi2.metric("Median Daily Screen Time", f"{dataset_df['daily_screen_time_hours'].median():.1f} hrs")
        kpi3.metric("Avg Nightly Sleep", f"{dataset_df['sleep_hours'].mean():.1f} hrs")
        severe_pct = (dataset_df['addiction_level'] == 'Severe').mean() * 100
        kpi4.metric("Severe Addiction Cases", f"{severe_pct:.1f}%")
        
        st.markdown("---")
        
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("#### Screen Time by Addiction Severity")
            fig_box = px.box(
                dataset_df,
                x="addiction_level",
                y="daily_screen_time_hours",
                color="addiction_level",
                category_orders={"addiction_level": ["None", "Mild", "Moderate", "Severe"]},
                color_discrete_map={"None": "#94A3B8", "Mild": "#10B981", "Moderate": "#F59E0B", "Severe": "#EF4444"},
                labels={"addiction_level": "Addiction Category", "daily_screen_time_hours": "Daily Screen Time (Hours)"}
            )
            fig_box.update_layout(height=350, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, use_container_width=True)
            
        with eda_col2:
            st.markdown("#### Sleep Duration vs Screen Time")
            sample_sub = dataset_df.sample(min(1200, len(dataset_df)), random_state=42)
            fig_scatter = px.scatter(
                sample_sub,
                x="daily_screen_time_hours",
                y="sleep_hours",
                color="addiction_level",
                opacity=0.6,
                category_orders={"addiction_level": ["None", "Mild", "Moderate", "Severe"]},
                color_discrete_map={"None": "#94A3B8", "Mild": "#10B981", "Moderate": "#F59E0B", "Severe": "#EF4444"},
                labels={"daily_screen_time_hours": "Daily Screen Time (h)", "sleep_hours": "Sleep (h)"}
            )
            fig_scatter.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("---")
        eda_col3, eda_col4 = st.columns(2)
        
        with eda_col3:
            st.markdown("#### Correlation Heatmap (Digital Habits)")
            numeric_cols = [
                "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
                "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day"
            ]
            corr_mat = dataset_df[numeric_cols].corr().round(2)
            fig_heat = px.imshow(
                corr_mat,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(color="Correlation")
            )
            fig_heat.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with eda_col4:
            st.markdown("#### Stress Level Breakdown by Addiction Category")
            stress_ct = pd.crosstab(dataset_df['stress_level'], dataset_df['addiction_level'], normalize='index') * 100
            stress_ct = stress_ct.reset_index().melt(id_vars='stress_level', var_name='Addiction Level', value_name='Percentage')
            fig_bar = px.bar(
                stress_ct,
                x="stress_level",
                y="Percentage",
                color="Addiction Level",
                barmode="group",
                color_discrete_map={"None": "#94A3B8", "Mild": "#10B981", "Moderate": "#F59E0B", "Severe": "#EF4444"},
                labels={"stress_level": "Reported Stress Level", "Percentage": "Cohort Proportion (%)"}
            )
            fig_bar.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with st.expander("📋 View & Filter Raw Dataset (First 100 Records)"):
            st.dataframe(dataset_df.head(100), use_container_width=True)
    else:
        st.info("Dataset file `Smartphone_Usage_And_Addiction_Analysis.csv` not found in root directory.")

# ---------------------------------------------------------
# TAB 3: Batch CSV Prediction
# ---------------------------------------------------------
with tab_batch:
    st.markdown("### 📁 Batch Prediction Engine")
    st.caption("Upload a `.csv` file containing multiple user records to predict addiction severity across cohorts simultaneously.")
    
    sample_template = pd.DataFrame([{
        "age": 21,
        "gender": "Female",
        "daily_screen_time_hours": 7.5,
        "social_media_hours": 3.8,
        "gaming_hours": 1.2,
        "work_study_hours": 2.5,
        "sleep_hours": 6.5,
        "notifications_per_day": 140,
        "app_opens_per_day": 95,
        "weekend_screen_time": 9.0,
        "stress_level": "Medium",
        "academic_work_impact": "Yes"
    }, {
        "age": 26,
        "gender": "Male",
        "daily_screen_time_hours": 4.0,
        "social_media_hours": 1.5,
        "gaming_hours": 0.5,
        "work_study_hours": 2.0,
        "sleep_hours": 7.5,
        "notifications_per_day": 50,
        "app_opens_per_day": 40,
        "weekend_screen_time": 4.5,
        "stress_level": "Low",
        "academic_work_impact": "No"
    }])
    
    st.download_button(
        label="📥 Download CSV Input Template",
        data=sample_template.to_csv(index=False),
        file_name="smartphone_prediction_batch_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV file for Batch Evaluation", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded successfully! Loaded {len(batch_df)} rows.")
            
            missing_cols = [c for c in feature_columns if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                eval_features = batch_df[feature_columns].copy()
                batch_preds = model.predict(eval_features)
                batch_probs = model.predict_proba(eval_features)
                
                batch_df["Predicted_Addiction_Level"] = batch_preds
                batch_df["Confidence_Score (%)"] = [round(float(max(p)) * 100, 2) for p in batch_probs]
                
                bcol1, bcol2 = st.columns([1, 2])
                with bcol1:
                    st.markdown("#### Prediction Summary")
                    summary_counts = batch_df["Predicted_Addiction_Level"].value_counts().reset_index()
                    summary_counts.columns = ["Level", "Count"]
                    fig_batch = px.pie(
                        summary_counts,
                        names="Level",
                        values="Count",
                        hole=0.45,
                        color="Level",
                        color_discrete_map={"Mild": "#10B981", "Moderate": "#F59E0B", "Severe": "#EF4444"}
                    )
                    fig_batch.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_batch, use_container_width=True)
                    
                with bcol2:
                    st.markdown("#### Batch Results Preview")
                    st.dataframe(batch_df.head(20), use_container_width=True)
                    
                st.download_button(
                    label="📥 Download Processed Results (CSV)",
                    data=batch_df.to_csv(index=False),
                    file_name="batch_smartphone_addiction_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}")

# ---------------------------------------------------------
# TAB 4: Model Architecture & Transparency
# ---------------------------------------------------------
with tab_about:
    st.markdown("### 🧠 Model Architecture & Methodology")
    
    st.markdown("""
    #### ⚙️ Pipeline Specifications
    The predictive engine uses a composite machine learning pipeline trained with **scikit-learn 1.6.1**:
    1. **Numeric Feature Preprocessing**:
       - `SimpleImputer(strategy='median')` ensures resilience against missing or incomplete responses.
       - `StandardScaler()` scales numerical features (`age`, `screen time`, `sleep`, `notifications`, etc.) to zero mean and unit variance.
    2. **Categorical Feature Preprocessing**:
       - `SimpleImputer(strategy='most_frequent')` for missing categorical indicators.
       - `OneHotEncoder(handle_unknown='ignore')` transforms discrete variables (`gender`) into encoded vector representations.
    3. **Classification Algorithm**:
       - **Random Forest Classifier** with 300 estimators, balanced class weights, and bounded tree depth (`max_depth=12`) to prevent overfitting.
       - Selected as top-performing model against Decision Trees and Logistic Regression using **Macro F1 Score** and **Stratified Test Accuracy**.
    """)
    
    st.markdown("---")
    st.markdown("""
    #### ⚠️ Clinical & Ethical Disclaimer
    - **Demonstration / Behavioral Classification**: This machine learning model evaluates self-reported digital usage patterns and behavioral survey data.
    - **Not a Diagnostic Instrument**: Smartphone addiction is not an official standalone diagnosis in the DSM-5. This application provides risk categorization and behavioral insights for educational, academic, and self-awareness purposes. It does not replace professional clinical psychological evaluation.
    """)
