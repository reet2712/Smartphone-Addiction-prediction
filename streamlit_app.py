# -*- coding: utf-8 -*-
"""
Smartphone Addiction Prediction
College SI Lab — Machine Learning Project
Author: Reet Advani
"""

import os, pickle
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Smartphone Addiction Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────
# GLOBAL CSS  (Terra-inspired clean style)
# ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* remove default top padding */
.block-container { padding-top: 0rem !important; }

/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #4338ca 100%);
    border-radius: 0 0 32px 32px;
    padding: 3.5rem 3rem 3rem 3rem;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    background: rgba(255,255,255,0.15);
    display: inline-block;
    padding: 4px 16px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #c7d2fe;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.2);
}
.hero-headline {
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 1rem 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #c7d2fe;
    line-height: 1.7;
    max-width: 520px;
    margin-bottom: 1.5rem;
}
.hero-badge-row { display: flex; gap: 10px; flex-wrap: wrap; }
.hero-badge {
    background: rgba(255,255,255,0.12);
    color: #e0e7ff;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.18);
}
.hero-img-col {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 8rem;
    line-height: 1;
}

/* ── SECTION HEADING ── */
.section-eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    color: #6366f1;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e1b4b;
    margin: 0 0 0.25rem 0;
}
.section-sub {
    font-size: 0.92rem;
    color: #64748b;
    margin: 0 0 1.5rem 0;
}

/* ── NUMBERED SECTION PILL ── */
.num-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e1b4b;
    margin-bottom: 1rem;
}
.num-circle {
    background: #4338ca;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── SNAPSHOT CARDS ── */
.snap-grid {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.snap-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    min-width: 130px;
}
.snap-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.snap-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s;
    cursor: pointer;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── RESULT CARD ── */
.result-wrap {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
    margin: 1.5rem 0;
}
.res-mild    { background: linear-gradient(135deg,#065f46,#059669); }
.res-moderate{ background: linear-gradient(135deg,#92400e,#d97706); }
.res-severe  { background: linear-gradient(135deg,#7f1d1d,#dc2626); }
.result-eyebrow {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.85;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.result-big {
    font-size: 3rem;
    font-weight: 800;
    margin: 0.2rem 0 0.8rem;
    line-height: 1;
}
.result-desc {
    font-size: 1.05rem;
    line-height: 1.6;
    opacity: 0.92;
    max-width: 500px;
    margin: 0 auto;
}

/* ── PROB BAR ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.7rem;
}
.prob-label {
    width: 90px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #1e293b;
}
.prob-track {
    flex: 1;
    height: 10px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill-mild     { height:100%; background:#059669; border-radius:999px; }
.prob-fill-moderate { height:100%; background:#d97706; border-radius:999px; }
.prob-fill-severe   { height:100%; background:#dc2626; border-radius:999px; }
.prob-pct {
    width: 48px;
    text-align: right;
    font-size: 0.88rem;
    font-weight: 700;
    color: #475569;
}

/* ── RULE NOTICE ── */
.rule-notice {
    background: #fef9c3;
    border: 1px solid #fde047;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #713f12;
    line-height: 1.6;
    margin-top: 0.5rem;
}

/* ── STAT CHIPS ── */
.stat-chip {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-chip-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4338ca;
}
.stat-chip-label {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
    margin-top: 2px;
}

/* ── DISCLAIMER ── */
.disclaimer {
    margin-top: 3rem;
    padding: 1rem 1.25rem;
    background: #f8fafc;
    border-left: 4px solid #cbd5e1;
    border-radius: 4px;
    font-size: 0.83rem;
    color: #64748b;
    line-height: 1.6;
}

/* ── DIVIDER ── */
.section-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 2.5rem 0;
}

/* hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    files = {
        "model":    "smartphone_addiction_best_model.pkl",
        "features": "smartphone_addiction_feature_columns.pkl",
        "classes":  "smartphone_addiction_target_classes.pkl",
        "metadata": "smartphone_addiction_model_metadata.pkl",
    }
    missing = [v for v in files.values() if not os.path.exists(os.path.join(BASE_DIR, v))]
    if missing:
        return None, None, None, None, f"Missing: {', '.join(missing)}"
    try:
        objs = {}
        for k, v in files.items():
            with open(os.path.join(BASE_DIR, v), "rb") as f:
                objs[k] = pickle.load(f)
        return objs["model"], objs["features"], objs["classes"], objs["metadata"], None
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data(show_spinner=False)
def load_dataset():
    path = os.path.join(BASE_DIR, "Smartphone_Usage_And_Addiction_Analysis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model, feature_columns, target_classes, model_metadata, load_error = load_artifacts()
df_raw = load_dataset()
df = df_raw.dropna(subset=["addiction_level"]) if df_raw is not None else None


# ──────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────
left_hero, right_hero = st.columns([3, 1])
with left_hero:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">📱 SI Lab · Machine Learning Project</div>
        <h1 class="hero-headline">Understand your<br>screen habits.<br>Better choices start here.</h1>
        <p class="hero-sub">
            Enter your daily smartphone usage and lifestyle details below.
            Our trained Random Forest model predicts your addiction risk level —
            Mild, Moderate, or Severe — based on real survey data.
        </p>
        <div class="hero-badge-row">
            <span class="hero-badge">🤖 Random Forest Classifier</span>
            <span class="hero-badge">📊 7,500 Survey Responses</span>
            <span class="hero-badge">⚡ Instant Prediction</span>
            <span class="hero-badge">🎓 Educational Tool</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model loading failed: {load_error}")
    st.stop()


# ──────────────────────────────────────────
# DATASET STATS BAR
# ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if df is not None:
    s1, s2, s3, s4, s5 = st.columns(5)
    severe_pct = (df["addiction_level"] == "Severe").mean() * 100
    s1.markdown(f"""<div class="stat-chip"><div class="stat-chip-value">{len(df):,}</div><div class="stat-chip-label">Survey Respondents</div></div>""", unsafe_allow_html=True)
    s2.markdown(f"""<div class="stat-chip"><div class="stat-chip-value">{df['daily_screen_time_hours'].median():.1f}h</div><div class="stat-chip-label">Median Screen Time</div></div>""", unsafe_allow_html=True)
    s3.markdown(f"""<div class="stat-chip"><div class="stat-chip-value">{df['sleep_hours'].mean():.1f}h</div><div class="stat-chip-label">Avg Nightly Sleep</div></div>""", unsafe_allow_html=True)
    s4.markdown(f"""<div class="stat-chip"><div class="stat-chip-value">{df['social_media_hours'].mean():.1f}h</div><div class="stat-chip-label">Avg Social Media</div></div>""", unsafe_allow_html=True)
    s5.markdown(f"""<div class="stat-chip"><div class="stat-chip-value">{severe_pct:.0f}%</div><div class="stat-chip-label">Severe Addiction Cases</div></div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">Step-by-step</div>
<h2 class="section-title">Tell us about your usage</h2>
<p class="section-sub">Fill in the fields below. Use your typical day's numbers for the most accurate prediction.</p>
""", unsafe_allow_html=True)

with st.form("prediction_form"):

    # ── Section 01: Who you are ──────────────────────
    st.markdown("""<div class="num-pill"><span class="num-circle">01</span> About You</div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=22, step=1)
    with c2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    with c3:
        stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1,
                                    help="Your overall perceived daily stress level.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 02: Screen Time ──────────────────────
    st.markdown("""<div class="num-pill"><span class="num-circle">02</span> Screen Time</div>""", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        daily_screen_time_hours = st.number_input(
            "Daily Screen Time (hrs)", min_value=0.0, max_value=24.0, value=5.5, step=0.25,
            help="Total active smartphone screen hours on a typical weekday.")
    with c5:
        weekend_screen_time = st.number_input(
            "Weekend Screen Time (hrs)", min_value=0.0, max_value=24.0, value=6.5, step=0.25,
            help="Average daily screen hours over the weekend.")
    with c6:
        sleep_hours = st.number_input(
            "Nightly Sleep (hrs)", min_value=0.0, max_value=24.0, value=7.0, step=0.25,
            help="Average sleep duration per night.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 03: App Activity ─────────────────────
    st.markdown("""<div class="num-pill"><span class="num-circle">03</span> App Activity &amp; Behaviour</div>""", unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        social_media_hours = st.number_input(
            "Social Media (hrs/day)", min_value=0.0, max_value=24.0, value=2.5, step=0.25,
            help="Instagram, TikTok, YouTube, X, Snapchat etc.")
    with c8:
        gaming_hours = st.number_input(
            "Gaming (hrs/day)", min_value=0.0, max_value=24.0, value=1.0, step=0.25,
            help="Time spent on mobile games.")
    with c9:
        work_study_hours = st.number_input(
            "Work / Study (hrs/day)", min_value=0.0, max_value=24.0, value=2.0, step=0.25,
            help="Productive device usage — emails, lectures, assignments.")

    c10, c11, c12 = st.columns(3)
    with c10:
        notifications_per_day = st.number_input(
            "Notifications / Day", min_value=0, max_value=1000, value=120, step=5,
            help="Total push notifications received daily.")
    with c11:
        app_opens_per_day = st.number_input(
            "App Opens / Day", min_value=0, max_value=1000, value=75, step=5,
            help="Number of times you unlock or check your phone.")
    with c12:
        academic_work_impact = st.selectbox(
            "Negative Impact on Work/Studies?", ["No", "Yes"],
            help="Has phone use caused distraction, missed deadlines, or lower grades?")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SNAPSHOT ─────────────────────────────────────
    st.markdown("""
    <p style="font-size:0.82rem;font-weight:700;color:#94a3b8;
               text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">
        Your snapshot · confirm before predicting
    </p>
    """, unsafe_allow_html=True)

    snap_cols = st.columns(6)
    snap_cols[0].metric("Age", f"{22}yr")          # placeholders shown pre-submit
    snap_cols[1].metric("Screen Time", f"5.50h")
    snap_cols[2].metric("Social Media", f"2.50h")
    snap_cols[3].metric("Sleep", f"7.00h")
    snap_cols[4].metric("Notifications", "120")
    snap_cols[5].metric("App Opens", "75")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Predict My Addiction Level", use_container_width=True, type="primary")


# ──────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────
if submitted:
    # sanity warning
    act_sum = social_media_hours + gaming_hours + work_study_hours
    if act_sum > daily_screen_time_hours:
        st.warning(f"ℹ️ Activity sub-total ({act_sum:.2f}h) exceeds Daily Screen Time ({daily_screen_time_hours:.2f}h). Check your inputs.")

    # live snapshot after submit
    st.markdown("""<p style="font-size:0.82rem;font-weight:700;color:#94a3b8;
    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">
    Your snapshot</p>""", unsafe_allow_html=True)
    sc = st.columns(6)
    sc[0].metric("Age", f"{age}yr")
    sc[1].metric("Screen Time", f"{daily_screen_time_hours:.2f}h")
    sc[2].metric("Social Media", f"{social_media_hours:.2f}h")
    sc[3].metric("Sleep", f"{sleep_hours:.2f}h")
    sc[4].metric("Notifications", str(notifications_per_day))
    sc[5].metric("App Opens", str(app_opens_per_day))

    # build DataFrame
    input_dict = {
        "age": age, "gender": gender,
        "daily_screen_time_hours": daily_screen_time_hours,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact,
    }
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # prediction logic
    is_rule = False
    prediction = None
    prob_dict = None

    if daily_screen_time_hours >= 13.0:
        is_rule = True
        prediction = "Severe"
    else:
        try:
            prediction = str(model.predict(input_df)[0])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_dict = {c: float(p) for c, p in zip(model.classes_, probs)}
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # result colours & text
    res_class_map = {"Mild": "res-mild", "Moderate": "res-moderate", "Severe": "res-severe"}
    icon_map      = {"Mild": "🟢", "Moderate": "🟡", "Severe": "🔴"}
    explain_map   = {
        "Mild":     "Your digital habits appear balanced. Screen exposure is within healthy limits with minimal disruption to sleep and daily life.",
        "Moderate": "Screen usage is noticeable. Elevated social media time or notifications may warrant mindful boundary-setting.",
        "Severe":   "High screen exposure patterns detected. Frequent checking and possible sleep displacement indicate significant digital dependency.",
    }
    fill_map = {"Mild": "prob-fill-mild", "Moderate": "prob-fill-moderate", "Severe": "prob-fill-severe"}

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    lcol, rcol = st.columns([1.2, 1])

    with lcol:
        st.markdown(f"""
        <div class="result-wrap {res_class_map.get(prediction, 'res-mild')}">
            <div class="result-eyebrow">Predicted Addiction Level</div>
            <div class="result-big">{icon_map.get(prediction,'')} {prediction}</div>
            <div class="result-desc">{explain_map.get(prediction,'')}</div>
        </div>
        """, unsafe_allow_html=True)

        if is_rule:
            st.markdown("""
            <div class="rule-notice">
                ⚠️ <strong>Out-of-distribution notice:</strong> Daily screen time ≥ 13 hours exceeds the
                training dataset range (3 – 12 hours). Result is based on a project-specific business rule,
                not a learned statistical prediction.
            </div>""", unsafe_allow_html=True)

    with rcol:
        if prob_dict and not is_rule:
            st.markdown("""
            <p style="font-size:0.82rem;font-weight:700;color:#94a3b8;
            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
            Model confidence breakdown</p>""", unsafe_allow_html=True)

            for cls_name, prob_val in prob_dict.items():
                pct = prob_val * 100
                fill = fill_map.get(cls_name, "prob-fill-mild")
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{cls_name}</div>
                    <div class="prob-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="padding:2rem;background:#f8fafc;border-radius:16px;
                        border:1px solid #e2e8f0;text-align:center;color:#94a3b8;font-size:0.9rem;">
                Probability breakdown not available<br>for out-of-distribution inputs.
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# DATASET ANALYTICS SECTION
# ──────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">Exploratory Data Analysis</div>
<h2 class="section-title">What the data tells us</h2>
<p class="section-sub">Insights derived from 7,500 survey respondents across demographic and digital behaviour dimensions.</p>
""", unsafe_allow_html=True)

if df is not None:
    # ── Class distribution ─────────────────────
    st.markdown("#### 📊 Addiction Level Distribution")
    counts = df["addiction_level"].value_counts().reindex(["Mild", "Moderate", "Severe"])
    total  = counts.sum()
    col_a, col_b, col_c = st.columns(3)
    colour_map = {"Mild": "#059669", "Moderate": "#d97706", "Severe": "#dc2626"}

    for col_widget, (label, cnt) in zip([col_a, col_b, col_c], counts.items()):
        pct = cnt / total * 100
        col_widget.markdown(f"""
        <div style="background:{colour_map[label]};color:white;border-radius:16px;
                    padding:1.4rem 1.2rem;text-align:center;">
            <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;opacity:0.85;">{label}</div>
            <div style="font-size:2.4rem;font-weight:800;line-height:1.1;margin:6px 0;">{cnt:,}</div>
            <div style="font-size:0.9rem;opacity:0.9;">{pct:.1f}% of respondents</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mean stats by addiction level ─────────────
    st.markdown("#### 📈 Average Metrics by Addiction Level")
    group_cols = ["daily_screen_time_hours", "social_media_hours", "sleep_hours",
                  "gaming_hours", "notifications_per_day", "app_opens_per_day"]
    group_labels = ["Screen Time (h)", "Social Media (h)", "Sleep (h)",
                    "Gaming (h)", "Notifications/day", "App Opens/day"]

    means = df.groupby("addiction_level")[group_cols].mean().reindex(["Mild", "Moderate", "Severe"])

    row1 = st.columns(3)
    row2 = st.columns(3)
    rows = row1 + row2

    for i, (col_key, col_label) in enumerate(zip(group_cols, group_labels)):
        vals = means[col_key]
        max_val = vals.max()
        with rows[i]:
            st.markdown(f"**{col_label}**")
            for lvl in ["Mild", "Moderate", "Severe"]:
                v   = vals[lvl]
                pct = (v / max_val * 100) if max_val > 0 else 0
                col = colour_map[lvl]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
                    <div style="width:68px;font-size:0.82rem;font-weight:600;color:#475569;">{lvl}</div>
                    <div style="flex:1;height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;">
                        <div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:999px;"></div>
                    </div>
                    <div style="width:42px;text-align:right;font-size:0.82rem;font-weight:700;color:#334155;">{v:.1f}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Stress & impact breakdown ──────────────────
    st.markdown("#### 🧠 Stress Level & Academic Impact Breakdown")
    col_stress, col_impact = st.columns(2)

    with col_stress:
        st.markdown("**Stress Level Distribution per Addiction Tier**")
        stress_ct = pd.crosstab(df["addiction_level"], df["stress_level"]).reindex(["Mild", "Moderate", "Severe"])
        for lvl in ["Mild", "Moderate", "Severe"]:
            row_total = stress_ct.loc[lvl].sum()
            st.markdown(f"<div style='font-size:0.8rem;font-weight:700;color:{colour_map[lvl]};margin:6px 0 2px;'>{lvl}</div>", unsafe_allow_html=True)
            s_cols = st.columns(3)
            for idx, stress in enumerate(["Low", "Medium", "High"]):
                cnt_s = stress_ct.loc[lvl, stress] if stress in stress_ct.columns else 0
                s_cols[idx].markdown(f"""
                <div style="background:#f1f5f9;border-radius:8px;padding:6px 10px;text-align:center;">
                    <div style="font-size:0.72rem;color:#94a3b8;font-weight:600;">{stress}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{cnt_s:,}</div>
                </div>""", unsafe_allow_html=True)

    with col_impact:
        st.markdown("**Negative Work/Study Impact per Addiction Tier**")
        imp_ct = pd.crosstab(df["addiction_level"], df["academic_work_impact"]).reindex(["Mild", "Moderate", "Severe"])
        for lvl in ["Mild", "Moderate", "Severe"]:
            row_total = imp_ct.loc[lvl].sum()
            st.markdown(f"<div style='font-size:0.8rem;font-weight:700;color:{colour_map[lvl]};margin:6px 0 2px;'>{lvl}</div>", unsafe_allow_html=True)
            i_cols = st.columns(2)
            for idx, impact_val in enumerate(["No", "Yes"]):
                cnt_i = imp_ct.loc[lvl, impact_val] if impact_val in imp_ct.columns else 0
                pct_i = cnt_i / row_total * 100 if row_total > 0 else 0
                i_cols[idx].markdown(f"""
                <div style="background:#f1f5f9;border-radius:8px;padding:6px 10px;text-align:center;">
                    <div style="font-size:0.72rem;color:#94a3b8;font-weight:600;">Impact: {impact_val}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{cnt_i:,}</div>
                    <div style="font-size:0.72rem;color:#64748b;">{pct_i:.0f}%</div>
                </div>""", unsafe_allow_html=True)

    # ── Raw data preview ──────────────────────────
    with st.expander("📋 View raw dataset (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)

else:
    st.info("Dataset file `Smartphone_Usage_And_Addiction_Analysis.csv` not found — analytics section is unavailable.")


# ──────────────────────────────────────────
# DISCLAIMER
# ──────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <strong>Educational Disclaimer:</strong> This tool was built for the college SI Lab Machine Learning coursework.
    Predictions are for educational demonstration only and do not constitute medical, clinical, or psychological advice.
    Always consult a qualified professional for health-related concerns.
</div>
<br>
""", unsafe_allow_html=True)
