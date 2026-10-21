<div align="center">

# 📱 Smartphone Addiction Prediction

### *Machine Learning Prediction of Digital Habits, Psychological Stress, and Addiction Risk*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/reet2712/Smartphone-Addiction-prediction)

</div>

---

## 📌 Project Overview

With modern smartphones serving as hubs for communication, entertainment, education, and work, excessive usage often leads to compulsive screen behavior and attention fragmentation. This project delivers an **interactive Machine Learning prediction web application** designed to:

1. **Predict smartphone addiction severity** (`Mild`, `Moderate`, `Severe`) using a trained **Random Forest** classification pipeline.
2. **Analyze daily screen usage** across discrete habit categories (Social Media, Gaming, Work/Study, Idle Screen Time).
3. **Benchmark individual metrics** against population distributions derived from an empirical dataset of **7,500 surveyed individuals**.
4. **Provide personalized habit insights** regarding sleep debt, notification frequency, and screen balance.

---

## ✨ Key Features & Interactive Capabilities

### 🎯 1. Interactive Assessment & Real-Time Prediction
- **Dynamic Persona Presets**: 1-click loading of archetypal usage personas (*Casual Social Scroller*, *Exam-Stressed Student*, *Hardcore Mobile Gamer*, *Balanced Minimalist*).
- **Addiction Severity Gauge**: Semicircular indicator visually demarcating risk thresholds (Green / Amber / Red).
- **Multi-Class Confidence Distribution**: Horizontal bar charts displaying model probability across all risk tiers (`Mild`, `Moderate`, `Severe`).
- **Screen Time Donut Chart**: Proportional breakdown of Social Media, Mobile Gaming, Work/Study, and Other apps.
- **Habit Benchmark Radar Chart**: Real-time multi-axial comparison of your metrics against population norms.
- **Actionable Habit Insights**: Dynamic recommendations addressing sleep duration, notification management, and focus balance.

### 📊 2. Exploratory Data Analytics (EDA)
- **Cohort Metrics**: Interactive metrics tracking median screen time, sleep duration, and severe addiction prevalence.
- **Distribution Visualizations**: Box plots and scatter distributions demonstrating correlations between screen time, sleep duration, and addiction tiers.
- **Correlation Heatmap**: Multi-variable Pearson correlation matrix mapping interdependencies across digital behaviors.
- **Cross-Tabulation**: Interactive examination of stress levels and academic/work impact across addiction classes.
- **Raw Data Viewer**: Filterable preview of survey records from the 7,500-sample dataset.

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    A[User Digital Inputs<br/>Screen Time, Social, Gaming, Sleep, Stress] --> B[Feature Pipeline]
    B --> C[SimpleImputer<br/>Missing Value Handling]
    C --> D[StandardScaler & OneHotEncoder]
    D --> E[Random Forest Classifier<br/>300 Trees, Balanced Class Weights]
    E --> F[Inference Engine]
    F --> G[Predicted Severity Tier<br/>Mild / Moderate / Severe]
    F --> H[Class Probabilities<br/>% Confidence]
    G & H --> I[Streamlit Interactive Dashboard]
    I --> J[Plotly Severity Gauge]
    I --> K[Activity Donut Chart]
    I --> L[Cohort Radar Benchmark]
    I --> M[Actionable Habit Insights]
```

---

## 📂 Project Repository Structure

```text
Smartphone-Addiction-prediction/
├── .streamlit/
│   └── config.toml                  # Streamlit visual theme & server parameters
├── .gitignore                       # Git ignored files & environments
├── requirements.txt                 # Exact dependencies for cloud deployment
├── streamlit_app.py                 # Interactive Streamlit Web Application
├── app.py                           # Application entrypoint wrapper
├── main.py                          # Legacy Flask prototype
├── Smartphone_Usage_And_Addiction_Analysis.csv        # 7,500-sample dataset
├── Smartphone_Usage_Classification_and_PKL_Download.ipynb # Model training & EDA notebook
├── smartphone_addiction_best_model.pkl                # Trained Random Forest pipeline
├── smartphone_addiction_feature_columns.pkl           # Feature schema order
├── smartphone_addiction_target_classes.pkl            # Target class labels
├── smartphone_addiction_model_metadata.pkl            # Model metadata & evaluation criteria
└── README.md                        # Documentation & deployment guide
```
Open your browser at `http://localhost:8501` to interact with the application.

## 🔬 Model Performance & Feature Space

The predictive model assesses 12 input features:

| Feature Name | Description | Type | Range / Domain |
|:---|:---|:---|:---|
| `age` | User chronological age | Numeric | 16 – 80 |
| `gender` | Identified gender category | Categorical | Female, Male, Other |
| `daily_screen_time_hours` | Active weekday screen hours | Numeric | 1.0 – 18.0 |
| `social_media_hours` | Hours on social & short-video apps | Numeric | 0.0 – 12.0 |
| `gaming_hours` | Mobile gaming engagement | Numeric | 0.0 – 10.0 |
| `work_study_hours` | Productive academic/work use | Numeric | 0.0 – 12.0 |
| `sleep_hours` | Average nightly sleep | Numeric | 2.0 – 12.0 |
| `notifications_per_day` | Total daily push notifications | Numeric | 10 – 800 |
| `app_opens_per_day` | Phone pick-ups & unlocks per day | Numeric | 10 – 500 |
| `weekend_screen_time` | Average daily weekend screen hours | Numeric | 1.0 – 20.0 |
| `stress_level` | Self-reported stress indicator | Categorical | Low, Medium, High |
| `academic_work_impact` | Work or study degradation | Categorical | Yes, No |

- 
 [Reet Advani](https://github.com/reet2712)
- **Repository**: [https://github.com/reet2712/Smartphone-Addiction-prediction](https://github.com/reet2712/Smartphone-Addiction-prediction)
