# Smartphone Addiction Prediction

An educational Machine Learning project for predicting smartphone addiction risk tiers (**Mild**, **Moderate**, **Severe**) based on lifestyle, device interaction metrics, and psychological stress markers.

---

## 📌 Project Overview

This project was built for the college SI Lab Machine Learning coursework. It trains and deploys a supervised classification model to evaluate digital wellness and device dependency patterns.

- **Target Classes**: `Mild`, `Moderate`, `Severe`
- **Model Type**: scikit-learn `Pipeline` (ColumnTransformer with imputation, scaling, one-hot encoding, and a `RandomForestClassifier`)
- **Dataset**: 7,500 survey responses (`Smartphone_Usage_And_Addiction_Analysis.csv`)
- **Key Features**: 12 behavioral and demographic attributes
- **Web Interface**: Lightweight, responsive Streamlit application

---

## 📂 Repository File Structure

```text
Smartphone-Addiction-prediction/
├── .streamlit/
│   └── config.toml                              # Streamlit theme & server configuration
├── .gitignore                                   # Ignore list for logs, environments, cache
├── app.py                                       # Main Streamlit web application entrypoint
├── streamlit_app.py                             # Mirror/canonical Streamlit script
├── main.py                                      # Legacy local Flask prototype (kept for reference)
├── Smartphone_Usage_Classification_and_PKL_Download.ipynb # Model training & experimentation notebook
├── smartphone_addiction_best_model.pkl          # Trained scikit-learn Random Forest Pipeline
├── smartphone_addiction_feature_columns.pkl     # Feature names in expected order (12 columns)
├── smartphone_addiction_target_classes.pkl      # Target class labels ['Mild', 'Moderate', 'Severe']
├── smartphone_addiction_model_metadata.pkl      # Pipeline configuration and metadata
├── render.yaml                                  # Render deployment blueprint specification
├── requirements.txt                             # Production dependencies
└── README.md                                    # Project documentation and deployment guide
```

---

## ⚙️ Model Input Features

The model requires 12 features in the exact following schema:

| Feature Name | Type | UI Input / Range | Description |
| :--- | :--- | :--- | :--- |
| `age` | Integer | 10 – 100 | Age of user in years |
| `gender` | Categorical | Female, Male, Other | Gender identity |
| `daily_screen_time_hours` | Float | 0.0 – 24.0 | Total weekday screen hours |
| `social_media_hours` | Float | 0.0 – 24.0 | Screen time on social networks |
| `gaming_hours` | Float | 0.0 – 24.0 | Time spent playing mobile games |
| `work_study_hours` | Float | 0.0 – 24.0 | Productive device hours |
| `sleep_hours` | Float | 0.0 – 24.0 | Nightly sleep duration |
| `notifications_per_day` | Integer | 0 – 1000 | Number of notifications received |
| `app_opens_per_day` | Integer | 0 – 1000 | Number of times phone is unlocked |
| `weekend_screen_time` | Float | 0.0 – 24.0 | Daily screen hours on weekends |
| `stress_level` | Categorical | Low, Medium, High | Perceived mental stress |
| `academic_work_impact` | Categorical | No, Yes | Perceived negative impact on studies/work |

*Note: `transaction_id`, `user_id`, and `addicted_label` are excluded from model inputs to prevent data leakage.*

---

## 🔬 High Screen-Time Prediction Rule (13+ Hours)

- **Dataset Training Range**: The empirical dataset's maximum `daily_screen_time_hours` is **12.0 hours**.
- **Out-of-Distribution Behavior**: Screen times $\ge$ 13 hours are outside the training distribution, where decision trees cannot extrapolate beyond their observed split thresholds.
- **Application Business Rule**: For educational demonstration, if `daily_screen_time_hours` $\ge 13.0$, the application returns **Severe** via an explicit, transparent rule and displays an informative banner informing the user that the input exceeds the training distribution.
- **In-Range Behavior**: Inputs with daily screen time $< 13.0$ hours are evaluated directly by the trained `RandomForestClassifier` pipeline.

---

## 💻 Local Setup & Running in PyCharm

### 1. Open Project in PyCharm
1. Open PyCharm and select **Open** -> choose this project directory.
2. Configure a Python 3.10 or 3.11 interpreter:
   - Go to **File** -> **Settings** -> **Project: new_project** -> **Python Interpreter**.
   - Select **Add Interpreter** -> **New Virtual Environment** (Virtualenv).
   - Ensure the Base Python is **Python 3.10** or **Python 3.11**.

### 2. Install Dependencies
Open the PyCharm **Terminal** window (Alt + F12) and run:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
In the PyCharm Terminal:
```bash
streamlit run app.py
```
Streamlit will launch locally and display the URL:
```text
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

---

## 🚀 Cloud Deployment Guide

### Option A: Streamlit Community Cloud (Recommended - Free & Instant)
1. Push your repository to GitHub (see GitHub steps below).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Select your repository: `Smartphone-Addiction-prediction`.
5. Select branch: `main`.
6. Set **Main file path**: `app.py`.
7. Click **Deploy!**.

### Option B: Render (Web Service)
1. Go to [render.com](https://render.com) and log in.
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Select the following settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**.

*(Alternatively, connect via `render.yaml` blueprint included in the root directory).*

---

## 🐙 GitHub Steps to Update Repository

To update your remote GitHub repository:

```bash
# 1. Check changed files
git status

# 2. Stage changes
git add app.py streamlit_app.py requirements.txt render.yaml README.md .gitignore

# 3. Commit changes
git commit -m "fix: implement high screen-time rule, simplify UI for deployment, and add Render configuration"

# 4. Push to GitHub
git push origin main
```

---

## 🛡️ Disclaimer
This project is created strictly for academic and educational demonstration within the college SI Lab. It is not a medical diagnostic tool or psychological instrument.
