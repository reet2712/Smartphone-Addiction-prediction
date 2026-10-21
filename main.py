from flask import Flask, request, render_template_string
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Get the current project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved model files
try:
    with open(os.path.join(BASE_DIR, "smartphone_addiction_best_model.pkl"), "rb") as file:
        model = pickle.load(file)

    with open(os.path.join(BASE_DIR, "smartphone_addiction_feature_columns.pkl"), "rb") as file:
        feature_columns = pickle.load(file)

    with open(os.path.join(BASE_DIR, "smartphone_addiction_target_classes.pkl"), "rb") as file:
        target_classes = pickle.load(file)

    model_error = None

except Exception as error:
    model = None
    feature_columns = []
    target_classes = []
    model_error = str(error)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smartphone Addiction Prediction</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef2f7;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 750px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px #cccccc;
        }

        h1 {
            text-align: center;
            color: #263b65;
        }

        p {
            text-align: center;
            color: #555555;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        label {
            font-weight: bold;
            display: block;
            margin-bottom: 6px;
        }

        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #cccccc;
            border-radius: 6px;
            box-sizing: border-box;
        }

        button {
            width: 100%;
            padding: 13px;
            margin-top: 22px;
            background: #263b65;
            color: white;
            border: none;
            border-radius: 7px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #405b91;
        }

        .result {
            margin-top: 25px;
            padding: 18px;
            background: #e6f4ea;
            border-radius: 8px;
            text-align: center;
        }

        .error {
            margin-top: 20px;
            padding: 12px;
            background: #ffe5e5;
            color: #a00000;
            overflow-wrap: anywhere;
        }

        .note {
            font-size: 12px;
            margin-top: 20px;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <h1>Smartphone Addiction Prediction</h1>

    <p>Enter your details to predict the dataset-based addiction level.</p>

    {% if model_error %}
        <div class="error">
            Model loading error: {{ model_error }}
            <br><br>
            Check that all required PKL files are in the same folder as main.py.
        </div>
    {% else %}

    <form method="POST">
        <div class="form-grid">

            <div>
                <label>Age</label>
                <input type="number" name="age" min="1" max="120"
                       value="{{ values.get('age', '20') }}" required>
            </div>

            <div>
                <label>Gender</label>
                <select name="gender" required>
                    {% for option in ['Female', 'Male', 'Other'] %}
                    <option value="{{ option }}"
                    {% if values.get('gender', 'Female') == option %}selected{% endif %}>
                        {{ option }}
                    </option>
                    {% endfor %}
                </select>
            </div>

            <div>
                <label>Daily Screen Time (hours)</label>
                <input type="number" name="daily_screen_time_hours"
                       min="0" max="24" step="0.1"
                       value="{{ values.get('daily_screen_time_hours', '6') }}" required>
            </div>

            <div>
                <label>Social Media (hours)</label>
                <input type="number" name="social_media_hours"
                       min="0" max="24" step="0.1"
                       value="{{ values.get('social_media_hours', '3') }}" required>
            </div>

            <div>
                <label>Gaming (hours)</label>
                <input type="number" name="gaming_hours"
                       min="0" max="24" step="0.1"
                       value="{{ values.get('gaming_hours', '1') }}" required>
            </div>

            <div>
                <label>Work/Study (hours)</label>
                <input type="number" name="work_study_hours"
                       min="0" max="24" step="0.1"
                       value="{{ values.get('work_study_hours', '4') }}" required>
            </div>

            <div>
                <label>Sleep (hours)</label>
                <input type="number" name="sleep_hours"
                       min="0" max="24" step="0.1"
                       value="{{ values.get('sleep_hours', '7') }}" required>
            </div>

            <div>
                <label>Notifications per day</label>
                <input type="number" name="notifications_per_day"
                       min="0" max="10000"
                       value="{{ values.get('notifications_per_day', '50') }}" required>
            </div>

            <div>
                <label>App Opens per day</label>
                <input type="number" name="app_opens_per_day"
                       min="0" max="10000"
                       value="{{ values.get('app_opens_per_day', '60') }}" required>
            </div>

            <div>
                <label>Weekend Screen Time (hours)</label>
                <input type="number" name="weekend_screen_time"
                       min="0" max="24" step="0.1"
                       value="{{ values.get('weekend_screen_time', '7') }}" required>
            </div>

            <div>
                <label>Stress Level</label>
                <select name="stress_level" required>
                    {% for option in ['Low', 'Medium', 'High'] %}
                    <option value="{{ option }}"
                    {% if values.get('stress_level', 'Medium') == option %}selected{% endif %}>
                        {{ option }}
                    </option>
                    {% endfor %}
                </select>
            </div>

            <div>
                <label>Academic/Work Impact</label>
                <select name="academic_work_impact" required>
                    {% for option in ['Yes', 'No'] %}
                    <option value="{{ option }}"
                    {% if values.get('academic_work_impact', 'No') == option %}selected{% endif %}>
                        {{ option }}
                    </option>
                    {% endfor %}
                </select>
            </div>

        </div>

        <button type="submit">Predict Addiction Level</button>
    </form>

    {% endif %}

    {% if prediction %}
    <div class="result">
        <h2>Predicted Level: {{ prediction }}</h2>
        {% if confidence is not none %}
        <p>Model confidence: {{ confidence }}%</p>
        {% endif %}
    </div>
    {% endif %}

    {% if prediction_error %}
    <div class="error">{{ prediction_error }}</div>
    {% endif %}

    <p class="note">
        This prediction reflects patterns learned from the dataset.
        It is not a medical diagnosis.
    </p>

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    prediction_error = None
    values = {}

    if request.method == "POST":
        values = request.form.to_dict()

        try:
            if model is None:
                raise ValueError(model_error or "Model could not be loaded.")

            # Read the user's input values
            age = int(request.form["age"])
            daily_screen_time = float(request.form["daily_screen_time_hours"])
            social_media = float(request.form["social_media_hours"])
            gaming = float(request.form["gaming_hours"])
            work_study = float(request.form["work_study_hours"])
            sleep = float(request.form["sleep_hours"])
            notifications = int(request.form["notifications_per_day"])
            app_opens = int(request.form["app_opens_per_day"])
            weekend_screen = float(request.form["weekend_screen_time"])

            # Check valid ranges
            if not 1 <= age <= 120:
                raise ValueError("Age must be between 1 and 120.")

            if any(x < 0 for x in [
                daily_screen_time, social_media, gaming,
                work_study, sleep, weekend_screen, notifications, app_opens
            ]):
                raise ValueError("Values cannot be negative.")

            if any(x > 24 for x in [
                daily_screen_time, social_media, gaming,
                work_study, sleep, weekend_screen
            ]):
                raise ValueError("Hours must be between 0 and 24.")

            # Prepare the input data
            input_data = {
                "age": age,
                "gender": request.form["gender"],
                "daily_screen_time_hours": daily_screen_time,
                "social_media_hours": social_media,
                "gaming_hours": gaming,
                "work_study_hours": work_study,
                "sleep_hours": sleep,
                "notifications_per_day": notifications,
                "app_opens_per_day": app_opens,
                "weekend_screen_time": weekend_screen,
                "stress_level": request.form["stress_level"],
                "academic_work_impact": request.form["academic_work_impact"]
            }

            # Arrange columns in the same order used during training
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_columns)

            # Predict the addiction level
            result = model.predict(input_df)[0]
            prediction = str(result)

            # Get confidence if the model supports probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                confidence = round(float(max(probabilities)) * 100, 2)

        except Exception as error:
            prediction_error = str(error)

    return render_template_string(
        HTML_PAGE,
        prediction=prediction,
        confidence=confidence,
        prediction_error=prediction_error,
        model_error=model_error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
