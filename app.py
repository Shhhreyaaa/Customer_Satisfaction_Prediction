from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# =========================
# Load trained model
# =========================
with open("model/customer_satisfaction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# =========================
# UI HOME (ROOT)
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# =========================
# UI Prediction
# =========================
@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    input_data = []

    for feature in feature_columns:
        value = request.form.get(feature)
        input_data.append(float(value))

    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    return render_template(
        "index.html",
        prediction=f"Predicted Customer Satisfaction Rating: {prediction}"
    )


# =========================
# API Prediction (JSON)
# =========================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    input_data = []
    for feature in feature_columns:
        if feature not in data:
            return jsonify({"error": f"Missing feature: {feature}"}), 400
        input_data.append(float(data[feature]))

    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    return jsonify({
        "predicted_customer_satisfaction_rating": int(prediction)
    })


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
