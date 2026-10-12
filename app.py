"""
MaternaCare AI — Standalone Web Application Server
"""

from __future__ import annotations

from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

from main import (
    CLASS_ORDER,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    get_model_performance_details,
    get_or_create_model,
    predict_risk,
)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

print("Initializing trained model for MaternaCare AI...")
model_payload = get_or_create_model()
trained_model = model_payload["model"] if model_payload else None
model_name = model_payload.get("model_name", "Random Forest") if model_payload else "Random Forest"
print(f"Model ready: {model_name}")


@app.route("/")
def index():
    """Render the main clinical risk dashboard."""
    return render_template(
        "index.html",
        model_name=model_name,
        feature_columns=FEATURE_COLUMNS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict risk level and return probabilities based on 6 inputs."""
    if not trained_model:
        return jsonify({"success": False, "error": "Model not available"}), 500

    data = request.get_json(silent=True) or {}
    values = {}
    missing = []
    invalid = []

    for col in FEATURE_COLUMNS:
        raw_val = data.get(col)
        if raw_val is None or str(raw_val).strip() == "":
            missing.append(col)
            continue
        try:
            values[col] = float(raw_val)
        except (ValueError, TypeError):
            invalid.append(col)

    if missing:
        return jsonify({"success": False, "error": f"Missing inputs: {', '.join(missing)}"}), 400
    if invalid:
        return jsonify({"success": False, "error": f"Invalid numeric values: {', '.join(invalid)}"}), 400

    if values["SystolicBP"] < values["DiastolicBP"]:
        return jsonify({"success": False, "error": "SystolicBP must be greater than or equal to DiastolicBP."}), 400
    if values["Age"] < 5 or values["Age"] > 100:
        return jsonify({"success": False, "error": "Age should reasonably be between 10 and 80 years."}), 400
    if values["BS"] <= 0:
        return jsonify({"success": False, "error": "Blood sugar (BS) must be a positive value."}), 400

    result = predict_risk(trained_model, values)
    pred_label = result["predicted_label"]
    probabilities = result.get("probabilities") or {}

    flags = []
    if values["BS"] >= 11.0:
        flags.append({"type": "high", "text": f"Severe Hyperglycemia: BS is {values['BS']} mmol/L (≥ 11.0 mmol/L)"})
    elif values["BS"] >= 7.8:
        flags.append({"type": "warning", "text": f"Elevated Blood Glucose: BS is {values['BS']} mmol/L (normal: < 7.8)"})

    if values["SystolicBP"] >= 140 or values["DiastolicBP"] >= 90:
        flags.append({"type": "high", "text": f"Stage 2 Hypertension: {values['SystolicBP']:.0f}/{values['DiastolicBP']:.0f} mmHg"})
    elif values["SystolicBP"] >= 130 or values["DiastolicBP"] >= 80:
        flags.append({"type": "warning", "text": f"Elevated / Stage 1 BP: {values['SystolicBP']:.0f}/{values['DiastolicBP']:.0f} mmHg"})

    if values["Age"] >= 35:
        flags.append({"type": "info", "text": f"Advanced Maternal Age: {values['Age']:.0f} years (≥ 35 years)"})

    if values["HeartRate"] > 100:
        flags.append({"type": "warning", "text": f"Tachycardia: Resting Heart Rate {values['HeartRate']:.0f} bpm (> 100 bpm)"})

    return jsonify({
        "success": True,
        "prediction": pred_label,
        "probabilities": probabilities,
        "flags": flags,
        "values": values,
    })


@app.route("/api/performance", methods=["GET"])
def api_performance():
    """Return model performance details, confusion matrices, and comparisons."""
    try:
        details = get_model_performance_details()
        return jsonify({"success": True, "data": details})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serve the generated model evaluation and confusion matrix plots."""
    figures_folder = BASE_DIR / "figures"
    return send_from_directory(str(figures_folder), filename)


if __name__ == "__main__":
    port = 5000
    print("\n=======================================================")
    print(f"🚀 MaternaCare AI Web App running at: http://127.0.0.1:{port}")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=True)
