# MaternaCare AI — Maternal Health Risk Level Classification (Web Application)

Interactive clinical machine learning web dashboard built for **HDSSBL701: Data Science for Health and Social Care: Lab**.

Trained on the **UCI Maternal Health Risk dataset (Dataset #863)** using six maternal biomarkers:
1. `Age` (years)
2. `SystolicBP` (mmHg)
3. `DiastolicBP` (mmHg)
4. `BS` (Blood Glucose in mmol/L)
5. `BodyTemp` (Body Temperature in °F)
6. `HeartRate` (bpm)

---

## How to Run

### 1. Prerequisites & Environment Setup

Open PowerShell or terminal in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Launch the Web Application

```powershell
python app.py
```

Open your browser at:
👉 **`http://127.0.0.1:5000`**

---

## Features
- **Sleek Clinical UI**: Built with responsive HTML5, modern CSS3 (Plus Jakarta Sans & Inter), and interactive JavaScript.
- **Biomarker Controls**: Dual synchronized sliders and numeric input fields for all 6 maternal features.
- **1-Click Test Presets**: Quick demonstration profiles for **Low Risk**, **Mid Risk**, and **High Risk**.
- **Model Performance & Confusion Matrix**: Click the header button to inspect the Confusion Matrix plot, Accuracy (86.21%), Macro Precision (87.22%), Recall (87.12%), F1-Score (86.98%), and 3-model comparative benchmarks.
