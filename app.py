from flask import Flask, render_template, request
import pickle
import numpy as np
import datetime
import csv
import os

app = Flask(__name__)

# Load trained model
try:
    with open("rwanda_rainfall_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERROR: Model file 'rwanda_rainfall_model.pkl' not found.")
    model = None

LOG_FILE = "prediction_log.csv"

# =======================================================
# ADVICE LOGIC FUNCTION
# =======================================================

def generate_farmer_advice(rainfall_mm):
    # Round the rainfall for clean comparison and display
    rainfall_mm = round(rainfall_mm, 2)
    
    advice = {
        "title": "Weather Advisory",
        "general": "No specific advice available.",
        "maize": "No specific advice available.",
        "beans": "No specific advice available."
    }

    # ----------------------------------------------------
    # 1. Severe Drought (0 to < 1 mm)
    # ----------------------------------------------------
    if rainfall_mm < 1:
        advice["title"] = f"Drought Alert (Predicted: {rainfall_mm} mm) ☀️"
        advice["general"] = (
            "**Drought Alert.** Extremely dry conditions. Prioritize immediate **water conservation** "
            "using heavy mulch. Postpone planting until reliable rain is forecasted."
        )
        advice["maize"] = (
            "If in the growth stage, provide **spot irrigation** to prevent tassel and cob failure. "
            "Avoid excessive fertilizer application."
        )
        advice["beans"] = (
            "Do not sow new beans. If planted, use shading and **deep watering only once per week** (if feasible), "
            "as frequent light watering encourages shallow roots."
        )

    # ----------------------------------------------------
    # 2. Light Rain (1 to < 5 mm)
    # ----------------------------------------------------
    elif rainfall_mm < 5:
        advice["title"] = f"Light Rain Forecast (Predicted: {rainfall_mm} mm) 🌦️"
        advice["general"] = (
            "**Light Rain Forecast.** Good for **soil preparation, weeding, and applying fertilizer**, "
            "but new, full-scale planting is still risky without follow-up moisture."
        )
        advice["maize"] = (
            "The rain may be insufficient for deep root hydration. **Supplement with irrigation** if possible. "
            "Excellent time for **top-dressing** fertilizer."
        )
        advice["beans"] = (
            "Plant only if follow-up irrigation or rain is assured. Light rain and high humidity increase "
            "the risk of **fungal diseases**; monitor closely."
        )
        
    # ----------------------------------------------------
    # 3. Moderate Rain (5 to < 20 mm)
    # ----------------------------------------------------
    elif rainfall_mm < 20:
        advice["title"] = f"Optimal Rainfall (Predicted: {rainfall_mm} mm) ✅"
        advice["general"] = (
            "**Optimal Conditions.** Excellent conditions for germination and sustained growth. "
            "Ensure drainage is adequate but conserve soil moisture."
        )
        advice["maize"] = (
            "**Planting is highly recommended.** Apply the main NPK basal dressing now to support rapid growth."
        )
        advice["beans"] = (
            "**Ideal time for sowing.** Ensure proper depth (approx. 5 cm). Perfect conditions for root nodulation (nitrogen fixing)."
        )

    # ----------------------------------------------------
    # 4. Heavy Rain (20 to < 50 mm)
    # ----------------------------------------------------
    elif rainfall_mm < 50:
        advice["title"] = f"Heavy Rain Warning (Predicted: {rainfall_mm} mm) ⚠️"
        advice["general"] = (
            "**Heavy Rain Warning.** Focus intensely on **drainage management**. Clear ditches, and "
            "check terraces to prevent soil erosion and waterlogging."
        )
        advice["maize"] = (
            "Maize is vulnerable to waterlogging. If fields flood, the yield will drop. **Create temporary drainage channels** "
            "between rows. Inspect for stalk rot."
        )
        advice["beans"] = (
            "Highly susceptible to **root rot** in saturated soil. **Avoid field access** to prevent soil compaction. "
            "Consider applying preventative **fungicide** if this pattern persists."
        )
        
    # ----------------------------------------------------
    # 5. Severe Flooding (>= 50 mm)
    # ----------------------------------------------------
    else: # rainfall_mm >= 50
        advice["title"] = f"Severe Flood Risk (Predicted: {rainfall_mm} mm) 🚨"
        advice["general"] = (
            "**Severe Flood Risk.** Take protective action. Secure harvested crops and livestock. All field work must cease. "
            "Be alert for flash floods and soil collapse."
        )
        advice["maize"] = (
            "If planted, assume significant crop loss in low areas. Focus on **re-establishing proper drainage** immediately "
            "after the storm passes."
        )
        advice["beans"] = (
            "If fields are submerged for more than 48 hours, **crop loss is likely.** Be ready for potential **replanting** "
            "once the soil has dried out completely."
        )

    return advice

# =======================================================

# ------------------------
# HOME PAGE (Fixed for variable consistency)
# ------------------------
@app.route("/")
def home():
    # Passes required variables to avoid 'Undefined' errors on initial load
    return render_template("index.html", prediction_value=None, error_message=None, farmer_advice=None)


# ------------------------
# PREDICTION ENDPOINT (Fixed for POST method and advice passing)
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction_value=None, error_message="⚠ Model is not loaded. Check server logs.", farmer_advice=None)
        
    try:
        # Features extraction (must match input names in index.html)
        temp = float(request.form["temperature"])
        wind = float(request.form["wind"])
        pressure = float(request.form["pressure"]) 
        humidity = float(request.form["humidity"])
        cloud = float(request.form["cloud"])

        features = np.array([[temp, wind, pressure, humidity, cloud]])
        prediction = model.predict(features)[0]
        
        # Call the advice function
        farmer_advice = generate_farmer_advice(prediction)
        
        # --- Logging Logic ---
        row = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temp,
            "wind": wind,
            "pressure": pressure,
            "humidity": humidity,
            "cloud": cloud,
            "predicted_rainfall": prediction
        }
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        # --- End Logging Logic ---
        
        # Pass prediction value and structured advice
        return render_template("index.html", 
                               prediction_value=prediction, 
                               error_message=None, 
                               farmer_advice=farmer_advice)

    except Exception as e:
        # Pass error message
        return render_template("index.html", 
                               prediction_value=None, 
                               error_message=f"⚠ Error processing input: {str(e)}",
                               farmer_advice=None)


if __name__ == "__main__":
    app.run(debug=True)