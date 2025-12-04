# app.py
import os
import csv
import datetime
import joblib
import requests


from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required, logout_user,
    current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from fpdf import FPDF
from dotenv import load_dotenv
from twilio.rest import Client

from log_utils import append_log, read_recent

load_dotenv()

# Config
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///weather_advisory.db")
MODEL_PATH = "rwanda_rainfall_model.pkl"
LOG_FILE = "prediction_log.csv"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TW_SID = os.getenv("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TW_FROM = os.getenv("TWILIO_FROM")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# App init
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(30), nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="farmer")

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# Load ML model (joblib)
model = None
if os.path.isfile(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Model load error:", e)
else:
    print("Model file not found:", MODEL_PATH)

# Twilio client (optional)
twilio_client = None
if TW_SID and TW_TOKEN:
    try:
        twilio_client = Client(TW_SID, TW_TOKEN)
    except Exception as e:
        print("Twilio init error:", e)
        twilio_client = None

# Helper: chart data (reads prediction_log csv produced by log_utils)
def _get_chart_data():
    rows = read_recent(20)
    if not rows:
        return [], []
    dates = [r.get("date") for r in rows]
    vals = [r.get("predicted_rainfall") for r in rows]
    return dates, vals

# Advice generator (same rules as before)
def generate_farmer_advice(rain_mm):
    rain_mm = round(float(rain_mm), 2)
    if rain_mm < 1:
        return {"title": f"Drought Alert ({rain_mm} mm)", "general":"Very dry...", "maize":"Irrigate", "beans":"Delay sowing"}
    if rain_mm < 5:
        return {"title": f"Light Rain ({rain_mm} mm)", "general":"Light...", "maize":"Top-dress", "beans":"Monitor"}
    if rain_mm < 20:
        return {"title": f"Optimal ({rain_mm} mm)", "general":"Good for planting", "maize":"Plant", "beans":"Plant"}
    if rain_mm < 50:
        return {"title": f"Heavy ({rain_mm} mm)", "general":"Manage drainage", "maize":"Risk of rot", "beans":"Avoid field access"}
    return {"title": f"Severe Flooding ({rain_mm} mm)", "general":"Cease activity", "maize":"Loss likely", "beans":"Loss likely"}

# ----------------- Routes -----------------
# Home (protected)
@app.route("/")
@login_required
def home():
    chart_dates, chart_values = _get_chart_data()
    return render_template("index.html",
                           google_maps_api_key=GOOGLE_MAPS_API_KEY,
                           chart_dates=chart_dates,
                           chart_values=chart_values,
                           prediction_value=None,
                           error_message=None,
                           farmer_advice=None)

# Manual predict (protected)
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        flash("Model not loaded. Train or place rwanda_rainfall_model.pkl in project root.", "danger")
        return redirect(url_for("home"))
    try:
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        wind = float(request.form["wind"])
        pressure = float(request.form["pressure"])
        cloud = float(request.form["cloud"])
        features = np.array([[temp, wind, pressure, humidity, cloud]])
        pred = float(model.predict(features)[0])
        advice = generate_farmer_advice(pred)

        # Log
        row = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temp,
            "wind": wind,
            "pressure": pressure,
            "humidity": humidity,
            "cloud": cloud,
            "predicted_rainfall": pred
        }
        append_log(row)

        # Optional Twilio alert
        if twilio_client and ADMIN_PHONE and pred >= 50:
            try:
                twilio_client.messages.create(body=f"ALERT: predicted rainfall {pred} mm", from_=TW_FROM, to=ADMIN_PHONE)
            except Exception as e:
                print("Twilio send failed:", e)

        chart_dates, chart_values = _get_chart_data()
        return render_template("index.html",
                               prediction_value=pred,
                               farmer_advice=advice,
                               chart_dates=chart_dates,
                               chart_values=chart_values,
                               google_maps_api_key=GOOGLE_MAPS_API_KEY,
                               error_message=None)
    except Exception as e:
        flash(f"Error processing input: {e}", "danger")
        return redirect(url_for("home"))

# Auto predict from OpenWeather (protected)
@app.route("/predict_auto", methods=["POST"])
@login_required
def predict_auto():
    if model is None:
        flash("Model not loaded.", "danger")
        return redirect(url_for("home"))

    city = request.form.get("city")
    if not city:
        flash("City required", "warning")
        return redirect(url_for("home"))
    if not OPENWEATHER_API_KEY:
        flash("OpenWeather API key missing.", "danger")
        return redirect(url_for("home"))

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        flash(f"Weather API error: {r.text}", "danger")
        return redirect(url_for("home"))
    data = r.json()
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind = data["wind"].get("speed", 0) * 3.6
    cloud = data.get("clouds", {}).get("all", 0)

    features = np.array([[temp, wind, pressure, humidity, cloud]])
    pred = float(model.predict(features)[0])
    advice = generate_farmer_advice(pred)

    append_log({
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temp, "wind": wind, "pressure": pressure,
        "humidity": humidity, "cloud": cloud, "predicted_rainfall": pred
    })

    chart_dates, chart_values = _get_chart_data()
    return render_template("index.html",
                           prediction_value=pred, farmer_advice=advice,
                           chart_dates=chart_dates, chart_values=chart_values,
                           google_maps_api_key=GOOGLE_MAPS_API_KEY, error_message=None)

# Chart data endpoint
@app.route("/chart_data")
@login_required
def chart_data():
    return jsonify(read_recent(50))

# PDF download (protected)
@app.route("/download_report")
@login_required
def download_report():
    latest = read_recent(1)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "AI Weather Advisory Report", ln=True)
    if latest:
        rec = latest[0]
        for k, v in rec.items():
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
        advice = generate_farmer_advice(rec["predicted_rainfall"])
        pdf.ln(4)
        pdf.cell(0, 8, "Advice:", ln=True)
        pdf.multi_cell(0, 8, f"{advice['title']}\nGeneral: {advice['general']}\nMaize: {advice['maize']}\nBeans: {advice['beans']}")
    out = "weather_report.pdf"
    pdf.output(out)
    return send_file(out, as_attachment=True, download_name="weather_report.pdf")

# PWA endpoints
@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name":"AI Weather Advisory",
        "short_name":"WeatherAdvisory",
        "start_url":"/",
        "display":"standalone",
        "background_color":"#ecf8f0",
        "theme_color":"#00994d"
    })

@app.route("/service-worker.js")
def service_worker():
    js = """
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => self.clients.claim());
self.addEventListener('fetch', e => {});
"""
    return app.response_class(js, mimetype='application/javascript')

# ----------------- Auth routes -----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        phone = request.form.get("phone")
        password = request.form.get("password")

        if not username or not password:
            flash("Username and password are required.", "warning")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        user = User(username=username, phone=phone)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Registered and logged in.", "success")
        return redirect(url_for("home"))
    return render_template("register.html")

# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         username = request.form.get("username")
#         password = request.form.get("password")

#         if not username or not password:
#             flash("Please enter username and password", "warning")
#             return redirect(url_for("login"))

#         user = User.query.filter_by(username=username).first()
#         if not user or not user.check_password(password):
#             flash("Invalid username or password", "danger")
#             return redirect(url_for("login"))

#         login_user(user)
#         flash("Logged in successfully.", "success")
#         return redirect(url_for("home"))

#     return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

        login_user(user)
        flash("Login successful", "success")
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

# Admin dashboard (role-protected)
@app.route("/dashboard")
@login_required
def dashboard():
    if getattr(current_user, "role", None) != "admin":
        flash("Admin only", "warning")
        return redirect(url_for("home"))
    df = pd.read_csv(LOG_FILE) if os.path.isfile(LOG_FILE) else pd.DataFrame()
    return render_template("dashboard.html", table=df.to_html(classes="table", index=False))

# Bootstrap DB + admin on first run
@app.before_request
def bootstrap():
    if not hasattr(app, "has_setup"):
        db.create_all()
        if not User.query.filter_by(role="admin").first():
            admin = User(username="admin", role="admin")
            admin.set_password(os.getenv("ADMIN_PASSWORD", "admin123"))
            db.session.add(admin)
            db.session.commit()
            print("Admin created: admin")
        app.has_setup = True

if __name__ == "__main__":
    app.run(debug=True)
