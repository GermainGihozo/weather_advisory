# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

CSV_PATH = "GlobalWeatherRepository.csv"   # adjust if needed
MODEL_OUT = "rwanda_rainfall_model.pkl"

# 1. Load
df = pd.read_csv(CSV_PATH)

# 2. Keep only required columns (drop rows with missing)
cols = ['temperature_celsius','wind_kph','pressure_mb','humidity','cloud','precip_mm']
df = df[cols].dropna()

# 3. Features & target
X = df[['temperature_celsius','wind_kph','pressure_mb','humidity','cloud']]
y = df['precip_mm']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Eval
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
# rmse = mean_squared_error(y_test, preds, squared=False)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# 7. Save
joblib.dump(model, MODEL_OUT)
print("Saved model to", MODEL_OUT)
