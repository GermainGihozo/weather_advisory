import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your filtered Rwanda dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

# Features & Target
X = df[['temperature_celsius', 'wind_kph', 'pressure_mb', 'humidity', 'cloud']]
y = df['precip_mm']  # Predict rainfall

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save properly
with open("rwanda_rainfall_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("MODEL SAVED SUCCESSFULLY!")