# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from scipy.stats.mstats import winsorize
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib
import warnings
import geodatasets

# Suppress warnings
warnings.filterwarnings("ignore")

# Load world map for visualization
def plot_world_map():
    """Plots a simple world map."""
    try:
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        fig, ax = plt.subplots(figsize=(12, 6))
        world.plot(ax=ax, color='lightgrey', edgecolor='black')
        plt.title("World Map")
        plt.savefig("world_map.png")  # Save instead of showing
        plt.close()  # Close the figure to prevent blocking

    except Exception as e:
        print(f"Error loading world map: {e}")

plot_world_map()

# Load dataset with error handling
DATA_PATH = "data/cleaned_weather_data.csv"

if not os.path.exists(DATA_PATH):
    print(f"Error: Dataset not found at {DATA_PATH}. Please check the file path.")
    exit()

df = pd.read_csv(DATA_PATH)
print(" Dataset Loaded Successfully.")
print(df.head())

# Anomaly Detection using Isolation Forest
def detect_anomalies(df):
    """Detects anomalies using Isolation Forest."""
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(df.select_dtypes(include=['number']))
    anomalies = df[df['anomaly_score'] == -1]
    print(f" Anomalies detected: {len(anomalies)}")
    return df, anomalies

df, anomalies = detect_anomalies(df)

# Feature Importance using Random Forest
def plot_feature_importance(df, target_col):
    """Trains a Random Forest model and plots feature importance."""
    X = df.drop(columns=[target_col, 'anomaly_score'], errors='ignore').select_dtypes(include=['number'])
    y = df[target_col]

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)

    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', title='Top 10 Important Features', color='skyblue')
    plt.xlabel("Feature Importance Score")
    plt.show()

plot_feature_importance(df, 'temperature_celsius')

# Geographical Analysis
def plot_geographical_distribution(df):
    """Plots geographical temperature distribution if latitude and longitude are present."""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        try:
            world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
            fig, ax = plt.subplots(figsize=(12, 6))
            world.plot(ax=ax, color='lightgrey', edgecolor='black')
            scatter = ax.scatter(df['longitude'], df['latitude'], c=df['temperature_celsius'], cmap='coolwarm', alpha=0.5)
            plt.colorbar(scatter, label='Temperature (Celsius)')
            plt.title("Geographical Temperature Distribution")
            plt.show()
        except Exception as e:
            print(f"Error in geographical plotting: {e}")
    else:
        print(" Missing 'latitude' or 'longitude' in dataset!")

plot_geographical_distribution(df)

# Data Preprocessing: Winsorization & Log Transform
def preprocess_data(df, cols):
    """Applies winsorization and log transformation to specified columns."""
    for col in cols:
        if col in df.columns:
            df[col] = winsorize(df[col], limits=[0.05, 0.05])
            df[col] = np.log1p(df[col].replace(0, 1e-6))  # Avoid log(0)
    return df

df = preprocess_data(df, ['precip_mm', 'air_quality_Nitrogen_dioxide', 'visibility_km'])

# Forecasting Models
def forecast_arima(series, steps=30):
    """Fits an ARIMA model and forecasts future values."""
    try:
        model = ARIMA(series, order=(5,1,0)).fit()
        forecast_values = model.forecast(steps=steps)
        return forecast_values
    except Exception as e:
        print(f"ARIMA Forecasting Error: {e}")
        return None

# LSTM Model for Forecasting
def build_lstm_model(input_shape):
    """Builds and compiles an LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Model Training
def train_models(df, target_col):
    """Trains Random Forest and XGBoost models, evaluates performance, and saves the best model."""
    X = df.drop(columns=[target_col, 'anomaly_score'], errors='ignore').select_dtypes(include=['number'])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    print(f" Random Forest MAE: {rf_mae:.4f}")

    # Train XGBoost Model
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    print(f" XGBoost MAE: {xgb_mae:.4f}")

    # Ensemble Model (Average Predictions)
    y_pred_ensemble = (y_pred_rf + y_pred_xgb) / 2
    ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
    print(f" Ensemble Model MAE: {ensemble_mae:.4f}")

    # Save the best model
    best_model = xgb_model if xgb_mae < rf_mae else rf
    joblib.dump(best_model, "weather_model.pkl")
    print(" Best model saved as 'weather_model.pkl'.")

train_models(df, 'temperature_celsius')
