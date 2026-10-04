"""
dataset_loader.py
Generates synthetic but realistic multivariate air quality time-series data
and loads it into matrix form X in R^(T x d) for mathematical analysis.

Features produced:
  CO          Carbon Monoxide (ppm)
  NO2         Nitrogen Dioxide (ppb)
  Temp        Ambient Temperature (Celsius)
  Humidity    Relative Humidity (percent)
  PM10        Coarse Particulate Matter (ug/m^3)
  PM2.5       Fine Particulate Matter (ug/m^3)  <- prediction target

Z-score normalization is applied:
  x_norm = (x - mu) / sigma
"""

import os
import numpy as np
import pandas as pd


def generate_air_quality_dataset(filepath="data/air_quality.csv", num_hours=720):
    """
    Produces T = 720 hourly observations with realistic physical coupling:
      - CO, NO2 peak during morning and evening rush hours.
      - Temperature follows a 24-hour sinusoidal diurnal cycle.
      - Humidity is inversely correlated with temperature.
      - PM10 is driven by traffic and low humidity conditions.
      - PM2.5 is a downstream linear combination of PM10, CO, and NO2 plus noise.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    rng = np.random.default_rng(42)
    t = np.arange(num_hours)

    # Diurnal and weekly cycles
    diurnal = np.sin(2 * np.pi * t / 24)
    weekly = 0.4 * np.sin(2 * np.pi * t / 168)

    # Traffic rush hour signal: peaks at 8am and 6pm
    rush = (
        np.exp(-((t % 24 - 8) ** 2) / 6.0) +
        1.2 * np.exp(-((t % 24 - 18) ** 2) / 8.0)
    )

    # Individual feature generation
    temp = 20.0 + 8.0 * np.sin(2 * np.pi * (t - 9) / 24) + rng.normal(0, 1.2, num_hours)
    humidity = np.clip(60.0 - 20.0 * diurnal + rng.normal(0, 3.5, num_hours), 20.0, 95.0)
    co = 0.8 + 1.6 * rush + 0.3 * diurnal + rng.gamma(2.0, 0.15, num_hours)
    no2 = np.clip(25.0 + 35.0 * rush - 0.4 * temp + rng.normal(0, 3.0, num_hours), 5.0, 120.0)
    pm10 = 35.0 + 25.0 * rush + 5.0 * (100 - humidity) / 40.0 + rng.gamma(3.0, 3.0, num_hours)
    pm25 = np.clip(0.55 * pm10 + 8.0 * co + 0.2 * no2 + 4.0 * weekly + rng.normal(0, 2.5, num_hours), 5.0, 180.0)

    df = pd.DataFrame({
        "Hour": t,
        "CO": np.round(co, 3),
        "NO2": np.round(no2, 2),
        "Temp": np.round(temp, 2),
        "Humidity": np.round(humidity, 2),
        "PM10": np.round(pm10, 2),
        "PM2.5": np.round(pm25, 2)
    })
    df.to_csv(filepath, index=False)
    return df


class MatrixDataLoader:
    """
    Loads the air quality CSV and constructs:
      X_raw in R^(T x d): raw unnormalized feature matrix.
      X in R^(T x d): Z-score normalized feature matrix used for computation.
      y_raw in R^T: raw target PM2.5 vector.
      y in R^T: normalized target vector.
    """

    def __init__(self, filepath="data/air_quality.csv", target_col="PM2.5"):
        if not os.path.exists(filepath):
            generate_air_quality_dataset(filepath)

        self.df = pd.read_csv(filepath)
        self.target_col = target_col
        self.feature_cols = [c for c in self.df.columns if c not in ["Hour", target_col]]

        # Raw matrices
        self.X_raw = self.df[self.feature_cols].values.astype(np.float32)
        self.y_raw = self.df[self.target_col].values.astype(np.float32)

        # Z-score normalization: x_norm = (x - mu) / sigma
        self.mu_X = np.mean(self.X_raw, axis=0)
        self.sigma_X = np.std(self.X_raw, axis=0) + 1e-8
        self.mu_y = float(np.mean(self.y_raw))
        self.sigma_y = float(np.std(self.y_raw) + 1e-8)

        self.X = (self.X_raw - self.mu_X) / self.sigma_X
        self.y = (self.y_raw - self.mu_y) / self.sigma_y

    def get_matrix_info(self):
        """Returns basic shape and normalization stats."""
        return {
            "num_timesteps_T": int(self.X.shape[0]),
            "num_features_d": int(self.X.shape[1]),
            "feature_names": self.feature_cols,
            "target_name": self.target_col,
            "mu_y": self.mu_y,
            "sigma_y": self.sigma_y
        }
