import numpy as np
import joblib

scaler = joblib.load("models/scaler.pkl")

def preprocess(df):

    features = df[["temperature","pressure","flow_rate","vibration"]]
    scaled = scaler.transform(features.values)
    return scaled
