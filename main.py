from fastapi import FastAPI, UploadFile, File, HTTPException
import io

import joblib
import ttsnet
import numpy as np
import pandas as pd

from tcn import TCN
from sklearn.cluster import DBSCAN
from tensorflow.keras.saving import load_model

model_path = "models/TTSNet/v4_Model.h5"
scaler = joblib.load("models/TTSNet/v4_Scaler.pkl")

model = load_model(
                model_path,
                custom_objects={
                    'PositionalEncoding': ttsnet.PositionalEncoding,
                    'TCN': TCN,
                    'TransformerEncoder': ttsnet.TransformerEncoder
                },
                compile=False
            )

CLASS_NAMES = ["Non-Fall", "Fall"]
MIN_POINTS = 10
EPS_VALUE = 0.5
TIMESTEPS = 30

### Model
def extract_features(df):
    return df[["x", "y", "z", "doppler", "SNR"]].to_numpy()

def run_inference(df: pd.DataFrame):
    grouped = df.groupby("timestamp")

    # DBSCAN per frame
    dbscan_results = {}
    for frame_id, frame_points in grouped:
        if len(frame_points) < MIN_POINTS:
            continue
        xyz = frame_points[["x", "y", "z"]].to_numpy()
        db  = DBSCAN(eps=EPS_VALUE, min_samples=10).fit(xyz)
        dbscan_results[frame_id] = db.labels_

    if not dbscan_results:
        raise HTTPException(status_code=422, detail="No usable frames found in uploaded file")

    # Extract centroid per frame
    processed_frames = []
    for frame_id, frame_points in df.groupby("timestamp"):
        if frame_id not in dbscan_results:
            continue

        labels_db = dbscan_results[frame_id]
        clusters  = [c for c in set(labels_db) if c != -1]
        if not clusters:
            continue

        largest       = max(clusters, key=lambda c: np.sum(labels_db == c))
        cluster_points = frame_points.iloc[labels_db == largest]
        centroid      = np.mean(extract_features(cluster_points), axis=0)
        processed_frames.append(centroid)

    # Sliding window
    if len(processed_frames) < TIMESTEPS:
        raise HTTPException(status_code=422, detail=f"Not enough frames — need at least {TIMESTEPS}, got {len(processed_frames)}")

    windows = np.array([
        processed_frames[i:i + TIMESTEPS]
        for i in range(len(processed_frames) - TIMESTEPS + 1)
    ])

    # Scale and predict
    windows_scaled = scaler.transform(
        windows.reshape(-1, windows.shape[-1])
    ).reshape(windows.shape)

    predictions  = model.predict(windows_scaled)
    window_classes = np.argmax(predictions, axis=1)
    final_class  = int(np.bincount(window_classes).argmax())

    return {
        "predicted_class": CLASS_NAMES[final_class],
        "fall_detected": final_class == 1,
        "confidence": float(np.mean(predictions[:, final_class]))
    }

### API Backend
app = FastAPI()

@app.get('/')
def root():
    return {"status":"Fall detection API is running..."}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    return run_inference(df)