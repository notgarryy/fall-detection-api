# Fall Detection Inference API

A production-ready REST API for real-time human fall detection using **FMCW radar point cloud data** and **TTSNet**, a spatiotemporal deep learning architecture.

> 🏆 **Related Research**: This API is based on work published in **IEEE Access (Q1)**, 2026 and **Advance Sustainable Science, Engineering and Technology (Q3, SINTA 1)**, 2026 (under revision).
> [View IEEE Access Paper](https://doi.org/10.1109/ACCESS.2026.3676850)

---

## Overview

This API accepts FMCW radar point cloud data as a CSV file and returns a real-time fall detection result with a confidence score.

**Pipeline:**
```
CSV Upload (radar point cloud)
        ↓
DBSCAN Clustering (noise filtering)
        ↓
Sliding Window (30 frames)
        ↓
TTSNet Inference
        ↓
JSON Response (fall detected + confidence)
```

**Output Classes:**
| Class | Description |
|---|---|
| Non-Fall | Standing, sitting, or walking |
| Fall | Fall event detected ⚠️ |

---

## Tech Stack

- **FastAPI** — REST API framework
- **TensorFlow / Keras** — TTSNet model inference
- **Scikit-learn** — DBSCAN clustering and data preprocessing
- **Docker** — containerization
- **MLflow** — experiment tracking
- **GitHub Actions** — CI/CD pipeline
- **Railway** — cloud deployment

---

## Live API

> **Base URL**: `https://fall-detection-api-production.up.railway.app`

---

## Repository Structure

```
fall-detection-api/
├── .github/
│   └── workflows/
│       └── fall_det_api.yml    
├── models/
│   ├── TCN/                    ← to be implemented in future updates
│   └── TTSNet 
|       ├── v4_Model.h5
|       └── v4_Scaler.pkl                
├── test/
│   └── datasets/              
├── main.py                     
├── ttsnet.py                   
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10
- Docker (for containerized run)

### Option 1 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/notgarryy/fall-detection-api.git
cd fall-detection-api
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the API**
```bash
uvicorn main:app --reload
```

**5. Open Swagger UI**
```
http://localhost:8000/docs
```

---

### Option 2 — Run with Docker

**1. Build the image**
```bash
docker build -t fall-detection-api .
```

**2. Run the container**
```bash
docker run -p 8000:8000 fall-detection-api
```

**3. Open Swagger UI**
```
http://localhost:8000/docs
```

---

## API Endpoints

### GET /
Health check — confirms the API is running.

**Response:**
```json
{
  "status": "Fall Detection API is running"
}
```

---

### POST /predict
Accepts a CSV file of FMCW radar point cloud data and returns a fall detection result.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: CSV file upload

**Required CSV columns:**
| Column | Description |
|---|---|
| `timestamp` | Frame timestamp |
| `x` | X coordinate (meters) |
| `y` | Y coordinate (meters) |
| `z` | Z coordinate (meters) |
| `doppler` | Doppler velocity |
| `SNR` | Signal-to-noise ratio |

**Response:**
```json
{
  "predicted_class": "Fall",
  "fall_detected": true,
  "confidence": 0.94
}
```

**Error Responses:**
| Status | Reason |
|---|---|
| 400 | File is not a CSV |
| 400 | Required columns missing |
| 422 | Not enough valid frames for inference |

---

## How to Test

### Method 1 — Swagger UI (Recommended for beginners)

1. Open `http://localhost:8000/docs` in your browser
2. Click on `POST /predict`
3. Click **Try it out**
4. Click **Choose File** and upload one of the sample CSV files from `test/datasets/`
5. Click **Execute**
6. View the response in the **Response body** section

---

### Method 2 — curl (Command Line)

**Test with fall sample:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test/datasets/fall_sample.csv;type=text/csv"
```

**Test with non-fall sample:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test/datasets/nonfall_sample.csv;type=text/csv"
```

**Test against live deployed API:**
```bash
curl -X POST https://fall-detection-api-production.up.railway.app/predict \
  -F "file=@test/datasets/fall_sample.csv;type=text/csv"
```

**Expected responses:**

Fall sample:
```json
{
  "predicted_class": "Fall",
  "fall_detected": true,
  "confidence": 0.92
}
```

Non-fall sample:
```json
{
  "predicted_class": "Non-Fall",
  "fall_detected": false,
  "confidence": 0.97
}
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | TTSNet (Spatiotemporal Feature Learning) |
| Input | FMCW radar point cloud sequences |
| Features | x, y, z, doppler, SNR |
| Window size | 30 frames |
| Classification accuracy | 98% |
| Output classes | Non-Fall, Fall |

### Preprocessing Pipeline
1. **DBSCAN clustering** — filters noise, isolates dominant human target
2. **Centroid extraction** — reduces each frame to a single feature vector
3. **Sliding window** — groups 30 consecutive frames per inference
4. **StandardScaler normalization** — scales features to match training distribution

---

## CI/CD Pipeline

This repository uses GitHub Actions for continuous integration. On every push to `main`:

1. Installs dependencies
2. Runs lint checks
3. Builds Docker image
4. Runs container and verifies the API starts correctly

View workflow status in the **Actions** tab of this repository.

---

## Future Work

- [ ] TCN-based model variant 
- [ ] Model versioning with MLflow Model Registry

---

## Related Repositories

- [Real-Time Fall Detection with TTSNet](https://github.com/notgarryy/real-time-fall-detection-system-with-TTSNet) — Q3 SINTA 1
- [Real-Time Fall Detection with TCN](https://github.com/notgarryy/real-time-fall-detection-system-with-TCN) — IEEE Access Q1

---

## Author

**Garry Nelson**
Electrical Engineering, Telkom University — Bandung, Indonesia

[GitHub](https://github.com/notgarryy) | [LinkedIn](https://www.linkedin.com/in/garry-nelson-889834277/)

---