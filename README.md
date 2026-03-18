# EcoRoute ML

Machine learning pipeline for real-time driving behaviour classification (smooth / moderate / aggressive) using speed telemetry. Part of the EcoRoute carbon-aware route planner FYP.

## Project Structure

```
ecoroute-ml/
├── ml/
│   ├── features.py       # Feature extraction from speed windows
│   └── detector.py       # RealTimeDetector for streaming inference
├── training/
│   ├── 01_load.py        # Load and clean eVED dataset
│   ├── 02_features.py    # Extract rolling-window features
│   ├── 03_label.py       # K-Means labelling on fuel rate
│   └── 04_train.py       # Train XGBoost classifier
├── tests/
│   └── test_detector.py  # Unit tests for RealTimeDetector
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- eVED dataset (weekly CSV files) — download from [https://bitbucket.org/datarepo/eved-dataset](https://bitbucket.org/datarepo/eved-dataset)

## Setup

**1. Clone the repository**

```bash
git clone https://git.infotech.monash.edu/carbon-route-planner/ecoroute-ml.git
cd ecoroute-ml
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place the eVED weekly CSV files in:

```
training/data/eved/*.csv
```

> The `data/` folder is excluded from git. Do not commit raw data files.

## Training Pipeline

Run each step in order from the repo root:

```bash
# Step 1 — Load and clean raw eVED CSVs
python training/01_load.py

# Step 2 — Extract rolling-window features (10s window, 5s step)
python training/02_features.py

# Step 3 — Label windows via K-Means on fuel rate
python training/03_label.py

# Step 4 — Train XGBoost classifier and save model artifacts
python training/04_train.py
```

Trained model artifacts are saved to `ml/models/`:
- `clf_c1.pkl` — XGBoost classifier
- `scaler_c1.pkl` — fitted StandardScaler

## Running Tests

```bash
python -m pytest tests/ -v
```

Or without pytest:

```bash
python tests/test_detector.py
```

> Tests require trained model artifacts in `ml/models/`. Run the training pipeline first.

## Usage (Inference)

```python
from ml.detector import load_detector

detector = load_detector()

# Push a 10-second speed window (m/s)
alert = detector.push_window([10.0, 12.0, 8.0, 5.0, 0.0, 4.0, 11.0, 15.0, 9.0, 6.0], timestamp=10.0)

print(detector.last_label)   # 0=smooth, 1=moderate, 2=aggressive
print(alert)                 # Alert string or None

# Get trip summary
print(detector.get_trip_summary())
```
