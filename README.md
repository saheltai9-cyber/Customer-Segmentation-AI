# Customer Segmentation AI

This repository contains a complete Python machine learning application for customer segmentation using KMeans clustering.

## Project Structure
```
PROJECT NUMBER 2/
├─ app.py                # Streamlit dashboard
├─ train.py              # Model training script
├─ predict.py            # Prediction helper
├─ requirements.txt      # Python dependencies
├─ run.bat               # Windows shortcut to start Streamlit app
├─ package_project.py    # Packages the project into a zip
├─ .gitignore
├─ src/
│   ├─ data_loader.py
│   ├─ preprocessing.py
│   ├─ clustering.py
│   └─ visualization.py
├─ data/
│   ├─ raw/               # Raw CSV download
│   └─ processed/         # Processed data (optional)
├─ models/                # Saved scaler + KMeans pipeline
├─ reports/               # Evaluation reports & figures
├─ tests/                 # Unit tests
└─ dist/                  # Distribution zip
```

## How to Run
1. **Install dependencies** (requires internet):
   ```
   pip install -r requirements.txt
   ```
2. **Download the dataset and train the model**:
   ```
   python train.py
   ```
   This will:
   - Validate and clean the data
   - Run EDA and produce plots in `reports/`
   - Train a KMeans model (pipeline saved to `models/pipeline.joblib`)
   - Output evaluation metrics.
3. **Start the Streamlit dashboard**:
   ```
   ./run.bat
   ```
   The dashboard will be available at `http://localhost:8501`.
4. **Package the project** (optional):
   ```
   python package_project.py
   ```
   The zip file will be created at `dist/Customer-Segmentation-AI.zip`.

## Dataset
The dataset is downloaded from:
https://raw.githubusercontent.com/tanishq21/Mall-Customers/main/Mall_Customers.csv

## License
The dataset is provided under its original license. The code in this project is MIT licensed.

