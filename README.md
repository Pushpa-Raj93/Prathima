# Medicine Shortage Predictor

A small Flask app using SQLite and TensorFlow/Keras (LSTM) to predict medicine shortage risk.

Setup & Run (VS Code / Windows):

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

1. Run the app:

```powershell
python app.py
```

1. Open your browser at `http://127.0.0.1:5000/`.

Admin credentials (hardcoded in `app.py`):

- username: `admin`
- password: `password123`

Usage:

- Add medicines (set minimum threshold).
- Add daily stock records (opening, used, received).
- Open a medicine page and click "Train Model" (requires >=20 records).
- After training, the app predicts next 7 days; if any predicted value is below threshold it shows a shortage alert.

Files:

- `app.py` - Flask app and model training/prediction logic
- `templates/` - Jinja2 templates
- `model/` - saved Keras models (created at runtime)
- `database.db` - created at runtime
