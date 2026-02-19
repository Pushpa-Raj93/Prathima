import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
import numpy as np
import json

os.makedirs('model', exist_ok=True)

APP_SECRET = 'change-me-please'
ADMIN_USER = 'admin'
ADMIN_PASS = 'password123'

DB_PATH = 'database.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS medicines (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        min_threshold INTEGER DEFAULT 10
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        id INTEGER PRIMARY KEY,
        medicine_id INTEGER,
        date TEXT,
        opening_stock INTEGER,
        used_stock INTEGER,
        received_stock INTEGER,
        FOREIGN KEY(medicine_id) REFERENCES medicines(id)
    )
    ''')
    db.commit()

app = Flask(__name__)
app.secret_key = APP_SECRET

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def compute_closing(opening, used, received):
    return opening - used + received

@app.route('/')
def index():
    medicines = query_db('SELECT * FROM medicines')
    rows = []
    for m in medicines:
        r = query_db('SELECT * FROM stocks WHERE medicine_id=? ORDER BY date DESC LIMIT 1', (m['id'],), one=True)
        latest = None
        if r:
            latest = compute_closing(r['opening_stock'], r['used_stock'], r['received_stock'])
        rows.append({'id': m['id'], 'name': m['name'], 'min_threshold': m['min_threshold'], 'latest': latest})
    return render_template('index.html', medicines=rows)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        if u == ADMIN_USER and p == ADMIN_PASS:
            session['user'] = u
            flash('Logged in')
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out')
    return redirect(url_for('login'))

@app.route('/medicine/add', methods=['GET', 'POST'])
def add_medicine():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form.get('name')
        min_th = int(request.form.get('min_threshold') or 10)
        try:
            db = get_db()
            db.execute('INSERT INTO medicines (name, min_threshold) VALUES (?,?)', (name, min_th))
            db.commit()
            flash('Medicine added')
            return redirect(url_for('index'))
        except Exception as e:
            flash('Error: ' + str(e))
    return render_template('add_medicine.html')

@app.route('/record/add', methods=['GET', 'POST'])
def add_record():
    if 'user' not in session:
        return redirect(url_for('login'))
    meds = query_db('SELECT * FROM medicines')
    if request.method == 'POST':
        medicine_id = int(request.form.get('medicine_id'))
        date = request.form.get('date')
        opening = int(request.form.get('opening_stock') or 0)
        used = int(request.form.get('used_stock') or 0)
        received = int(request.form.get('received_stock') or 0)
        db = get_db()
        db.execute('INSERT INTO stocks (medicine_id, date, opening_stock, used_stock, received_stock) VALUES (?,?,?,?,?)',
                   (medicine_id, date, opening, used, received))
        db.commit()
        flash('Record added')
        return redirect(url_for('index'))
    return render_template('add_record.html', medicines=meds)

def load_timeseries(medicine_id):
    rows = query_db('SELECT date, opening_stock, used_stock, received_stock FROM stocks WHERE medicine_id=? ORDER BY date', (medicine_id,))
    dates = [r['date'] for r in rows]
    values = [compute_closing(r['opening_stock'], r['used_stock'], r['received_stock']) for r in rows]
    return dates, values

def scale_series(series):
    arr = np.array(series, dtype=float)
    mn = arr.min() if len(arr)>0 else 0.0
    mx = arr.max() if len(arr)>0 else 1.0
    if mx - mn == 0:
        return arr, mn, mx
    scaled = (arr - mn) / (mx - mn)
    return scaled, mn, mx

def inverse_scale(scaled, mn, mx):
    return scaled * (mx - mn) + mn

def create_sequences(data, seq_len=14):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

@app.route('/medicine/<int:medicine_id>')
def view_medicine(medicine_id):
    med = query_db('SELECT * FROM medicines WHERE id=?', (medicine_id,), one=True)
    if not med:
        flash('Medicine not found')
        return redirect(url_for('index'))
    dates, values = load_timeseries(medicine_id)
    preds = []
    shortage = False
    model_path = os.path.join('model', f'model_{med["name"]}.h5')
    if os.path.exists(model_path) and len(values) > 0:
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            seq_len = 14
            scaled, mn, mx = scale_series(values)
            if len(scaled) >= seq_len:
                last_seq = scaled[-seq_len:]
                for _ in range(7):
                    pred = model.predict(last_seq.reshape(1, seq_len, 1))
                    last_seq = np.append(last_seq[1:], pred[0,0])
                    preds.append(float(inverse_scale(pred[0,0], mn, mx)))
                if any(p < med['min_threshold'] for p in preds):
                    shortage = True
        except Exception as e:
            flash('Prediction error: ' + str(e))
    return render_template('view_medicine.html', med=med, dates=dates, values=values, preds=preds, shortage=shortage)

@app.route('/train/<int:medicine_id>')
def train_medicine(medicine_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    med = query_db('SELECT * FROM medicines WHERE id=?', (medicine_id,), one=True)
    if not med:
        flash('Medicine not found')
        return redirect(url_for('index'))
    dates, values = load_timeseries(medicine_id)
    if len(values) < 20:
        flash('Not enough data to train (need at least 20 days).')
        return redirect(url_for('view_medicine', medicine_id=medicine_id))
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping

        seq_len = 14
        scaled, mn, mx = scale_series(values)
        X, y = create_sequences(scaled, seq_len)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential()
        model.add(LSTM(64, input_shape=(seq_len,1), return_sequences=False))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=30, batch_size=8, callbacks=[es], verbose=0)
        model_path = os.path.join('model', f'model_{med["name"]}.h5')
        model.save(model_path)
        flash('Model trained and saved')
    except Exception as e:
        flash('Training error: ' + str(e))
    return redirect(url_for('view_medicine', medicine_id=medicine_id))

@app.route('/api/predict/<int:medicine_id>')
def api_predict(medicine_id):
    med = query_db('SELECT * FROM medicines WHERE id=?', (medicine_id,), one=True)
    if not med:
        return jsonify({'error':'medicine not found'})
    dates, values = load_timeseries(medicine_id)
    preds = []
    model_path = os.path.join('model', f'model_{med["name"]}.h5')
    if os.path.exists(model_path) and len(values) > 0:
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            seq_len = 14
            scaled, mn, mx = scale_series(values)
            if len(scaled) >= seq_len:
                last_seq = scaled[-seq_len:]
                for _ in range(7):
                    pred = model.predict(last_seq.reshape(1, seq_len, 1))
                    last_seq = np.append(last_seq[1:], pred[0,0])
                    preds.append(float(inverse_scale(pred[0,0], mn, mx)))
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'dates': dates, 'values': values, 'preds': preds})

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True)
