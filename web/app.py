import os
import pandas as pd
import numpy as np
import datetime
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

app = Flask(__name__)

# --- CONFIGURATION ---
# Set this to the folder where your CSV files are located
DATA_DIR = "data" 

PLANTS = {
    "Gibe I": "Gibe1.csv",
    "Gibe III": "Gibe3.csv",
    "Koka": "Koka Plant.csv",
    "Tana Beles": "Tana_Beles.csv",
    "Tekeze": "Tekeze.csv",
    "Fincha": "fincha.csv",
}

# --- HELPERS (From your Notebook) ---
def clean_numeric(x):
    if isinstance(x, str): 
        x = x.replace(',', '').replace(' ', '')
    return pd.to_numeric(x, errors='coerce')

def load_and_clean(plant_name):
    filepath = os.path.join(DATA_DIR, PLANTS[plant_name])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    
    if 'Date_GC' in df.columns:
        df['Date'] = pd.to_datetime(df['Date_GC'])
        df.set_index('Date', inplace=True)
    
    cols = df.select_dtypes(include=[np.number, 'object']).columns
    for col in cols:
        df[col] = df[col].apply(clean_numeric)
        # Fill missing values using median (notebook logic)
        if not df[col].isna().all():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    return df

def build_model(model_type, input_dim):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(64, input_shape=(1, input_dim)))
    elif model_type == 'GRU':
        model.add(GRU(64, input_shape=(1, input_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', plants=list(PLANTS.keys()))

# Change this to '/predict' to match your JavaScript fetch() call
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        plant = data.get('plant')
        target = data.get('target', 'Energy') 
        model_type = data.get('model', 'XGBoost')
        years = 1 # Default to 1 year for web performance

        df = load_and_clean(plant)
        
        # Feature Engineering: Seasonality (Notebook logic)
        df['month'] = df.index.month
        df['year'] = df.index.year
        features = ['month', 'year']
        
        X = df[features].values
        y = df[target].values.reshape(-1, 1)
        
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Training
        if model_type == 'XGBoost':
            model = xgb.XGBRegressor(n_estimators=100)
            model.fit(X_scaled, y_scaled)
        else:
            model = build_model(model_type, X_scaled.shape[1])
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            # Reduced epochs for web speed
            model.fit(X_reshaped, y_scaled, epochs=5, batch_size=32, verbose=0)

        # Create Future Dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=years*365, freq='D')
        future_X = pd.DataFrame({'month': future_dates.month, 'year': future_dates.year})
        future_X_scaled = scaler_x.transform(future_X.values)

        if model_type == 'XGBoost':
            preds = model.predict(future_X_scaled)
        else:
            future_X_reshaped = future_X_scaled.reshape((future_X_scaled.shape[0], 1, future_X_scaled.shape[1]))
            preds = model.predict(future_X_reshaped).flatten()

        final_preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

        return jsonify({
            'labels': future_dates.strftime('%Y-%m-%d').tolist(),
            'values': final_preds.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)