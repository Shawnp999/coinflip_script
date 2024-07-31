import os
import sqlite3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from database import fetch_all_results
import logging
import joblib

MODEL_PATH = 'coinflip_model.h5'
SCALER_PATH = 'scaler.pkl'
BACKUP_PATH = 'coinflips_backup.csv'

def prepare_data(sequence_length=50):
    data = fetch_all_results()
    if len(data) == 0:
        logging.warning("No data available. Unable to prepare data for the model.")
        return None, None, None

    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])  # Remove rows where 'datetime' could not be parsed

    data = data.sort_values('datetime', ascending=False).reset_index(drop=True)

    # Use only the most recent 'sequence_length' entries
    data = data.head(sequence_length)

    # If we have less than sequence_length entries, pad with default values
    if len(data) < sequence_length:
        pad_length = sequence_length - len(data)
        pad_data = pd.DataFrame({
            'result': [0.5] * pad_length,
            'amount': [10000] * pad_length,
            'datetime': [data['datetime'].min()] * pad_length,
            'consecutive_losses_or_wins': [0] * pad_length
        })
        data = pd.concat([pad_data, data]).reset_index(drop=True)

    data = data.iloc[::-1].reset_index(drop=True)

    # Calculate features
    data['total_winrate'] = data['result'].cumsum() / (data.index + 1)
    data['recent_winrate'] = data['result'].rolling(window=10).mean().fillna(0.5)
    data['consecutive_losses'] = data['consecutive_losses_or_wins']
    data.loc[data['result'] == 1, 'consecutive_losses'] = 0

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[['total_winrate', 'recent_winrate', 'consecutive_losses', 'amount', 'consecutive_losses_or_wins']])

    # Verify the number of features in X
    print(f"Number of features: {X.shape[1]}")  # Should print 5 now

    y = data['result'].values

    # Reshape X to be 3D: [samples, time steps, features]
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], 1))

    return X, y, scaler



def create_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y, scaler = prepare_data()
    if X is None or y is None:
        logging.warning("Not enough data to train the model. Please play more games.")
        return None, None
    model = create_model((X.shape[1], X.shape[2]))  # Use the correct input shape
    model.fit(X, y, epochs=100, batch_size=1, verbose=1)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def predict_next_bet(model, scaler):
    X, _, _ = prepare_data()
    if X is None:
        logging.warning("Not enough data to make a prediction. Using default 50% win probability.")
        return 0.5
    prediction = model.predict(X)
    return prediction[0][-1][0]


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logging.info("Training the model...")
        model, scaler = train_model()
        if model is None or scaler is None:
            logging.warning("Failed to train the model. Using default prediction.")
            return None, None
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    return model, scaler

def calculate_bet_amount(prediction, last_bet, consecutive_losses, min_bet=10000, multiplier=2.2):
    if prediction > 0.5:
        confidence = (prediction - 0.5) * 2  # Scale confidence to 0-1
        bet = min_bet * (1 + confidence)
        return bet
    else:
        return last_bet * multiplier ** consecutive_losses

def backup_database():
    conn = sqlite3.connect('coinflips.db')
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    df.to_csv(BACKUP_PATH, index=False)
    conn.close()
    logging.info(f"Database backed up to {BACKUP_PATH}")
