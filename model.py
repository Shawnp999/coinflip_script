import os
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib

from database import fetch_all_results, DATABASE_NAME

MODEL_PATH = 'coinflip_model.h5'
SCALER_PATH = 'scaler.pkl'
BACKUP_PATH = 'coinflips_backup.csv'

def prepare_data(sequence_length=50):
    data = fetch_all_results()
    if data.empty:
        logging.warning("No data available. Unable to prepare data for the model.")
        return None, None, None

    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime']).sort_values('datetime', ascending=False).reset_index(drop=True)

    if len(data) < sequence_length:
        pad_length = sequence_length - len(data)
        pad_data = pd.DataFrame({
            'result': [0.5] * pad_length,
            'amount': [10000] * pad_length,
            'datetime': [data['datetime'].min()] * pad_length,
            'consecutive_losses_or_wins': [0] * pad_length,
            'balance': [data['amount'].sum()] * pad_length
        })
        data = pd.concat([pad_data, data]).reset_index(drop=True)

    data = data.iloc[::-1].reset_index(drop=True)

    logging.info(f"Data used for training: {data}")

    data['total_winrate'] = data['result'].cumsum() / (data.index + 1)
    data['recent_winrate'] = data['result'].rolling(window=10).mean().fillna(0.5)
    data['consecutive_losses'] = data['consecutive_losses_or_wins']
    data.loc[data['result'] == 1, 'consecutive_losses'] = 0

    data['balance'] = data['amount'].cumsum()

    features = ['total_winrate', 'recent_winrate', 'consecutive_losses', 'amount', 'consecutive_losses_or_wins']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features])

    logging.info(f"Number of features: {X.shape[1]}")

    y = data['result'].values

    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], 1))

    return X, y, scaler

def create_model(input_shape):
    model = Sequential([
        LSTM(100, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y, scaler = prepare_data()
    if X is None or y is None:
        logging.warning("Not enough data to train the model. Please play more games.")
        return None, None
    model = create_model((X.shape[2], X.shape[1]))  # Fix input shape
    model.fit(X, y, epochs=200, batch_size=1, verbose=1, validation_split=0.2)
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

def calculate_bet_amount(prediction, balance, min_bet=10000, max_bet_fraction=0.5, loss_streak_multiplier=1.5, win_streak_multiplier=1.1):
    """
    Calculate the bet amount based on the model prediction, balance, and additional factors.

    Parameters:
    - prediction: Probability of winning (from the model).
    - balance: Current balance.
    - min_bet: Minimum bet amount.
    - max_bet_fraction: Maximum fraction of the balance to bet.
    - loss_streak_multiplier: Multiplier to increase bet after losses.
    - win_streak_multiplier: Multiplier to adjust bet after wins.

    Returns:
    - bet_amount: Calculated bet amount.
    """
    if prediction > 0.5:
        base_bet = balance * win_streak_multiplier * prediction
    else:
        base_bet = balance * loss_streak_multiplier * (1 - prediction)

    # Ensure the bet amount is at least min_bet and not more than a fraction of the balance
    bet_amount = max(min_bet, int(base_bet))
    bet_amount = min(bet_amount, balance * max_bet_fraction)

    logging.info(f"Calculated bet amount: {bet_amount}, based on prediction: {prediction:.2f}, balance: {balance}")

    return bet_amount


def backup_database():
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    df.to_csv(BACKUP_PATH, index=False)
    conn.close()
    logging.info(f"Database backed up to {BACKUP_PATH}")
