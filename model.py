import os
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
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
    data['win_streak'] = (data['result'] != data['result'].shift(1)).cumsum()
    data.loc[data['result'] == 0, 'win_streak'] = 0
    data['loss_streak'] = (data['result'] != data['result'].shift(1)).cumsum()
    data.loc[data['result'] == 1, 'loss_streak'] = 0

    data['rolling_win_rate_10'] = data['result'].rolling(window=10).mean()
    data['rolling_win_rate_20'] = data['result'].rolling(window=20).mean()
    data['rolling_win_rate_50'] = data['result'].rolling(window=50).mean()

    data['bet_amount_ratio'] = data['amount'] / data['balance']
    data['profit_loss'] = data['result'].apply(lambda x: 1 if x else -1) * data['amount']
    data['cumulative_profit_loss'] = data['profit_loss'].cumsum()

    features = ['total_winrate', 'recent_winrate', 'consecutive_losses', 'amount', 'consecutive_losses_or_wins',
                'win_streak', 'loss_streak', 'rolling_win_rate_10', 'rolling_win_rate_20', 'rolling_win_rate_50',
                'bet_amount_ratio', 'cumulative_profit_loss']

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features])
    y = data['result'].values

    return X, y, scaler

def create_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=(None, input_shape), return_sequences=True, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y, scaler = prepare_data()
    if X is None or y is None or len(X) < 2:  # Check if there are at least 2 samples
        logging.warning("Not enough data to train the model. Please play more games.")
        return None, None

    model = create_model(X.shape[1])  # Pass the number of features

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Reshape X for LSTM input
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model.fit(X, y, epochs=300, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def predict_next_bet(model, scaler):
    X, _, _ = prepare_data()
    if X is None or len(X) == 0:
        logging.warning("Not enough data to make a prediction. Using default 50% win probability.")
        return 0.5

    # Use only the most recent data point
    latest_data = X[-1].reshape(1, -1)

    # Scale the data
    scaled_data = scaler.transform(latest_data)

    # Reshape for LSTM input (samples, time steps, features)
    X_reshaped = scaled_data.reshape((1, 1, scaled_data.shape[1]))

    prediction = model.predict(X_reshaped)
    return prediction[0][0]

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

def calculate_bet_amount(prediction, balance, min_bet=10000, max_bet_fraction=0.1):
    confidence = abs(prediction - 0.5) * 2
    base_bet = balance * max_bet_fraction * confidence
    bet_amount = max(min_bet, int(base_bet))
    bet_amount = min(bet_amount, balance * max_bet_fraction)
    return bet_amount

def backup_database():
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    df.to_csv(BACKUP_PATH, index=False)
    conn.close()
    logging.info(f"Database backed up to {BACKUP_PATH}")
