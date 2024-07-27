import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib

# Assuming these functions are defined in a 'database.py' file
from database import fetch_all_results, get_result_count, fetch_latest_result

MODEL_PATH = 'coinflip_model.h5'
SCALER_PATH = 'scaler.pkl'

def prepare_data(sequence_length=50):
    data = fetch_all_results()
    if len(data) == 0:
        print("No data available. Unable to prepare data for the model.")
        return None, None, None

    # Ensure 'datetime' column is in datetime format
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])  # Remove rows where 'datetime' could not be parsed

    data = data.sort_values('datetime', ascending=False).reset_index(drop=True)

    # Use the most recent 'sequence_length' entries
    data = data.head(sequence_length)

    # If we have less than sequence_length entries, pad with default values
    if len(data) < sequence_length:
        pad_length = sequence_length - len(data)
        pad_data = pd.DataFrame({
            'result': [0.5] * pad_length,
            'amount': [10000] * pad_length,
            'datetime': [data['datetime'].min()] * pad_length
        })
        data = pd.concat([pad_data, data]).reset_index(drop=True)

    # Reverse the order so that the most recent is last
    data = data.iloc[::-1].reset_index(drop=True)

    # Calculate features
    data['total_winrate'] = data['result'].cumsum() / (data.index + 1)
    data['winrate_50'] = data['result'].rolling(window=50).mean().fillna(0.5)
    data['winrate_20'] = data['result'].rolling(window=20).mean().fillna(0.5)
    data['consecutive_losses'] = data['result'].groupby((data['result'] != data['result'].shift()).cumsum()).cumcount()
    data.loc[data['result'] == 1, 'consecutive_losses'] = 0

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[['total_winrate', 'winrate_50', 'winrate_20', 'consecutive_losses', 'amount']])
    y = data['result'].values

    # Reshape X to be 3D: [samples, time steps, features]
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0]))

    print("X shape in prepare_data:", X.shape)
    print("y shape in prepare_data:", y.shape)
    return X, y, scaler

def create_model(input_shape):
    print("Input shape in create_model:", input_shape)
    model = Sequential([
        LSTM(50, input_shape=(input_shape[1], input_shape[2]), return_sequences=True),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y, scaler = prepare_data()
    print("X shape in train_model:", X.shape)
    print("y shape in train_model:", y.shape)
    if X is None or y is None:
        print("Not enough data to train the model. Please play more games.")
        return None, None
    model = create_model(X.shape)
    model.summary()
    model.fit(X, y, epochs=100, batch_size=1, verbose=1)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def load_model():
    X, _, _ = prepare_data()
    if X is None:
        print("Not enough data to create or load the model.")
        return None, None

    expected_input_shape = X.shape

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            if model.input_shape[1:] != expected_input_shape[1:]:
                raise ValueError("Loaded model input shape doesn't match expected input shape")
            scaler = joblib.load(SCALER_PATH)
            print("Model loaded successfully")
        except:
            print("Error loading model or model shape mismatch. Recreating model...")
            model = None
    else:
        model = None

    if model is None:
        print("Creating new model...")
        model = create_model(expected_input_shape)
        model.summary()
        scaler = MinMaxScaler()
        # Train the model here if you have enough data
        # model.fit(X, y, epochs=100, batch_size=1, verbose=1)
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    return model, scaler

def predict_next_bet(model, scaler):
    X, _, _ = prepare_data()
    if X is None:
        print("Not enough data to make a prediction. Using default 50% win probability.")
        return 0.5
    print("X shape in predict_next_bet:", X.shape)
    prediction = model.predict(X)
    return prediction[0][0]

def calculate_bet_amount(prediction, balance, consecutive_losses, min_bet=10000):
    if prediction > 0.5:
        # Increase bet based on confidence and consecutive losses
        confidence = (prediction - 0.5) * 2  # Scale confidence to 0-1
        compound_factor = 1.5 ** consecutive_losses  # Increase bet exponentially with consecutive losses
        bet = min_bet * (1 + confidence) * compound_factor
        return min(bet, balance * 0.1)  # Don't bet more than 10% of balance
    else:
        return min_bet

def main():
    last_processed_count = 0
    consecutive_losses = 0
    balance = int(input("Enter the starting balance (up to 100,000,000): "))

    while True:
        current_count = get_result_count()

        if current_count > last_processed_count:
            model, scaler = load_model()
            if model is None or scaler is None:
                print("Not enough data to train the model. Waiting for more results...")
                time.sleep(5)
                continue

            X, _, _ = prepare_data()
            if X is None:
                print("Not enough data for prediction. Waiting for more results...")
                time.sleep(5)
                continue

            prediction = model.predict(X)[0][0]

            min_bet = 10000
            bet_amount = calculate_bet_amount(prediction, balance, consecutive_losses, min_bet)

            print(f"New result received. Total results: {current_count}")
            print(f"Prediction (win probability): {prediction:.2f}")
            print(f"Consecutive losses: {consecutive_losses}")
            print(f"Recommended bet amount: {bet_amount:.0f}")

            print(f"Placing bet of {bet_amount:.0f}")

            # Simulating waiting for result
            time.sleep(10)

            # After getting the result, update consecutive_losses
            result = fetch_latest_result()
            if result == 0:  # Loss
                consecutive_losses += 1
                balance -= bet_amount
            else:  # Win
                consecutive_losses = 0
                balance += bet_amount

            print(f"New balance: {balance}")

            last_processed_count = current_count
        else:
            print("Waiting for new result...")
            time.sleep(5)

if __name__ == "__main__":
    main()