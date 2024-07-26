import os
from datetime import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from database import fetch_all_results, get_result_count

MODEL_PATH = 'coinflip_model.h5'
SCALER_PATH = 'scaler.pkl'

def prepare_data(sequence_length=15):
    data = fetch_all_results()
    if len(data) == 0:
        print("No data available. Unable to prepare data for the model.")
        return None, None, None

    # Ensure 'datetime' column is in datetime format
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
            'datetime': [data['datetime'].min()] * pad_length
        })
        data = pd.concat([pad_data, data]).reset_index(drop=True)

    # Reverse the order so that the most recent is last
    data = data.iloc[::-1].reset_index(drop=True)

    # Calculate features
    data['total_winrate'] = data['result'].cumsum() / (data.index + 1)
    data['recent_winrate'] = data['result'].rolling(window=10).mean().fillna(0.5)
    data['consecutive_losses'] = data['result'].groupby((data['result'] != data['result'].shift()).cumsum()).cumcount()
    data.loc[data['result'] == 1, 'consecutive_losses'] = 0

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[['total_winrate', 'recent_winrate', 'consecutive_losses', 'amount']])
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
        print("Not enough data to train the model. Please play more games.")
        return None, None
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=100, batch_size=1, verbose=1)
    model.save(MODEL_PATH)
    import joblib
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def predict_next_bet(model, scaler):
    X, _, _ = prepare_data()
    if X is None:
        print("Not enough data to make a prediction. Using default 50% win probability.")
        return 0.5
    prediction = model.predict(X)
    return prediction[0][-1][0]

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Training the model...")
        model, scaler = train_model()
        if model is None or scaler is None:
            print("Failed to train the model. Using default prediction.")
            return None, None
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        import joblib
        scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_next_bet(model, scaler):
    X, _, _ = prepare_data()
    prediction = model.predict(X)
    return prediction[0][-1][0]

def calculate_bet_amount(prediction, balance, min_bet=10000):
    if prediction > 0.5:
        # Increase bet based on confidence
        confidence = (prediction - 0.5) * 2  # Scale confidence to 0-1
        bet = min_bet * (1 + confidence)
        return min(bet, balance * 0.1)  # Don't bet more than 10% of balance
    else:
        return min_bet

def main():
    last_processed_count = 0

    while True:
        current_count = get_result_count()

        if current_count > last_processed_count:
            model, scaler = load_model()
            if model is None or scaler is None:
                print("Not enough data to train the model. Waiting for more results...")
                time.sleep(5)  # Wait for 5 seconds before checking again
                continue

            prediction = predict_next_bet(model, scaler)

            if prediction == 0.5:
                print("Not enough data for prediction. Waiting for more results...")
                time.sleep(5)  # Wait for 5 seconds before checking again
                continue

            balance = 50_000_000  # This should be replaced with the actual current balance
            min_bet = 10000
            bet_amount = calculate_bet_amount(prediction, balance, min_bet)

            print(f"New result received. Total results: {current_count}")
            print(f"Prediction (win probability): {prediction:.2f}")
            print(f"Recommended bet amount: {bet_amount:.0f}")

            # Here you would typically place the bet using the calculated bet_amount
            # For demonstration, we'll just print it
            print(f"Placing bet of {bet_amount:.0f}")

            # Wait for the result of this bet
            # This part should be replaced with actual result detection logic
            time.sleep(10)  # Simulating waiting for result

            # After getting the result, retrain the model
            print("Updating the model...")
            model, scaler = train_model()
            if model is None or scaler is None:
                print("Failed to update the model. Using previous model or default strategy.")

            last_processed_count = current_count
        else:
            print("Waiting for new result...")
            time.sleep(5)  # Check for new results every 5 seconds

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()