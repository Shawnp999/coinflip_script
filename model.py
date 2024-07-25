import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from database import fetch_all_results

MODEL_PATH = 'coinflip_model.h5'
SCALER_PATH = 'scaler.pkl'

def prepare_data(sequence_length=100):
    data = fetch_all_results()
    data = data.sort_values('datetime', ascending=False).reset_index(drop=True)

    # Use only the most recent 'sequence_length' entries
    data = data.head(sequence_length)

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
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=100, batch_size=1, verbose=1)
    model.save(MODEL_PATH)
    import joblib
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Training the model...")
        return train_model()
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
    model, scaler = load_model()
    balance = 50_000_000  # Starting balance
    min_bet = 10000

    prediction = predict_next_bet(model, scaler)
    bet_amount = calculate_bet_amount(prediction, balance, min_bet)

    print(f"Prediction (win probability): {prediction:.2f}")
    print(f"Recommended bet amount: {bet_amount:.0f}")

    # Print model summary
    print("\nModel Summary:")
    model.summary()

if __name__ == "__main__":
    main()