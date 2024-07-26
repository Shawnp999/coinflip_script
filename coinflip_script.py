import time
import pyautogui
from database import create_database, fetch_recent_results, save_to_database
from utils import capture_screen, detect_outcome, send_command
from model import load_model, predict_next_bet, train_model, calculate_bet_amount

def main():
    create_database()  # Ensure the database is created
    model, scaler = load_model()  # Load the trained model and scaler
    if model is None or scaler is None:
        print("Unable to load or train model. Starting with default strategy.")
        model, scaler = None, None

    click_coords = (820, 410)
    screen_width, screen_height = pyautogui.size()
    region_width = 1100
    region_height = 1000
    region_left = 0
    region_top = screen_height - region_height
    region = (region_left, region_top, region_width, region_height)

    # Get the balance from user input
    while True:
        try:
            balance = int(input("Enter the starting balance (up to 100,000,000): "))
            if 0 < balance <= 100_000_000:
                break
            else:
                print("Balance must be a positive integer not exceeding 100,000,000.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    min_bet = 10000
    flips_since_last_train = 0
    retrain_interval = 50  # Retrain every 50 flips

    while True:
        if model is not None and scaler is not None:
            prediction = predict_next_bet(model, scaler)
            bet_amount = calculate_bet_amount(prediction, balance, min_bet)
        else:
            prediction = 0.5  # Default to 50% win probability
            bet_amount = min_bet

        send_command(f'/cf {int(bet_amount)}', click_coords=click_coords)

        outcome = None
        timeout = 30
        start_time = time.time()
        while outcome is None and (time.time() - start_time) < timeout:
            screenshot = capture_screen(region=region)
            outcome = detect_outcome(screenshot)
            if outcome is None:
                print("Waiting for result...")
                time.sleep(1)
        if outcome is None:
            print("Timed out waiting for result.")
            continue

        if outcome == 'win':
            balance += bet_amount
            print(f"You won! New balance: {balance}")
        elif outcome == 'lose':
            balance -= bet_amount
            print(f"You lost. New balance: {balance}")

        save_to_database(outcome, bet_amount, 0)  # Save the result to the database

        flips_since_last_train += 1
        if flips_since_last_train >= retrain_interval:
            print("Retraining model...")
            model, scaler = train_model()
            flips_since_last_train = 0
            if model is None or scaler is None:
                print("Still not enough data to train model. Continuing with default strategy.")

        time.sleep(5)

if __name__ == '__main__':
    main()

