# main.py

import time
import pyautogui
from database import create_database, save_to_database, get_result_count, fetch_latest_result
from utils import capture_screen, detect_outcome
from model import load_model, predict_next_bet, train_model, calculate_bet_amount

def main():
    create_database()
    model, scaler = load_model()
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

    print(f"Screen size: {screen_width}x{screen_height}")
    print(f"Capture region: {region}")

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
    retrain_interval = 10
    last_processed_count = get_result_count()
    consecutive_losses = 0

    while True:
        print("Waiting for 5 seconds...")
        time.sleep(5)

        print("Typing 't' command...")
        pyautogui.typewrite('t')
        time.sleep(1)

        if model is not None and scaler is not None:
            prediction = predict_next_bet(model, scaler)
            bet_amount = calculate_bet_amount(prediction, balance, consecutive_losses, min_bet)
        else:
            prediction = 0.5
            bet_amount = min_bet

        command = f'/cf {int(bet_amount)}'
        print(f"Typing command: {command}")
        pyautogui.typewrite(command)
        pyautogui.press('enter')

        print("Clicking...")
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                pyautogui.click(click_coords[0] + dx, click_coords[1] + dy)
                time.sleep(0.5)

        outcome = None
        timeout = 30
        start_time = time.time()
        while outcome is None and (time.time() - start_time) < timeout:
            print("Capturing screen...")
            screenshot = capture_screen(region=region)
            print("Screen captured. Detecting outcome...")
            outcome = detect_outcome(screenshot)
            if outcome is None:
                print("Waiting for result...")
                time.sleep(2)

        if outcome is None:
            print("Timed out waiting for result.")
            continue

        print(f"Outcome detected: {outcome}")

        if outcome == 'win':
            balance += bet_amount
            consecutive_losses = 0
            print(f"You won! New balance: {balance}")
        elif outcome == 'lose':
            balance -= bet_amount
            consecutive_losses += 1
            print(f"You lost. New balance: {balance}")

        print("Saving to database...")
        save_to_database(outcome, bet_amount, balance)
        print("Saved to database.")

        flips_since_last_train += 1
        if flips_since_last_train >= retrain_interval:
            print("Retraining model...")
            model, scaler = train_model()
            flips_since_last_train = 0
        if model is None or scaler is None:
            print("Still not enough data to train model. Continuing with default strategy.")

        last_processed_count = get_result_count()

if __name__ == '__main__':
    main()
