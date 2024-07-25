import pyautogui
import cv2
import pytesseract
import numpy as np
import time
import platform
import os
import sqlite3
from datetime import datetime

# Create or connect to the SQLite database
def create_database():
    try:
        conn = sqlite3.connect('coinflips.db')
        cursor = conn.cursor()
        # Create the Coinflips table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Coinflips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result BOOLEAN,
                amount INTEGER,
                datetime DATETIME,
                consecutive_losses_or_wins INTEGER
            )
        ''')
        conn.commit()
        print("Database and table created successfully.")
    except Exception as e:
        print(f"Error creating database or table: {e}")
    finally:
        conn.close()

# Fetch recent results from the database
def fetch_recent_results():
    try:
        conn = sqlite3.connect('coinflips.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT result, amount, consecutive_losses_or_wins FROM Coinflips
            ORDER BY id DESC LIMIT 100
        ''')
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"Error fetching recent results: {e}")
        return []
    finally:
        conn.close()

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    enhanced_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.medianBlur(enhanced_image, 3)
    return enhanced_image

def detect_outcome(screenshot):
    try:
        _, thresh = cv2.threshold(screenshot, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

    if 'You have won the coinflip' in text:
        return 'win'
    elif 'You have lost the coinflip' in text:
        return 'lose'
    return None

def send_command(command, delay=0.1, click_coords=None):
    time.sleep(3)
    pyautogui.typewrite('t')
    time.sleep(2)
    for char in command:
        pyautogui.typewrite(char, interval=delay)
    pyautogui.press('enter')
    time.sleep(2)
    if click_coords:
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                pyautogui.click(click_coords[0] + dx, click_coords[1] + dy)
                time.sleep(0.5)

def play_notification_sound():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('play -nq -t alsa synth 0.5 sine 440')

def save_to_database(result, amount, consecutive_losses):
    try:
        conn = sqlite3.connect('coinflips.db')
        cursor = conn.cursor()
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        result_boolean = 1 if result == 'win' else 0
        cursor.execute('''
            INSERT INTO Coinflips (result, amount, datetime, consecutive_losses_or_wins)
            VALUES (?, ?, ?, ?)
        ''', (result_boolean, amount, datetime_str, consecutive_losses))
        conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        conn.close()

# Calculate the next bet amount
def calculate_next_bet(consecutive_losses):
    base_bet = 10000  # Starting bet amount
    return base_bet * (2 ** consecutive_losses)

# Calculate statistics
def calculate_statistics(coinflips):
    wins = sum(1 for cf in coinflips if cf[0] == 1)
    losses = sum(1 for cf in coinflips if cf[0] == 0)
    consecutive_wins = 0
    consecutive_losses = 0

    if coinflips:
        last_result = coinflips[0][0]
        for cf in coinflips:
            if cf[0] == last_result:
                if last_result == 1:
                    consecutive_wins += 1
                else:
                    consecutive_losses += 1
            else:
                break

    return wins, losses, consecutive_wins, consecutive_losses

# Ensure profit
def ensure_profit(coinflips, next_bet, consecutive_losses, base_bet=10000):
    total_winnings = sum([cf[1] for cf in coinflips if cf[0] == 1])
    total_losses = sum([cf[1] for cf in coinflips if cf[0] == 0])
    potential_loss = next_bet * (consecutive_losses + 1)
    potential_win = next_bet * 2
    if total_winnings - total_losses + potential_win - potential_loss > 0:
        return True, next_bet
    else:
        adjusted_bet = (total_losses - total_winnings + base_bet) / (2 ** (consecutive_losses + 1))
        return False, adjusted_bet

def main():
    create_database()  # Ensure the database is created
    consecutive_losses = 0
    click_coords = (820, 410)
    screen_width, screen_height = pyautogui.size()
    region_width = 1100
    region_height = 1000
    region_left = 0
    region_top = screen_height - region_height
    region = (region_left, region_top, region_width, region_height)

    while True:
        # Fetch recent results to calculate consecutive losses
        coinflips = fetch_recent_results()
        if coinflips:
            wins, losses, consecutive_wins, consecutive_losses = calculate_statistics(coinflips)

        # Calculate the next bet amount based on consecutive losses
        bet_amount = calculate_next_bet(consecutive_losses)

        # Ensure the betting strategy will be profitable
        profitable, adjusted_bet = ensure_profit(coinflips, bet_amount, consecutive_losses)
        if not profitable:
            bet_amount = adjusted_bet

        send_command(f'/cf {bet_amount}', click_coords=click_coords)

        outcome = None
        timeout = 30
        start_time = time.time()
        while outcome is None and (time.time() - start_time) < timeout:
            screenshot = capture_screen(region=region)
            outcome = detect_outcome(screenshot)
            if outcome is None:
                time.sleep(1)
        if outcome is None:
            print("Timed out waiting for result.")
            continue

        if outcome == 'win':
            consecutive_losses = 0
            print("You won! Resetting losses.")
        elif outcome == 'lose':
            consecutive_losses += 1
            print(f"You lost. Consecutive losses: {consecutive_losses}")

        save_to_database(outcome, bet_amount, consecutive_losses)
        if consecutive_losses >= 4:
            play_notification_sound()
            print("Four consecutive losses. Playing sound and exiting.")
            break
        time.sleep(5)

if __name__ == '__main__':
    main()
