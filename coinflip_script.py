import time

import pandas as pd
import pyautogui
import logging
import schedule
from database import create_database, fetch_recent_results, save_to_database, fetch_all_results, validate_database_contents, reset_database
from utils import capture_screen, detect_outcome, send_command, play_notification_sound
from model import prepare_data, create_model, train_model, predict_next_bet, load_model_and_scaler, calculate_bet_amount, backup_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoinFlipBetting:
    def __init__(self):
        self.model, self.scaler = train_model()  # Force retrain with new features
        self.min_bet = 10000
        self.current_bet = self.min_bet
        self.consecutive_losses = 0
        self.flips_since_last_train = 0
        self.retrain_interval = 20  # Retrain every 20 coinflips
        self.total_bets = 0
        self.total_wins = 0
        self.total_losses = 0
        self.balance = 0
        self.performance_log = []
        self.click_coords = (820, 410)
        screen_width, screen_height = pyautogui.size()
        self.region_width = 1100
        self.region_height = 1000
        self.region_left = 0
        self.region_top = screen_height - self.region_height
        self.region = (self.region_left, self.region_top, self.region_width, self.region_height)
        create_database()

    def start(self):
        try:
            self.balance = int(input("Enter the starting balance: "))
        except ValueError:
            logging.error("Invalid input for balance. Please enter an integer value.")
            return

        schedule.every().hour.do(self.retrain_model)
        schedule.every().day.at("00:00").do(self.backup_database)

        while True:
            self.place_bet()
            schedule.run_pending()
            time.sleep(1)

    def place_bet(self):
        max_retries = 3
        for attempt in range(max_retries):
            if self.model is not None and self.scaler is not None:
                prediction = predict_next_bet(self.model, self.scaler)
                self.current_bet = calculate_bet_amount(prediction, self.balance, self.min_bet)
            else:
                prediction = 0.5
                self.current_bet = self.min_bet

            if 0.45 <= prediction <= 0.55:
                logging.info(f"Prediction uncertain ({prediction:.2f}). Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(30)
                    continue
                else:
                    logging.info("Max retries reached. Using minimum bet as fallback.")
                    self.current_bet = self.min_bet

            try:
                self.execute_bet_sequence(self.current_bet)

                # Wait for and detect the result
                outcome = self.wait_for_result(self.region)
                if outcome is not None:
                    self.handle_outcome(outcome)
                    break  # Exit the retry loop if we got a result
                else:
                    logging.warning("Failed to detect result. Retrying...")
            except Exception as e:
                logging.error(f"Error during bet sequence: {e}")
                if attempt == max_retries - 1:
                    raise

        # After attempting to place a bet (whether successful or not), wait before the next iteration
        time.sleep(10)

    def execute_bet_sequence(self, bet_amount):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(5)
                pyautogui.typewrite('t')
                time.sleep(1)
                pyautogui.typewrite(f'/cf {int(bet_amount)}')
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(1)
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        pyautogui.click(self.click_coords[0] + dx, self.click_coords[1] + dy)
                        time.sleep(0.5)
                return
            except Exception as e:
                logging.error(f"Error executing bet sequence (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise

    def wait_for_result(self, region):
        timeout = 60
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            screenshot = capture_screen(region=region)
            outcome = detect_outcome(screenshot)
            if outcome is not None:
                return outcome
            logging.info("Waiting for result...")
            time.sleep(3)
        return None

    def handle_outcome(self, outcome):
        if outcome == 'win':
            self.balance += self.current_bet
            self.total_wins += 1
            self.consecutive_losses = 0
            logging.info(f"You won! New balance: {self.balance}")
        elif outcome == 'lose':
            self.balance -= self.current_bet
            self.total_losses += 1
            self.consecutive_losses += 1
            logging.info(f"You lost. New balance: {self.balance}")

        self.total_bets += 1
        win_rate = self.total_wins / self.total_bets if self.total_bets > 0 else 0

        logging.info(f"Total bets: {self.total_bets}, Total wins: {self.total_wins}, Total losses: {self.total_losses}, Win rate: {win_rate:.2f}")

        save_to_database(outcome, self.current_bet, self.consecutive_losses)

        performance_entry = {
            'timestamp': time.time(),
            'bet_amount': self.current_bet,
            'outcome': outcome,
            'balance': self.balance,
            'win_rate': win_rate
        }
        self.performance_log.append(performance_entry)

        if len(self.performance_log) % 100 == 0:
            self.analyze_performance()

    def analyze_performance(self):
        df = pd.DataFrame(self.performance_log)
        df['cumulative_profit'] = df['balance'] - df['balance'].iloc[0]
        df['rolling_win_rate'] = df['outcome'].apply(lambda x: 1 if x == 'win' else 0).rolling(window=50).mean()

        logging.info(f"Performance analysis:")
        logging.info(f"Total profit: {df['cumulative_profit'].iloc[-1]}")
        logging.info(f"Current win rate: {df['rolling_win_rate'].iloc[-1]:.2f}")
        logging.info(f"Average bet amount: {df['bet_amount'].mean():.2f}")

        df.to_csv('performance_log.csv', index=False)

    def retrain_model(self):
        logging.info("Retraining model...")
        try:
            self.model, self.scaler = train_model()
            self.flips_since_last_train = 0
        except ValueError as e:
            logging.warning(f"Skipping model training due to insufficient data: {e}")

    def backup_database(self):
        logging.info("Backing up database...")
        backup_database()

if __name__ == '__main__':
    # Uncomment the next line to reset the database if needed
    # reset_database()
    validate_database_contents()
    betting_system = CoinFlipBetting()
    betting_system.start()
