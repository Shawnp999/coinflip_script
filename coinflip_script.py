import time
import pyautogui
import logging
import schedule
from database import create_database, fetch_recent_results, save_to_database, fetch_all_results, validate_database_contents, reset_database
from utils import capture_screen, detect_outcome, send_command, play_notification_sound
from model import prepare_data, create_model, train_model, predict_next_bet, load_model_and_scaler, calculate_bet_amount, backup_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoinFlipBetting:
    def __init__(self):
        self.model, self.scaler = load_model_and_scaler()
        self.min_bet = 10000
        self.current_bet = self.min_bet
        self.consecutive_losses = 0
        self.flips_since_last_train = 0
        self.retrain_interval = 20  # Retrain every 20 coinflips
        self.total_bets = 0
        self.total_wins = 0
        self.total_losses = 0
        self.balance = 0
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
        if self.model is not None and self.scaler is not None:
            prediction = predict_next_bet(self.model, self.scaler)
            self.current_bet = calculate_bet_amount(prediction, self.balance, self.min_bet)
        else:
            prediction = 0.5
            self.current_bet = self.min_bet

        # Ensure the bet amount is at least the minimum bet
        if self.current_bet < self.min_bet:
            self.current_bet = self.min_bet

        logging.info(f"Placing bet: /cf {int(self.current_bet)} with current balance: {self.balance}")
        self.execute_bet_sequence(self.current_bet)

        outcome = self.wait_for_result(self.region)
        if outcome is None:
            logging.warning("Timed out waiting for result.")
            return

        self.handle_outcome(outcome)

        logging.info(f"Balance after bet: {self.balance}")

        self.flips_since_last_train += 1
        if self.flips_since_last_train >= self.retrain_interval:
            self.retrain_model()

        time.sleep(10)

    def execute_bet_sequence(self, bet_amount):
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

    def wait_for_result(self, region):
        timeout = 60
        start_time = time.time()
        outcome = None
        while outcome is None and (time.time() - start_time) < timeout:
            screenshot = capture_screen(region=region)
            outcome = detect_outcome(screenshot)
            if outcome is None:
                logging.info("Waiting for result...")
                time.sleep(3)
        return outcome

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
