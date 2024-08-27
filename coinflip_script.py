import time
import pandas as pd
import pyautogui
import logging
import schedule
from database import create_database, fetch_recent_results, save_to_database, calculate_win_rates, validate_database_contents, reset_database
from model import backup_database
from utils import capture_screen, detect_outcome, send_command, play_notification_sound

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoinFlipBetting:
    def __init__(self):
        self.min_bet = 10000
        self.current_bet = self.min_bet
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_bets = 0
        self.total_wins = 0
        self.total_losses = 0
        self.balance = 0
        self.initial_balance = 0
        self.performance_log = []
        self.click_coords = (820, 410)
        self.high_loss_coords = (970, 410)
        screen_width, screen_height = pyautogui.size()
        self.region_width = 1100
        self.region_height = 1000
        self.region_left = 0
        self.region_top = screen_height - self.region_height
        self.region = (self.region_left, self.region_top, self.region_width, self.region_height)
        create_database()

    def initialize_from_recent_results(self):
        recent_results = fetch_recent_results(limit=10)  # Fetch last 10 results
        if recent_results:
            # Count consecutive losses from the most recent result
            for result in recent_results:
                if result[0] == 0:  # 0 indicates a loss
                    self.consecutive_losses += 1
                else:
                    break  # Stop counting at the first win

            # Set the current bet based on consecutive losses
            self.current_bet = self.calculate_bet_amount()

            logging.info(f"Initialized from recent results. Consecutive losses: {self.consecutive_losses}, Starting bet: {self.current_bet}")
        else:
            logging.info("No recent results found. Starting with minimum bet.")

    def start(self):
        try:
            self.balance = int(input("Enter the starting balance: "))
            self.initial_balance = self.balance
        except ValueError:
            logging.error("Invalid input for balance. Please enter an integer value.")
            return

        self.initialize_from_recent_results()

        logging.info(f"Initial balance: {self.balance}")
        logging.info(f"Initial consecutive losses: {self.consecutive_losses}")
        logging.info(f"Initial bet amount: {self.current_bet}")

        schedule.every().day.at("00:00").do(self.backup_database)

        while True:
            if self.consecutive_losses > 11:
                logging.info("More than 10 consecutive losses. Stopping the script.")
                break
            self.place_bet()
            schedule.run_pending()
            time.sleep(1)

        logging.info("Betting session ended.")

    def place_bet(self):
        max_retries = 3
        for attempt in range(max_retries):
            self.current_bet = self.calculate_bet_amount()
            logging.info(f"Placing bet: {self.current_bet}")

            try:
                coords = self.high_loss_coords if self.consecutive_losses >= 2 else self.click_coords
                self.execute_bet_sequence(self.current_bet, coords)
                outcome = self.wait_for_result(self.region)
                if outcome is not None:
                    self.handle_outcome(outcome)
                    break
                else:
                    logging.warning("Failed to detect result. Retrying...")
            except Exception as e:
                logging.error(f"Error during bet sequence: {e}")
                if attempt == max_retries - 1:
                    raise

        time.sleep(10)

    def calculate_bet_amount(self):
        logging.info(f"Calculating bet amount. Consecutive losses: {self.consecutive_losses}")
        if self.consecutive_losses == 0:
            bet = max(self.min_bet, self.round_to_nearest_10k(self.balance * 0.03))
        if self.consecutive_losses == 1:
            bet = max(self.min_bet, self.round_to_nearest_10k(self.balance * 0.06))
        elif self.consecutive_losses >= 2:
            bet = self.handle_high_consecutive_losses()
        else:
            bet = self.min_bet
        logging.info(f"Calculated bet amount: {bet}")
        return bet

    def round_to_nearest_10k(self, amount):
        return round(amount / 10000) * 10000

    def execute_bet_sequence(self, bet_amount, click_coords):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                send_command(f'/cf {int(bet_amount)}', click_coords=click_coords)
                return
            except Exception as e:
                logging.error(f"Error executing bet sequence (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise

    def wait_for_result(self, region):
        timeout = 420
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            screenshot = capture_screen(region=region)
            outcome = detect_outcome(screenshot)
            if outcome is not None:
                return outcome
            logging.info("Waiting for result...")
            time.sleep(3)
        return None

    def handle_high_consecutive_losses(self):

        if self.consecutive_losses == 2:
            return 1
        elif self.consecutive_losses == 3:
            return 3
        elif self.consecutive_losses == 4:
            return 7
        elif self.consecutive_losses == 5:
            return 15
        elif self.consecutive_losses == 6:
            return 31
        elif self.consecutive_losses == 7:
            return 65
        elif self.consecutive_losses == 8:
            return 130
        elif self.consecutive_losses == 9:
            return 260
        elif self.consecutive_losses == 10:
            return 520
        elif self.consecutive_losses == 11:
            return 920
        else:
            return self.min_bet


        # def handle_high_consecutive_losses(self):

        # 1 3 6 13 28 55 110 220 440 880 1000
        #     if self.consecutive_losses == 3:
        #         return 2
        #     elif self.consecutive_losses == 4:
        #         return 5
        #     elif self.consecutive_losses == 5:
        #         return 11
        #     elif self.consecutive_losses == 6:
        #         return 24
        #     elif self.consecutive_losses == 7:
        #         return 50
        #     elif self.consecutive_losses == 8:
        #         return 102
        #     elif self.consecutive_losses == 9:
        #         return 210
        #     elif self.consecutive_losses == 10:
        #         return 450
        #     else:
        #         return self.min_bet

    def handle_outcome(self, outcome):
        if outcome == 'win':
            self.balance += self.current_bet
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            logging.info(f"You won! New balance: {self.balance}")
        elif outcome == 'lose':
            self.balance -= self.current_bet
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
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
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses
        }
        self.performance_log.append(performance_entry)

        if len(self.performance_log) % 50 == 0:
            self.analyze_performance()

    def analyze_performance(self):
        df = pd.DataFrame(self.performance_log)
        df['cumulative_profit'] = df['balance'] - self.initial_balance
        df['rolling_win_rate'] = df['outcome'].apply(lambda x: 1 if x == 'win' else 0).rolling(window=50).mean()

        logging.info(f"Performance analysis:")
        logging.info(f"Total profit: {df['cumulative_profit'].iloc[-1]}")
        logging.info(f"Current win rate: {df['rolling_win_rate'].iloc[-1]:.2f}")
        logging.info(f"Average bet amount: {df['bet_amount'].mean():.2f}")

        df.to_csv('performance_log.csv', index=False)

    def backup_database(self):
        logging.info("Backing up database...")
        backup_database()

if __name__ == '__main__':
    validate_database_contents()
    betting_system = CoinFlipBetting()
    betting_system.start()