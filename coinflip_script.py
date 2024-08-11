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
        self.max_bet_fraction = 0.1
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
        screen_width, screen_height = pyautogui.size()
        self.region_width = 1100
        self.region_height = 1000
        self.region_left = 0
        self.region_top = screen_height - self.region_height
        self.region = (self.region_left, self.region_top, self.region_width, self.region_height)
        self.confidence = 0.5
        self.low_confidence_threshold = 0.6
        self.stop_loss = 0.5  # Allow losses up to 50% of initial balance
        self.take_profit = 0.5
        self.aggressive_mode = False
        self.aggressive_bet = 0
        create_database()

    def start(self):
        try:
            self.balance = int(input("Enter the starting balance: "))
            self.initial_balance = self.balance
        except ValueError:
            logging.error("Invalid input for balance. Please enter an integer value.")
            return

        schedule.every().day.at("00:00").do(self.backup_database)

        while True:
            if self.check_stop_conditions():
                break
            self.place_bet()
            schedule.run_pending()
            time.sleep(1)

    def place_bet(self):
        max_retries = 3
        for attempt in range(max_retries):
            results = fetch_recent_results(1000)
            win_rate_total, win_rate_50, win_rate_20 = calculate_win_rates(results)

            logging.info(f"Total win rate: {win_rate_total:.2f}")
            logging.info(f"Last 50 games win rate: {win_rate_50:.2f}")
            logging.info(f"Last 20 games win rate: {win_rate_20:.2f}")

            if not self.aggressive_mode:
                self.update_confidence(win_rate_total, win_rate_50, win_rate_20)
            self.current_bet = self.calculate_bet_amount()
            logging.info(f"Confidence: {self.confidence:.2f}, Calculated bet amount: {self.current_bet}")

            if not self.aggressive_mode and self.confidence <= self.low_confidence_threshold:
                self.current_bet = self.get_min_bet()
                logging.info(f"Low confidence. Placing minimum bet: {self.current_bet}")

            try:
                self.execute_bet_sequence(self.current_bet)
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

    def get_min_bet(self):
        return min(self.min_bet, self.balance)

    def update_confidence(self, win_rate_total, win_rate_50, win_rate_20):
        if win_rate_20 < 0.4 and win_rate_50 < 0.45 and win_rate_total < 0.48:
            self.confidence = min(1.0, self.confidence + 0.1)
        elif win_rate_20 > 0.6 or win_rate_50 > 0.55:
            self.confidence = max(0.1, self.confidence - 0.1)

        if self.consecutive_losses > 3:
            self.confidence = min(1.0, self.confidence + 0.05 * (self.consecutive_losses - 3))
        elif self.consecutive_wins > 3:
            self.confidence = max(0.1, self.confidence - 0.05 * (self.consecutive_wins - 3))

    def calculate_bet_amount(self):
        if self.aggressive_mode:
            return self.round_to_nearest_10k(self.aggressive_bet)
        base_bet = self.balance * self.max_bet_fraction * self.confidence
        bet_amount = max(self.min_bet, int(base_bet))
        bet_amount = min(bet_amount, self.balance * self.max_bet_fraction)
        return self.round_to_nearest_10k(bet_amount)

    def round_to_nearest_10k(self, amount):
        return round(amount / 10000) * 10000

    def execute_bet_sequence(self, bet_amount):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                send_command(f'/cf {int(bet_amount)}', click_coords=self.click_coords)
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
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.aggressive_mode = False
            logging.info(f"You won! New balance: {self.balance}")
        elif outcome == 'lose':
            self.balance -= self.current_bet
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            logging.info(f"You lost. New balance: {self.balance}")

            if self.consecutive_losses == 2:
                self.aggressive_mode = True
                self.aggressive_bet = self.round_to_nearest_10k(self.initial_balance * 0.04)
                logging.info(f"Entering aggressive mode. Next bet: {self.aggressive_bet}")
            elif self.consecutive_losses > 2 and self.aggressive_mode:
                self.aggressive_bet = self.round_to_nearest_10k(self.aggressive_bet * 2.2)
                logging.info(f"Increasing aggressive bet. Next bet: {self.aggressive_bet}")

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
            'confidence': self.confidence,
            'aggressive_mode': self.aggressive_mode
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
        logging.info(f"Average confidence: {df['confidence'].mean():.2f}")

        df.to_csv('performance_log.csv', index=False)

    def check_stop_conditions(self):
        if self.balance <= self.initial_balance * (1 - self.stop_loss):
            logging.info(f"Stop loss reached. Stopping betting.")
            return True
        elif self.balance >= self.initial_balance * (1 + self.take_profit):
            logging.info(f"Take profit reached. Stopping betting.")
            return True
        return False

    def backup_database(self):
        logging.info("Backing up database...")
        backup_database()

if __name__ == '__main__':
    validate_database_contents()
    betting_system = CoinFlipBetting()
    betting_system.start()