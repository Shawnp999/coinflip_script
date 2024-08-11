# database.py

import sqlite3
import pandas as pd
from datetime import datetime
import logging

DATABASE_NAME = 'coinflips.db'

def create_database():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
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
        logging.info("Database and table created successfully.")
    except Exception as e:
        logging.error(f"Error creating database or table: {e}")
    finally:
        conn.close()

def reset_database():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS Coinflips')
        conn.commit()
        create_database()
        logging.info("Database reset successfully.")
    except Exception as e:
        logging.error(f"Error resetting database: {e}")
    finally:
        conn.close()

def fetch_recent_results(limit=1000):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT result, amount, consecutive_losses_or_wins FROM Coinflips
            ORDER BY id DESC LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
        logging.info(f"Fetched {len(results)} recent results from database.")
        return results
    except Exception as e:
        logging.error(f"Error fetching recent results: {e}")
        return []
    finally:
        conn.close()

def save_to_database(result, amount, consecutive_losses):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_boolean = 1 if result == 'win' else 0
        cursor.execute('''
            INSERT INTO Coinflips (result, amount, datetime, consecutive_losses_or_wins)
            VALUES (?, ?, ?, ?)
        ''', (result_boolean, amount, now, consecutive_losses))
        conn.commit()
        logging.info(f"Result '{result}' saved to database with amount {amount} and consecutive_losses {consecutive_losses}.")
    except Exception as e:
        logging.error(f"Error saving to database: {e}")
    finally:
        conn.close()

def fetch_all_results():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error fetching all results: {e}")
        return pd.DataFrame()

def validate_database_contents():
    df = fetch_all_results()
    logging.info(f"Database contents:\n{df}")

def calculate_win_rates(results):
    total_games = len(results)
    total_wins = sum(result[0] for result in results)

    win_rate_total = total_wins / total_games if total_games > 0 else 0

    recent_50 = results[:50]
    recent_20 = results[:20]

    win_rate_50 = sum(result[0] for result in recent_50) / len(recent_50) if recent_50 else 0
    win_rate_20 = sum(result[0] for result in recent_20) / len(recent_20) if recent_20 else 0

    return win_rate_total, win_rate_50, win_rate_20
