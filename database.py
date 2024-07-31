import sqlite3
import pandas as pd
from datetime import datetime
import logging

def create_database():
    try:
        conn = sqlite3.connect('coinflips.db')
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

def fetch_recent_results():
    try:
        conn = sqlite3.connect('coinflips.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT result, amount, consecutive_losses_or_wins FROM Coinflips
            ORDER BY id DESC LIMIT 1000
        ''')
        results = cursor.fetchall()
        return results
    except Exception as e:
        logging.error(f"Error fetching recent results: {e}")
        return []
    finally:
        conn.close()

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
        logging.info("Result saved to database.")
    except Exception as e:
        logging.error(f"Error saving to database: {e}")
    finally:
        conn.close()

def fetch_all_results():
    conn = sqlite3.connect('coinflips.db')
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    conn.close()
    return df
