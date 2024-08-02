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

def fetch_recent_results(limit=1000):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT result, amount, consecutive_losses_or_wins FROM Coinflips
            ORDER BY id DESC LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
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
        logging.info("Result saved to database.")
    except Exception as e:
        logging.error(f"Error saving to database: {e}")
    finally:
        conn.close()

def fetch_all_results():
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    conn.close()
    return df
