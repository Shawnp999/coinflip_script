# database.py

import sqlite3
import pandas as pd
from datetime import datetime

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
        print("Database and table created successfully.")
    except Exception as e:
        print(f"Error creating database or table: {e}")
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
        print("Result saved to database.")
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        conn.close()

def fetch_all_results():
    conn = sqlite3.connect('coinflips.db')
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    conn.close()
    return df

def get_result_count():
    try:
        conn = sqlite3.connect('coinflips.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Coinflips")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        print(f"Error getting result count: {e}")
        return 0
    finally:
        conn.close()

def fetch_latest_result():
    try:
        conn = sqlite3.connect('coinflips.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT result, amount, datetime, consecutive_losses_or_wins 
            FROM Coinflips 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        if result:
            return {
                'result': bool(result[0]),
                'amount': result[1],
                'datetime': result[2],
                'consecutive_losses_or_wins': result[3]
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching latest result: {e}")
        return None
    finally:
        conn.close()
