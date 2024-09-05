# model.py

import os
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib

from database import fetch_all_results, DATABASE_NAME

MODEL_PATH = 'coinflip_model.h5'
SCALER_PATH = 'scaler.pkl'
BACKUP_PATH = 'coinflips_backup.csv'


def backup_database():
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT * FROM Coinflips", conn)
    df.to_csv(BACKUP_PATH, index=False)
    conn.close()
    logging.info(f"Database backed up to {BACKUP_PATH}")
