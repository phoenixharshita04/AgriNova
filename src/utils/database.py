import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = 'agrinova_logs.db'

def init_db():
    """
    Initializes the SQLite database and creates the farm_logs table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farm_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            region TEXT,
            crop TEXT,
            predicted_yield REAL,
            disease_status TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_log(region, crop, predicted_yield, disease_status):
    """
    Inserts a new scan log into the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO farm_logs (timestamp, region, crop, predicted_yield, disease_status)
        VALUES (?, ?, ?, ?, ?)
    ''', (now, region, crop, predicted_yield, disease_status))
    
    conn.commit()
    conn.close()

def get_all_logs():
    """
    Retrieves all farm logs from the database and returns them as a pandas DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    # Read directly into a DataFrame for easy Streamlit rendering
    df = pd.read_sql_query("SELECT * FROM farm_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df
