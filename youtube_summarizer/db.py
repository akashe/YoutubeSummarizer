import pdb

import sqlite3
from datetime import datetime


def create_db_and_table(db_name="token_usage.db"):
    """Create a SQLite database and table for storing daily tokens processed."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS daily_tokens (
    #         date TEXT PRIMARY KEY,
    #         tokens INTEGER
    #     )
    # ''')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_tokens (
                date TEXT PRIMARY KEY,
                input_tokens INTEGER,
                output_tokens INTEGER
            )
        ''')
    conn.commit()
    conn.close()


def save_or_update_tokens(input_tokens, output_tokens, db_name="token_usage.db", model_name="gpt-4o-mini"):
    """Save or update the number of tokens processed for the current day."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Check if an entry exists for today
    cursor.execute('SELECT * FROM daily_tokens WHERE date = ?', (today,))
    result = cursor.fetchone()

    if result:
        # Update existing entry for today
        input_tokens_till_now = result[1]
        output_tokens_till_now = result[2]
        input_tokens += input_tokens_till_now
        output_tokens += output_tokens_till_now
        cursor.execute('''
            UPDATE daily_tokens
            SET input_tokens = ?,
                output_tokens = ?
            WHERE date = ?
        ''', (input_tokens, output_tokens, today))
    else:
        # Insert new entry for today
        cursor.execute('''
            INSERT INTO daily_tokens (date, input_tokens, output_tokens)
            VALUES (?, ?, ?)
        ''', (today, input_tokens, output_tokens))

    conn.commit()
    conn.close()


def get_today_token_usage(db_name="token_usage.db"):
    """Get the number of tokens processed for the current day."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('SELECT input_tokens, output_tokens FROM daily_tokens WHERE date = ?', (today,))
    data = cursor.fetchone()

    conn.close()

    if data:
        return data[0], data[1]
    else:
        # If there's no entry for today yet, return zeros
        return 0, 0
