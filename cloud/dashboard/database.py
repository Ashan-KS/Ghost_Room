import sqlite3
import os
from datetime import datetime

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "bookings.db")

def init_db():
    """Initializes the SQLite database with the bookings table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            booked_by TEXT NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS calibration_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at DATETIME NOT NULL,
            completed_at DATETIME NOT NULL,
            duration_s INTEGER NOT NULL,
            samples_collected INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'success'
        )
    ''')
    conn.commit()
    conn.close()

def add_booking(title, booked_by, start_time, end_time):
    """Inserts a new booking into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO bookings (title, booked_by, start_time, end_time) VALUES (?, ?, ?, ?)",
        (title, booked_by, start_time, end_time)
    )
    conn.commit()
    conn.close()

def get_bookings():
    """Retrieves all bookings ordered by start time."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, booked_by, start_time, end_time FROM bookings ORDER BY start_time ASC")
    rows = c.fetchall()
    conn.close()
    
    # Return list of dictionaries, suitable for Pandas DataFrame conversion
    bookings = []
    for row in rows:
        bookings.append({
            "id": row[0],
            "title": row[1],
            "booked_by": row[2],
            "start_time": row[3],
            "end_time": row[4]
        })
    return bookings

def delete_booking(booking_id):
    """Deletes a single booking by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM bookings WHERE id = ?", (booking_id,))
    conn.commit()
    conn.close()

def add_calibration_run(started_at, completed_at, duration_s, samples_collected, status="success"):
    """Logs a completed calibration run to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO calibration_runs (started_at, completed_at, duration_s, samples_collected, status) VALUES (?, ?, ?, ?, ?)",
        (started_at, completed_at, duration_s, samples_collected, status)
    )
    conn.commit()
    conn.close()

def get_calibration_runs():
    """Retrieves all calibration runs, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, started_at, completed_at, duration_s, samples_collected, status FROM calibration_runs ORDER BY completed_at DESC")
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "started_at": r[1],
            "completed_at": r[2],
            "duration_s": r[3],
            "samples_collected": r[4],
            "status": r[5],
        }
        for r in rows
    ]
