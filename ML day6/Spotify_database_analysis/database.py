import sqlite3

def get_connection():
    return sqlite3.connect("spotify_predictions.db", check_same_thread=False)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        danceability REAL,
        energy REAL,
        valence REAL,
        tempo REAL,
        duration_ms REAL,
        popularity REAL,
        cluster INTEGER,
        cluster_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()