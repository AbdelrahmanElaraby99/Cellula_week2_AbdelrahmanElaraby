import sqlite3

# --- Connect to DB ---
conn = sqlite3.connect("helping_files/website.db")
cursor = conn.cursor()

# --- Create tables ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    password TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS toxic_classification (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    text_query TEXT,
    image_caption TEXT,
    image_src BLOB,
    class TEXT,
    confidence_score REAL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
)
""")

conn.commit()

conn.close()
