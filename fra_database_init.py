# fra_database_init.py
import sqlite3
import os


def create_fra_database(db_path="fra_app.db"):
    if os.path.exists(db_path):
        print(f"⚠️ Database already exists at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tables
    cur.executescript("""
    CREATE TABLE transformers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        manufacturer TEXT,
        rating_kva REAL,
        location TEXT
    );

    CREATE TABLE fra_tests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transformer_id INTEGER,
        file_name TEXT,
        date TEXT,
        status TEXT,
        predicted_fault TEXT,
        confidence REAL,
        notes TEXT,
        FOREIGN KEY (transformer_id) REFERENCES transformers(id)
    );

    CREATE TABLE fra_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fra_test_id INTEGER,
        feature_name TEXT,
        feature_value REAL,
        FOREIGN KEY (fra_test_id) REFERENCES fra_tests(id)
    );

    CREATE TABLE models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        train_date TEXT,
        accuracy REAL,
        notes TEXT
    );
    """)

    conn.commit()
    conn.close()
    print(f"✅ FRA database created successfully → {db_path}")


if __name__ == "__main__":
    create_fra_database()

