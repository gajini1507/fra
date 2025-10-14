# view_fra_results.py
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# --- Database path setup (auto-detect) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fra_app.db")  # same file created by fra_database_init.py

# --- Connect to the database ---
if not os.path.exists(DB_PATH):
    print(f"❌ Database not found at: {DB_PATH}")
else:
    conn = sqlite3.connect(DB_PATH)

    # --- Option 1: Print raw records ---
    print("\n=== FRA Test Records (raw) ===")
    for row in conn.execute("SELECT * FROM fra_tests"):
        print(row)

    # --- Option 2: Pretty table view using pandas ---
    print("\n=== FRA Test Records (DataFrame view) ===")
    df = pd.read_sql_query("SELECT * FROM fra_tests", conn)
    print(df)

    # --- Optional: Show graph on demand ---
    def show_graph(record_id):
        """Plot Frequency vs Magnitude for a specific record."""
        query = f"SELECT Frequency, Magnitude_dB FROM fra_data WHERE test_id={record_id}"
        data = pd.read_sql_query(query, conn)

        if data.empty:
            print(f"No data found for record ID {record_id}")
            return

        plt.figure(figsize=(8,5))
        plt.plot(data['Frequency'], data['Magnitude_dB'], color='blue', linewidth=1.5)
        plt.title(f"FRA Response for Record ID {record_id}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.show()

    # Example usage — only shows graph when manually called
    # Uncomment to test:
    # show_graph(1)

    conn.close()
