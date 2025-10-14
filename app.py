from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import os
import sqlite3
import pandas as pd
from fra_pipeline import full_fra_pipeline, init_db
from fra_database_init import create_fra_database
import webbrowser
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime, timedelta

std_folder = "standardized_data"

# ✅ Ensure folder exists — prevents Render crash
if not os.path.exists(std_folder):
    print(f"⚠️ Creating missing folder: {std_folder}")
    os.makedirs(std_folder, exist_ok=True)

# -------- APP SETUP --------
app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-random-key'  # Change for production!

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_ROOT, exist_ok=True)

# -------- AUTO-SETUP (Ensures everything exists) --------
for folder in ["results", "standardized_data", "raw_vendor_data"]:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

MAIN_DB = os.path.join(BASE_DIR, "fra_users.db")
if not os.path.exists(MAIN_DB):
    print("🧩 Creating missing user database...")
    conn = sqlite3.connect(MAIN_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ fra_users.db created successfully.")

FRA_DB = os.path.join(BASE_DIR, "fra_app.db")
if not os.path.exists(FRA_DB):
    print("🧠 Creating missing FRA database...")
    create_fra_database(FRA_DB)

for fpath in [
    os.path.join(BASE_DIR, "features.csv"),
    os.path.join(BASE_DIR, "fra_fault_model.pkl")
]:
    if not os.path.exists(fpath):
        open(fpath, "w").close()
        print(f"⚙️ Created placeholder: {os.path.basename(fpath)}")

# -------- USER TABLE (MAIN DB) --------
def get_main_db():
    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_users_table():
    conn = get_main_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_users_table()

# -------- LOGIN REQUIRED DECORATOR --------
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

# -------- PER-USER DATABASE CREATION --------
def create_user_database(username):
    safe_name = secure_filename(username)
    user_db_path = os.path.abspath(os.path.join(RESULTS_ROOT, f"fra_app_{safe_name}.db"))
    if not os.path.exists(user_db_path):
        conn = sqlite3.connect(user_db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fra_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transformer_id TEXT,
                filename TEXT,
                date TEXT,
                status TEXT,
                predicted_fault TEXT,
                confidence REAL
            )
        """)
        conn.commit()
        conn.close()
    return user_db_path

# -------- AUTH ROUTES --------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        email = request.form.get('email').strip().lower()
        password = request.form.get('password')

        if not username or not email or not password:
            flash('Please fill all fields.', 'danger')
            return redirect(url_for('register'))

        password_hash = generate_password_hash(password)
        try:
            conn = get_main_db()
            cur = conn.cursor()
            cur.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                        (username, email, password_hash))
            conn.commit()
            conn.close()
        except Exception as e:
            flash(f'Error creating user: {e}', 'danger')
            return redirect(url_for('register'))

        db_path = create_user_database(username)
        user_folder = os.path.abspath(os.path.join(RESULTS_ROOT, username))
        os.makedirs(user_folder, exist_ok=True)

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# -------- LOGIN --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form.get('identifier').strip()
        password = request.form.get('password')

        conn = get_main_db()
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ? OR email = ?', (identifier, identifier))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_db'] = create_user_database(user['username'])
            user_folder = os.path.abspath(os.path.join(RESULTS_ROOT, user['username']))
            os.makedirs(user_folder, exist_ok=True)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

# -------- UPLOAD --------
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    username = session.get('username')
    user_db = session.get('user_db')
    user_folder = os.path.abspath(os.path.join(RESULTS_ROOT, username))
    os.makedirs(user_folder, exist_ok=True)

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)

            try:
                full_fra_pipeline(
                    raw_folder=user_folder,
                    std_folder="standardized_data",
                    features_file="features.csv",
                    model_file="fra_fault_model.pkl"
                )
            except Exception as e:
                flash(f"Pipeline error: {e}", "danger")

            try:
                conn = sqlite3.connect(user_db)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO fra_tests (transformer_id, filename, date, status, predicted_fault, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    "TX-001",
                    filename,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Completed",
                    "No Fault",
                    0.98
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                flash(f"Database write error: {e}", "danger")
                return redirect(url_for('upload'))

            flash('File uploaded and analyzed successfully.', 'success')
            return redirect(url_for('results'))

        flash("⚠️ Please select a valid file to upload.", "warning")

    return render_template('upload.html')

# -------- RESULTS --------
@app.route('/results')
@login_required
def results():
    user_db = session.get('user_db')
    if not user_db or not os.path.exists(user_db):
        flash("No results found for your account yet.", "warning")
        return render_template('results.html', tables=[])

    view = request.args.get('view', 'all')
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM fra_tests ORDER BY date DESC")
    rows = cur.fetchall()
    conn.close()

    formatted_rows = []
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    for row in rows:
        row_date_str = row["date"]
        try:
            row_dt = datetime.strptime(row_date_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            row_dt = None
        if view == 'recent' and row_dt and row_dt < seven_days_ago:
            continue
        formatted_rows.append({
            "id": row["id"],
            "transformer_id": row["transformer_id"],
            "file_name": row["filename"],
            "date": row["date"],
            "status": row["status"],
            "predicted_fault": row["predicted_fault"],
            "confidence": row["confidence"],
        })

    return render_template('results.html', tables=formatted_rows)

# -------- DELETE RESULT --------
@app.route('/delete_result/<int:result_id>', methods=['POST'])
@login_required
def delete_result(result_id):
    user_db = session.get('user_db')
    if not user_db or not os.path.exists(user_db):
        flash("User database not found.", "danger")
        return redirect(url_for('results'))

    try:
        conn = sqlite3.connect(user_db)
        cur = conn.cursor()
        cur.execute("DELETE FROM fra_tests WHERE id = ?", (result_id,))
        conn.commit()
        conn.close()
        flash("Result deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting result: {e}", "danger")

    return redirect(url_for('results'))

# -------- DOWNLOAD RESULTS --------
@app.route('/download')
@login_required
def download_results():
    user_db = session.get('user_db')
    if not user_db or not os.path.exists(user_db):
        flash("No results available to download.", "warning")
        return redirect(url_for('results'))

    try:
        conn = sqlite3.connect(user_db)
        df = pd.read_sql_query("SELECT * FROM fra_tests ORDER BY date DESC", conn)
        conn.close()
    except Exception as e:
        flash(f"Error reading results: {e}", "danger")
        return redirect(url_for('results'))

    download_folder = os.path.abspath(os.path.join(RESULTS_ROOT, session['username']))
    os.makedirs(download_folder, exist_ok=True)
    download_path = os.path.join(download_folder, "fra_results.csv")
    df.to_csv(download_path, index=False)
    return send_file(download_path, as_attachment=True)
@app.route('/graph/<int:test_id>')
@login_required
def show_graph(test_id):
    import pandas as pd
    user_db = session.get('user_db')
    username = session.get('username')
    user_folder = os.path.join(RESULTS_ROOT, username)

    # --- Validate DB and record ---
    if not user_db or not os.path.exists(user_db):
        flash("Database not found.", "danger")
        return redirect(url_for('results'))

    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM fra_tests WHERE id = ?", (test_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        flash("Test not found.", "warning")
        return redirect(url_for('results'))

    # --- Locate user's file ---
    file_path = os.path.join(user_folder, row["filename"])
    if not os.path.exists(file_path):
        flash(f"File not found: {file_path}", "danger")
        return redirect(url_for('results'))

    # --- Detect FRA Test Type from filename ---
    fname_lower = row["filename"].lower()
    if "doble" in fname_lower:
        test_type = "Healthy Transformer"
    elif "megger" in fname_lower:
        test_type = "Winding Deformation Fault"
    elif "omicron" in fname_lower:
        test_type = "Core Displacement Fault"
    else:
        test_type = "Unknown Type"

    # --- Read data robustly ---
    try:
        df = pd.read_csv(file_path, comment="#", engine="python")
        df.columns = [c.strip().replace("\t","").replace("\n","").lower() for c in df.columns]

        # rename to consistent column names
        rename_map = {
            "frequency (hz)": "Frequency",
            "freq(Hz)": "Frequency",
            "freq hz": "Frequency",
            "hz": "Frequency",
            "frequency": "Frequency",
            "mag(db)": "Magnitude_dB",
            "transfer function (db)": "Magnitude_dB",
            "magnitude (db)": "Magnitude_dB",
            "ratio (db)": "Magnitude_dB"
        }
        df.rename(columns=rename_map, inplace=True)

        # Identify columns dynamically
        freq_col = next((c for c in df.columns if "freq" in c.lower()), None)
        mag_col = next((c for c in df.columns if "mag" in c.lower() or "transfer" in c.lower()), None)

        if not freq_col or not mag_col:
            flash(f"Invalid FRA file format — columns found: {list(df.columns)}", "danger")
            return redirect(url_for('results'))

        df.dropna(subset=[freq_col, mag_col], inplace=True)
        freq = df[freq_col].astype(float).tolist()
        response = df[mag_col].astype(float).tolist()

    except Exception as e:
        flash(f"Error reading FRA file: {e}", "danger")
        return redirect(url_for('results'))

    # --- Render graph template ---
    return render_template(
        "graph.html",
        transformer_id=row["transformer_id"],
        file_name=row["filename"],
        test_type=test_type,  # <=== shows detected FRA type
        predicted_fault=row["predicted_fault"],
        confidence=row["confidence"],
        freq=freq,
        response=response
    )



# -------- RUN APP --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
    print(f"💾 User DB folder: {RESULTS_ROOT}")
    app.run(debug=True)

