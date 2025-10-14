# fra_pipeline.py
import os
from fra_parser import preprocess_fra_file
from fra_feature_extractor import generate_feature_dataset
from fra_classifier_train import train_fault_classifier
import sqlite3
import joblib
import pandas as pd
import sqlite3


def init_db():
    conn = sqlite3.connect('fra_app.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS fra_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            filename TEXT,
            result TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "standardized_data")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "raw_vendor_data")
DEFAULT_FEATURE_PATH = os.path.join(BASE_DIR, "features.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "fra_fault_model.pkl")

def full_fra_pipeline(
    raw_folder= DEFAULT_OUTPUT_PATH,
    std_folder= DEFAULT_DATA_PATH,
    features_file=DEFAULT_FEATURE_PATH,
    model_file=DEFAULT_MODEL_PATH

):
    print("\n🚀 FRA Full Pipeline Started")
    print(f"📂 Raw data folder: {raw_folder}")
    print(f"📂 Standardized folder: {std_folder}\n")

    # ---------------------------
    # Step 1: Preprocessing
    # ---------------------------
    print("=== Step 1: Preprocessing raw FRA data ===")

# --- Safety: recreate folders if missing ---
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(std_folder, exist_ok=True)

# Ensure key files exist even if deleted
    if not os.path.exists(features_file):
        open(features_file, 'w').close()
        print(f"⚙️ Created placeholder: {features_file}")

    if not os.path.exists(model_file):
        open(model_file, 'w').close()
        print(f"⚙️ Created placeholder: {model_file}")


    print("✅ Standardization complete.\n")

    # ---------------------------
    # Step 2: Feature Extraction
    # ---------------------------
    print("=== Step 2: Extracting features ===")
    try:
        generate_feature_dataset(std_folder, features_file)
        if not os.path.exists(features_file):
            print("❌ Feature extraction failed — no features.csv created.")
            return
        else:
            print(f"📄 Feature file created: {features_file}")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    # ---------------------------
    # Step 3: Model Training
    # ---------------------------
    print("\n=== Step 3: Training fault classifier ===")
    try:
        train_fault_classifier(features_file, model_file)
        # ==========================================
        # 🔽 Step 4: Save model predictions to database
        # ==========================================


        def save_prediction_to_db(db_path, transformer_id, file_name, predicted_fault, confidence):
            """Insert prediction results into the fra_tests table."""
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO fra_tests (transformer_id, file_name, date, status, predicted_fault, confidence)
                VALUES (?, ?, datetime('now'), ?, ?, ?)
                """, (transformer_id, file_name, "Classified", predicted_fault, confidence))
            conn.commit()
            conn.close()
            print(f"✅ Saved prediction result for {file_name} → {predicted_fault} ({confidence*100:.1f}%)")
            print("✅ Saving to:", db_path)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            # --- Use the model to predict and save results ---
        DB_PATH = os.path.join(BASE_DIR, "fra_user.db")
        init_db()
        model = joblib.load(model_file)
        df = pd.read_csv(features_file)
        X = df.drop(columns=['File', 'Label'], errors='ignore')
        file_names = df['File'].tolist()
        preds = model.predict(X)
        confs = model.predict_proba(X).max(axis=1)

        for file, pred, conf in zip(file_names, preds, confs):
            save_prediction_to_db(DB_PATH, 1, file, pred, conf)
        # ==========================================

    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return

    # ---------------------------
    # Done
    # ---------------------------
    print("\n✅ All steps completed successfully!")
    print(f"📁 Standardized files saved to: {std_folder}")
    print(f"📄 Features file: {features_file}")
    print(f"🧠 Model saved: {model_file}")

if __name__ == "__main__":
    full_fra_pipeline()


def save_prediction_to_db(db_path, transformer_id, file_name, predicted_fault, confidence):
    """Insert prediction results into the fra_tests table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fra_tests (transformer_id, file_name, date, status, predicted_fault, confidence)
        VALUES (?, ?, datetime('now'), ?, ?, ?)
    """, (transformer_id, file_name, "Classified", predicted_fault, confidence))
    conn.commit()
    conn.close()
    print(f"✅ Saved prediction result for {file_name} → {predicted_fault} ({confidence*100:.1f}%)")
