import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_fault_classifier(file_path,model_file):
    # Load features
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ features.csv not found! Run fra_feature_extraction.py first.")
    
    df = pd.read_csv(file_path)
    print(f"📊 Loaded {len(df)} samples from {file_path}")

    # Check for Label column
    if 'Label' not in df.columns:
        raise ValueError("❌ Missing 'Label' column. Add transformer fault type labels to features.csv.")
    
    # Drop rows where Label is missing
    df = df.dropna(subset=['Label'])
    if df.empty:
        raise ValueError("❌ No labeled samples found. Please add some labels (e.g. 'Core Shift', 'Shorted Turns').")

    # Select features (excluding File and Label)
    X = df.drop(columns=['File', 'Label'], errors='ignore')
    y = df['Label']

    if len(df) < 2:
        raise ValueError("❌ Not enough data to split. Add at least 2 labeled samples for training.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n✅ Training complete!")
    print(f"📈 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_file)
    print("\n💾 Model saved as 'fra_fault_model.pkl'")

if __name__ == "__main__":
    train_fault_classifier()