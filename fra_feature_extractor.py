# fra_feature_extractor.py
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# --- Auto path detection ---
# Get the absolute directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Automatically locate the 'standardized_data' folder inside the project
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "standardized_data")

# Automatically save the features.csv file in the same folder as this script
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "features.csv")

# Check if the data folder exists
if not os.path.exists(DEFAULT_DATA_PATH):
    print(f"⚠️ Data folder not found at: {DEFAULT_DATA_PATH}")
    print("Please make sure your standardized_data folder is in the same directory as this script.")


# Optional label mapping (keeps previous convention)
FAULT_LABELS = {
    "omicron_test_std.csv": "Core_Displacement",
    "megger_sample_std.csv": "Winding_Deformation",
    "doble_test_std.csv": "Healthy"
}

def _db_to_linear(mag_db):
    return 10 ** (mag_db / 20.0)

def _log_area(freq, mag_db):
    mag_lin = _db_to_linear(mag_db)
    x = np.log(freq)
    return np.trapezoid(mag_lin, x)

def _spectral_centroid(freq, mag_db):
    mag_lin = _db_to_linear(mag_db)
    return np.sum(freq * mag_lin) / np.sum(mag_lin) if np.sum(mag_lin) > 0 else 0

def _slope_loglog(freq, mag_db, mask):
    if np.sum(mask) < 2:
        return 0.0
    x = np.log10(freq[mask]).reshape(-1, 1)
    y = mag_db[mask].reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    return float(reg.coef_[0][0])

def extract_features(file_path, label="Unknown", plot=False):
    df = pd.read_csv(file_path)
    freq = df['Frequency'].values
    mag_db = df['Magnitude_dB'].values
    phase = df['Phase_deg'].values if 'Phase_deg' in df.columns else np.zeros_like(mag_db)

    feats = {
        'Mean_Mag_dB': np.mean(mag_db),
        'Std_Mag_dB': np.std(mag_db),
        'Skew_Mag_dB': skew(mag_db),
        'Kurtosis_Mag_dB': kurtosis(mag_db, fisher=True),
        'RMS_Mag_lin': np.sqrt(np.mean(_db_to_linear(mag_db)**2)),
        'Spectral_Centroid': _spectral_centroid(freq, mag_db),
        'LogArea': _log_area(freq, mag_db)
    }

    # Peak detection
    mag_lin = _db_to_linear(mag_db)
    peaks, props = find_peaks(mag_lin, height=np.max(mag_lin)*0.05, prominence=0.1, distance=5)
    feats['Num_Peaks'] = len(peaks)
    feats['Max_Peak_Prominence'] = float(np.max(props['prominences'])) if 'prominences' in props and len(props['prominences']) > 0 else 0.0

    # Peak-based features (first 2 resonances)
    for i, p in enumerate(peaks[:2]):
        try:
            widths = peak_widths(mag_lin, [p], rel_height=0.707)
            left, right = int(widths[2][0]), int(widths[3][0])
            bw = freq[right] - freq[left]
            q = freq[p] / bw if bw > 0 else np.nan
        except Exception:
            bw, q = np.nan, np.nan
        feats[f'Peak{i+1}_Freq'] = float(freq[p])
        feats[f'Peak{i+1}_BW'] = float(bw)
        feats[f'Peak{i+1}_Q'] = float(q)

    # Frequency band slopes
    for name, mask in {
        'Low': (freq >= 1e2) & (freq < 1e3),
        'Mid': (freq >= 1e3) & (freq < 1e5),
        'High': (freq >= 1e5) & (freq <= 1e6)
    }.items():
        feats[f'Slope_{name}_dB_per_log10Hz'] = _slope_loglog(freq, mag_db, mask)

    # Phase stats
    phase_unwrapped = np.unwrap(np.deg2rad(phase))
    feats['Phase_Mean_deg'] = float(np.mean(np.rad2deg(phase_unwrapped)))
    feats['Phase_Std_deg'] = float(np.std(np.rad2deg(phase_unwrapped)))

    feats['File'] = os.path.basename(file_path)
    feats['Label'] = label

    # Show graph only when explicitly requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.semilogx(freq, mag_db, lw=2, label=label)
        plt.scatter(freq[peaks], mag_db[peaks], color='red', s=40, label='Resonances')
        plt.title(f"Frequency Response - {label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True, which='both', ls='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return feats


    # Peak detection
    mag_lin = _db_to_linear(mag_db)
    peaks, props = find_peaks(mag_lin, height=np.max(mag_lin)*0.05, prominence=0.1, distance=5)
    feats['Num_Peaks'] = len(peaks)
    if 'prominences' in props and len(props['prominences']) > 0:
        feats['Max_Peak_Prominence'] = float(np.max(props['prominences']))
    else:
        feats['Max_Peak_Prominence'] = 0.0

    peaks_sorted = peaks[np.argsort(mag_lin[peaks])[-2:]] if len(peaks) >= 2 else peaks
    for i, p in enumerate(peaks_sorted[:2]):
        bw, q = np.nan, np.nan
        try:
            widths = peak_widths(mag_lin, [p], rel_height=0.707)
            left, right = int(widths[2][0]), int(widths[3][0])
            bw = freq[right] - freq[left]
            q = freq[p] / bw if bw > 0 else np.nan
        except Exception:
            pass
        feats[f'Peak{i+1}_Freq'] = float(freq[p])
        feats[f'Peak{i+1}_BW'] = float(bw)
        feats[f'Peak{i+1}_Q'] = float(q)

    # Slope calculations
    bands = {
        'Low': (freq >= 1e2) & (freq < 1e3),
        'Mid': (freq >= 1e3) & (freq < 1e5),
        'High': (freq >= 1e5) & (freq <= 1e6)
    }
    for bname, mask in bands.items():
        feats[f'Slope_{bname}_dB_per_log10Hz'] = _slope_loglog(freq, mag_db, mask)

    # Phase stats
    phase_unwrapped = np.unwrap(np.deg2rad(phase))
    feats['Phase_Mean_deg'] = float(np.mean(np.rad2deg(phase_unwrapped)))
    feats['Phase_Std_deg'] = float(np.std(np.rad2deg(phase_unwrapped)))

    feats['File'] = os.path.basename(file_path)
    feats['Label'] = label

    # Plot FRA curve with peaks
    if plot:
        plt.figure(figsize=(8, 5))
        plt.semilogx(freq, mag_db, label=f'{label} - {os.path.basename(file_path)}', lw=2)
        plt.scatter(freq[peaks], mag_db[peaks], color='red', s=40, label='Resonance Peaks')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title(f'Frequency Response - {label}')
        plt.grid(True, which='both', ls='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return feats

# --- Auto Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "standardized_data")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "features.csv")

def generate_feature_dataset(std_folder=DEFAULT_DATA_PATH, output_file=DEFAULT_OUTPUT_PATH):
    all_feats = []
    for file in os.listdir(std_folder):
        if file.endswith(".csv"):
            label = FAULT_LABELS.get(file, "Unknown")
            feats = extract_features(os.path.join(std_folder, file), label, plot=True)
            all_feats.append(feats)
    df_feats = pd.DataFrame(all_feats)
    cols = [c for c in df_feats.columns if c not in ('File', 'Label')] + ['File', 'Label']
    df_feats = df_feats[cols]
    df_feats.to_csv(output_file, index=False)
    print(f"✅ Features with labels saved → {output_file}")
    return output_file

def compare_faults(std_folder=DEFAULT_DATA_PATH):
    """Compare Healthy vs Faulty transformer FRA curves."""
    files = [f for f in os.listdir(std_folder) if f.endswith('.csv')]
    data = {}
    for f in files:
        label = FAULT_LABELS.get(f, "Unknown")
        df = pd.read_csv(os.path.join(std_folder, f))
        data[label] = df

    plt.figure(figsize=(8, 5))
    for label, df in data.items():
        plt.semilogx(df['Frequency'], df['Magnitude_dB'], lw=2, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FRA Comparison Across Faults")
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

compare_faults()


if __name__ == "__main__":
    generate_feature_dataset()
