# fra_parser.py
import os
import pandas as pd
import numpy as np

def safe_read_csv(file_path):
    """Try a few read strategies to handle vendor CSV formats."""
    try:
        df = pd.read_csv(file_path, comment='#', engine='python')
        if df.shape[1] <= 1:
            # try flexible separator
            df = pd.read_csv(file_path, sep=r'[,\t; ]+', engine='python', comment='#')
    except Exception:
        df = pd.DataFrame()
        for enc in ['utf-8', 'utf-16', 'latin1', 'ISO-8859-1']:
            try:
                df = pd.read_csv(file_path, sep=r'[,\t; ]+', engine='python', comment='#', encoding=enc)
                if df.shape[1] > 1:
                    break
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            raise ValueError(f"Unable to read file properly: {file_path}")
    return df

def read_any_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in ['.csv', '.txt']:
        return safe_read_csv(file_path)
    elif ext in ['.xls','xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def _normalize_column_names(cols):
    # build a mapping for common vendor column names
    rename_map = {
        'frequency (hz)':'Frequency','freq(Hz)':'Frequency','freq hz':'Frequency','hz':'Frequency','frequency':'Frequency',
        'mag(dB)':'Magnitude_dB','magnitude (db)':'Magnitude_dB','ratio (db)':'Magnitude_dB','transfer function (db)':'Magnitude_dB','mag_db':'Magnitude_dB',
        'magnitude_db':'Magnitude_dB','magnitude (dB)':'Magnitude_dB',
        'phase (deg)':'Phase_deg','phase(deg)':'Phase_deg','phase angle':'Phase_deg','phase_deg':'Phase_deg',
        'phase_deg.':'Phase_deg'
    }
    new = []
    for c in cols:
        key = str(c).strip().lower()
        new.append(rename_map.get(key, c))
    return new

def preprocess_fra_file(file_path, output_folder="standardized_data", freq_min=1e2, freq_max=1e6, n_points=500):
    """
    Read a FRA file, clean it, resample to a log-frequency grid and save standardized CSV.
    Returns path to standardized file or None on failure.
    """
    print(f"🔹 Processing: {os.path.basename(file_path)}")
    try:
        df_raw = read_any_file(file_path)
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return None

    # drop empty columns and unnamed extras
    df_raw = df_raw.loc[:, ~df_raw.columns.astype(str).str.contains('^Unnamed', case=False, na=False)]
    df_raw.dropna(how='all', inplace=True)
    df_raw.columns = [str(c).strip().replace('\t','').replace('\n','') for c in df_raw.columns]
    df_raw.columns = _normalize_column_names(df_raw.columns)

    # Keep only relevant columns
    keep_cols = [c for c in ['Frequency','Magnitude_dB','Phase_deg'] if c in df_raw.columns]
    if 'Frequency' not in keep_cols or 'Magnitude_dB' not in keep_cols:
        print(f"⚠️ Missing required columns in {file_path}. Found: {list(df_raw.columns)}. Skipped.")
        return None

    df = df_raw[keep_cols].copy()
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['Frequency','Magnitude_dB'], inplace=True)
    df = df[df['Frequency']>0].drop_duplicates(subset=['Frequency']).sort_values(by='Frequency')

    # Interpolate on a log-spaced frequency axis (common in FRA)
    freq_common = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
    mag_interp = np.interp(freq_common, df['Frequency'], df['Magnitude_dB'])
    if 'Phase_deg' in df.columns:
        # unwrap phase before interpolation to avoid 360-degree jumps
        phase = np.unwrap(np.deg2rad(df['Phase_deg'].values))
        phase_interp = np.rad2deg(np.interp(np.log(freq_common), np.log(df['Frequency']), phase))
    else:
        phase_interp = np.zeros_like(freq_common)

    df_std = pd.DataFrame({'Frequency':freq_common,'Magnitude_dB':mag_interp,'Phase_deg':phase_interp})
    os.makedirs(output_folder, exist_ok=True)
    base = os.path.basename(file_path).split('.')[0]
    output_path = os.path.join(output_folder, base + "_std.csv")
    df_std.to_csv(output_path, index=False)
    print(f"✅ Saved standardized file → {output_path}")
    return output_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "standardized_data")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "raw_vendor_data")

if __name__=="__main__":
    # example usage if run directly
    input_folder = DEFAULT_OUTPUT_PATH
    std_folder = DEFAULT_DATA_PATH
    os.makedirs(std_folder,exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith(('.csv','.txt','.xlsx','.xls')):
            preprocess_fra_file(os.path.join(input_folder,file), std_folder)
