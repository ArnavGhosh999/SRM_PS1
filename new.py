import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import time
import seaborn as sns
import os
import warnings
import json
from datetime import datetime
from tqdm import tqdm
import re

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Create directories
os.makedirs("Graph_Images", exist_ok=True)
os.makedirs("Data/csv_results", exist_ok=True)
os.makedirs("Data/json_results", exist_ok=True)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def safe_plot(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            plt.close('all')
            return result
        except Exception as e:
            print(f"Error in plotting function {func.__name__}: {e}")
            plt.close('all')
            return None
    return wrapper

def load_and_format_20hz_data(dataset_path):
    """Load and properly format the 20Hz dataset with comma-separated values"""
    print(f"Loading and formatting 20Hz data from: {dataset_path}")
    
    # Load the raw CSV
    df_raw = pd.read_csv(dataset_path)
    print(f"Raw dataset shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")
    
    # Initialize lists to store formatted data
    formatted_data = []
    
    print(f"\nProcessing {len(df_raw)} records...")
    
    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Formatting data"):
        try:
            # Extract comma-separated values and convert to lists
            a_current_str = str(row['A Current']).strip('"')
            a_voltage_str = str(row['A Voltage']).strip('"')
            b_current_str = str(row['B Current']).strip('"')
            b_voltage_str = str(row['B Voltage']).strip('"')
            
            # Split by commas and convert to float arrays
            a_current_values = [float(x.strip()) for x in a_current_str.split(',') if x.strip()]
            a_voltage_values = [float(x.strip()) for x in a_voltage_str.split(',') if x.strip()]
            b_current_values = [float(x.strip()) for x in b_current_str.split(',') if x.strip()]
            b_voltage_values = [float(x.strip()) for x in b_voltage_str.split(',') if x.strip()]
            
            # Find the minimum length to ensure all arrays are same size
            min_length = min(len(a_current_values), len(a_voltage_values), 
                           len(b_current_values), len(b_voltage_values))
            
            if min_length > 0:
                # Truncate all arrays to minimum length
                a_current_values = a_current_values[:min_length]
                a_voltage_values = a_voltage_values[:min_length]
                b_current_values = b_current_values[:min_length]
                b_voltage_values = b_voltage_values[:min_length]
                
                # Create time indices for this record
                time_indices = np.arange(min_length)
                
                # Create individual rows for each time point
                for i in range(min_length):
                    formatted_row = {
                        'record_id': idx,
                        'time_point': i,
                        'site_name': row['Site Name'],
                        'machine_name': row['Point Machine Name'],
                        'direction': row['Direction'],
                        'timestamp': row['Time'],
                        'a_current': a_current_values[i],
                        'a_voltage': a_voltage_values[i],
                        'b_current': b_current_values[i],
                        'b_voltage': b_voltage_values[i],
                        'type_a': row['Type of A'],
                        'type_b': row['Type of B'],
                        'polling_a': row['Polling of A'],
                        'polling_b': row['Polling of B'],
                        'source_dataset': row['source_dataset']
                    }
                    formatted_data.append(formatted_row)
                    
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Create formatted DataFrame
    df_formatted = pd.DataFrame(formatted_data)
    
    print(f"\nFormatted dataset:")
    print(f"  - Shape: {df_formatted.shape}")
    print(f"  - Records processed: {df_formatted['record_id'].nunique()}")
    print(f"  - Total time points: {len(df_formatted)}")
    print(f"  - Average points per record: {len(df_formatted) / df_formatted['record_id'].nunique():.1f}")
    
    # Save formatted data
    df_formatted.to_csv(f"Data/csv_results/formatted_20hz_data_{current_time}.csv", index=False)
    print(f"  - Saved: Data/csv_results/formatted_20hz_data_{current_time}.csv")
    
    return df_formatted

def analyze_signal_characteristics(df):
    """Analyze the characteristics of the formatted signals"""
    print("\nAnalyzing signal characteristics...")
    
    # Basic statistics
    stats = {
        'a_current': {
            'mean': float(df['a_current'].mean()),
            'std': float(df['a_current'].std()),
            'min': float(df['a_current'].min()),
            'max': float(df['a_current'].max()),
            'unique_values': int(df['a_current'].nunique())
        },
        'a_voltage': {
            'mean': float(df['a_voltage'].mean()),
            'std': float(df['a_voltage'].std()),
            'min': float(df['a_voltage'].min()),
            'max': float(df['a_voltage'].max()),
            'unique_values': int(df['a_voltage'].nunique())
        },
        'b_current': {
            'mean': float(df['b_current'].mean()),
            'std': float(df['b_current'].std()),
            'min': float(df['b_current'].min()),
            'max': float(df['b_current'].max()),
            'unique_values': int(df['b_current'].nunique())
        },
        'b_voltage': {
            'mean': float(df['b_voltage'].mean()),
            'std': float(df['b_voltage'].std()),
            'min': float(df['b_voltage'].min()),
            'max': float(df['b_voltage'].max()),
            'unique_values': int(df['b_voltage'].nunique())
        }
    }
    
    for signal, data in stats.items():
        print(f"  {signal}:")
        print(f"    Range: {data['min']:.2f} to {data['max']:.2f}")
        print(f"    Mean: {data['mean']:.2f} ± {data['std']:.2f}")
        print(f"    Unique values: {data['unique_values']}")
    
    # Save statistics
    with open(f"Data/json_results/signal_characteristics_{current_time}.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

@safe_plot
def plot_formatted_time_series(df, max_points=5000):
    """Plot the properly formatted time series data"""
    print(f"\nPlotting formatted time series (showing up to {max_points} points)...")
    
    # Sample data if too large
    if len(df) > max_points:
        df_plot = df.sample(n=max_points, random_state=42).sort_index()
    else:
        df_plot = df.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Properly Formatted 20Hz Signal Data', fontsize=16, fontweight='bold')
    
    # A Current
    axes[0, 0].plot(df_plot.index, df_plot['a_current'], color='red', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title('A Current Signal', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index', fontsize=11)
    axes[0, 0].set_ylabel('Current (A Current)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # A Voltage
    axes[0, 1].plot(df_plot.index, df_plot['a_voltage'], color='blue', alpha=0.7, linewidth=0.8)
    axes[0, 1].set_title('A Voltage Signal', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Sample Index', fontsize=11)
    axes[0, 1].set_ylabel('Voltage (A Voltage)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # B Current
    axes[1, 0].plot(df_plot.index, df_plot['b_current'], color='orange', alpha=0.7, linewidth=0.8)
    axes[1, 0].set_title('B Current Signal', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Sample Index', fontsize=11)
    axes[1, 0].set_ylabel('Current (B Current)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # B Voltage
    axes[1, 1].plot(df_plot.index, df_plot['b_voltage'], color='green', alpha=0.7, linewidth=0.8)
    axes[1, 1].set_title('B Voltage Signal', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Voltage (B Voltage)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Graph_Images/formatted_signals_{current_time}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Graph_Images/formatted_signals_{current_time}.png")

@safe_plot
def plot_voltage_current_relationships(df, max_points=5000):
    """Plot voltage vs current relationships"""
    print(f"\nPlotting voltage-current relationships...")
    
    # Sample data if too large
    if len(df) > max_points:
        df_plot = df.sample(n=max_points, random_state=42)
    else:
        df_plot = df.copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Voltage vs Current Relationships', fontsize=16, fontweight='bold')
    
    # A: Voltage vs Current
    axes[0].scatter(df_plot['a_voltage'], df_plot['a_current'], alpha=0.6, s=1, c='blue')
    axes[0].set_title('A: Voltage vs Current', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('A Voltage', fontsize=11)
    axes[0].set_ylabel('A Current', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # B: Voltage vs Current
    axes[1].scatter(df_plot['b_voltage'], df_plot['b_current'], alpha=0.6, s=1, c='red')
    axes[1].set_title('B: Voltage vs Current', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('B Voltage', fontsize=11)
    axes[1].set_ylabel('B Current', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Graph_Images/voltage_current_relationships_{current_time}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Graph_Images/voltage_current_relationships_{current_time}.png")

def create_enhanced_features(voltage_data, current_data, window_size=20):
    """Create enhanced features for better anomaly detection"""
    n = len(voltage_data)
    
    # Ensure arrays are numpy arrays
    voltage_data = np.array(voltage_data)
    current_data = np.array(current_data)
    
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(voltage_data), np.std(voltage_data), np.median(voltage_data),
        np.percentile(voltage_data, 25), np.percentile(voltage_data, 75),
        np.min(voltage_data), np.max(voltage_data),
        np.mean(current_data), np.std(current_data), np.median(current_data),
        np.percentile(current_data, 25), np.percentile(current_data, 75),
        np.min(current_data), np.max(current_data)
    ])
    
    # Rate of change features
    v_diff = np.diff(voltage_data)
    c_diff = np.diff(current_data)
    features.extend([
        np.mean(v_diff), np.std(v_diff), np.max(np.abs(v_diff)),
        np.mean(c_diff), np.std(c_diff), np.max(np.abs(c_diff))
    ])
    
    # Power and energy features
    power = voltage_data * current_data
    features.extend([
        np.mean(power), np.std(power), np.max(power), np.min(power)
    ])
    
    # Correlation and cross-features
    if len(voltage_data) > 1 and np.std(voltage_data) > 0 and np.std(current_data) > 0:
        correlation = np.corrcoef(voltage_data, current_data)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    features.append(correlation)
    
    # Frequency domain features (simplified)
    v_fft = np.fft.fft(voltage_data)
    c_fft = np.fft.fft(current_data)
    v_magnitude = np.abs(v_fft)
    c_magnitude = np.abs(c_fft)
    
    features.extend([
        np.mean(v_magnitude), np.std(v_magnitude),
        np.mean(c_magnitude), np.std(c_magnitude)
    ])
    
    # Trend features
    v_trend = np.polyfit(range(len(voltage_data)), voltage_data, 1)[0]
    c_trend = np.polyfit(range(len(current_data)), current_data, 1)[0]
    features.extend([v_trend, c_trend])
    
    # Range and variation features
    v_range = np.max(voltage_data) - np.min(voltage_data)
    c_range = np.max(current_data) - np.min(current_data)
    features.extend([v_range, c_range])
    
    # Peak detection features
    if len(voltage_data) > 2:
        v_peaks = len([i for i in range(1, len(voltage_data)-1) 
                      if voltage_data[i] > voltage_data[i-1] and voltage_data[i] > voltage_data[i+1]])
        c_peaks = len([i for i in range(1, len(current_data)-1) 
                      if current_data[i] > current_data[i-1] and current_data[i] > current_data[i+1]])
    else:
        v_peaks = c_peaks = 0
    features.extend([v_peaks, c_peaks])
    
    # Ensure no NaN/inf values
    features = np.array(features, dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def enhanced_isolation_forest(features, contamination=0.05):
    """Enhanced Isolation Forest with better parameters"""
    print(f"Running Enhanced Isolation Forest (contamination={contamination})")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Enhanced Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=300,
        max_samples=min(256, len(features)),
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(features_scaled)
    scores = iso_forest.decision_function(features_scaled)
    
    anomaly_binary = (predictions == -1).astype(int)
    anomaly_count = np.sum(anomaly_binary)
    
    print(f"  Detected {anomaly_count} anomalies ({anomaly_count/len(features)*100:.1f}%)")
    
    return anomaly_binary, scores, iso_forest

def enhanced_xgboost_detection(features, contamination=0.05):
    """Enhanced XGBoost-based anomaly detection"""
    print(f"Running Enhanced XGBoost Detection (contamination={contamination})")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Use Isolation Forest to create training labels
    iso_temp = IsolationForest(contamination=contamination, random_state=42)
    temp_labels = iso_temp.fit_predict(features_scaled)
    temp_labels_binary = (temp_labels == -1).astype(int)
    
    # Enhanced XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_model.fit(features_scaled, temp_labels_binary)
    
    predictions = xgb_model.predict(features_scaled)
    probabilities = xgb_model.predict_proba(features_scaled)[:, 1]
    
    anomaly_count = np.sum(predictions)
    print(f"  Detected {anomaly_count} anomalies ({anomaly_count/len(features)*100:.1f}%)")
    
    return predictions, probabilities, xgb_model

def analyze_signals_with_sliding_window(df, window_size=100, step_size=25, contamination=0.08):
    """Analyze signals using sliding window approach with enhanced features"""
    print(f"\nAnalyzing signals with sliding windows:")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    print(f"  Contamination: {contamination}")
    
    # Prepare data arrays
    a_voltage = df['a_voltage'].values
    a_current = df['a_current'].values
    b_voltage = df['b_voltage'].values
    b_current = df['b_current'].values
    
    pairs = [
        ('A', a_voltage, a_current),
        ('B', b_voltage, b_current)
    ]
    
    all_results = {}
    
    for pair_name, voltage_data, current_data in pairs:
        print(f"\n--- Processing Signal Pair {pair_name} ---")
        
        # Create sliding windows
        windows = []
        window_indices = []
        
        for i in range(0, len(voltage_data) - window_size + 1, step_size):
            v_window = voltage_data[i:i + window_size]
            c_window = current_data[i:i + window_size]
            
            features = create_enhanced_features(v_window, c_window)
            windows.append(features)
            window_indices.append(i + window_size // 2)
        
        windows = np.array(windows)
        window_indices = np.array(window_indices)
        
        print(f"  Created {len(windows)} windows with {windows.shape[1]} features each")
        
        # Enhanced anomaly detection
        iso_anomalies, iso_scores, iso_model = enhanced_isolation_forest(windows, contamination)
        xgb_anomalies, xgb_scores, xgb_model = enhanced_xgboost_detection(windows, contamination)
        
        # Calculate overlaps
        overlap_anomalies = iso_anomalies & xgb_anomalies
        overlap_count = np.sum(overlap_anomalies)
        
        # Store results
        results = {
            'pair_name': pair_name,
            'total_windows': len(windows),
            'iso_anomalies': int(np.sum(iso_anomalies)),
            'xgb_anomalies': int(np.sum(xgb_anomalies)),
            'overlap_anomalies': int(overlap_count),
            'iso_percentage': float(np.sum(iso_anomalies) / len(windows) * 100),
            'xgb_percentage': float(np.sum(xgb_anomalies) / len(windows) * 100),
            'overlap_percentage': float(overlap_count / len(windows) * 100),
            'window_indices': window_indices.tolist(),
            'iso_anomaly_windows': window_indices[iso_anomalies == 1].tolist(),
            'xgb_anomaly_windows': window_indices[xgb_anomalies == 1].tolist(),
            'overlap_anomaly_windows': window_indices[overlap_anomalies == 1].tolist()
        }
        
        all_results[pair_name] = results
        
        print(f"  Results:")
        print(f"    Isolation Forest: {results['iso_anomalies']} ({results['iso_percentage']:.1f}%)")
        print(f"    XGBoost: {results['xgb_anomalies']} ({results['xgb_percentage']:.1f}%)")
        print(f"    Overlap: {results['overlap_anomalies']} ({results['overlap_percentage']:.1f}%)")
        
        # Plot results
        plot_enhanced_anomaly_results(
            voltage_data, current_data, window_indices,
            iso_anomalies, iso_scores, xgb_anomalies, xgb_scores,
            pair_name
        )
    
    return all_results

@safe_plot
def plot_enhanced_anomaly_results(voltage_data, current_data, window_indices,
                                 iso_anomalies, iso_scores, xgb_anomalies, xgb_scores,
                                 pair_name):
    """Plot enhanced anomaly detection results"""
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(f'Enhanced Anomaly Detection - Signal Pair {pair_name}', fontsize=16, fontweight='bold')
    
    # Map window indices back to signal space
    iso_anomaly_indices = window_indices[iso_anomalies == 1]
    xgb_anomaly_indices = window_indices[xgb_anomalies == 1]
    
    # Plot 1: Voltage with IF anomalies
    axes[0, 0].plot(voltage_data, color='blue', alpha=0.7, linewidth=0.8, label='Voltage')
    if len(iso_anomaly_indices) > 0:
        axes[0, 0].scatter(iso_anomaly_indices, voltage_data[iso_anomaly_indices], 
                          color='red', s=15, alpha=0.8, label=f'IF Anomalies ({len(iso_anomaly_indices)})')
    axes[0, 0].set_title(f'Voltage Signal with Isolation Forest Anomalies', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index', fontsize=11)
    axes[0, 0].set_ylabel(f'Voltage (Pair {pair_name})', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Current with IF anomalies
    axes[0, 1].plot(current_data, color='red', alpha=0.7, linewidth=0.8, label='Current')
    if len(iso_anomaly_indices) > 0:
        axes[0, 1].scatter(iso_anomaly_indices, current_data[iso_anomaly_indices], 
                          color='darkred', s=15, alpha=0.8, label=f'IF Anomalies ({len(iso_anomaly_indices)})')
    axes[0, 1].set_title(f'Current Signal with Isolation Forest Anomalies', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Sample Index', fontsize=11)
    axes[0, 1].set_ylabel(f'Current (Pair {pair_name})', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Voltage with XGB anomalies
    axes[1, 0].plot(voltage_data, color='blue', alpha=0.7, linewidth=0.8, label='Voltage')
    if len(xgb_anomaly_indices) > 0:
        axes[1, 0].scatter(xgb_anomaly_indices, voltage_data[xgb_anomaly_indices], 
                          color='orange', s=15, alpha=0.8, label=f'XGB Anomalies ({len(xgb_anomaly_indices)})')
    axes[1, 0].set_title(f'Voltage Signal with XGBoost Anomalies', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Sample Index', fontsize=11)
    axes[1, 0].set_ylabel(f'Voltage (Pair {pair_name})', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Current with XGB anomalies
    axes[1, 1].plot(current_data, color='red', alpha=0.7, linewidth=0.8, label='Current')
    if len(xgb_anomaly_indices) > 0:
        axes[1, 1].scatter(xgb_anomaly_indices, current_data[xgb_anomaly_indices], 
                          color='darkorange', s=15, alpha=0.8, label=f'XGB Anomalies ({len(xgb_anomaly_indices)})')
    axes[1, 1].set_title(f'Current Signal with XGBoost Anomalies', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel(f'Current (Pair {pair_name})', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: IF Anomaly scores
    axes[2, 0].plot(window_indices, iso_scores, color='blue', alpha=0.7, linewidth=1, label='IF Scores')
    if len(iso_anomaly_indices) > 0:
        axes[2, 0].scatter(iso_anomaly_indices, iso_scores[iso_anomalies == 1], 
                          color='red', s=15, alpha=0.8, label='IF Anomalies')
    axes[2, 0].set_title(f'Isolation Forest Anomaly Scores', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Window Center Index', fontsize=11)
    axes[2, 0].set_ylabel('Anomaly Score', fontsize=11)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: XGB Anomaly scores
    axes[2, 1].plot(window_indices, xgb_scores, color='orange', alpha=0.7, linewidth=1, label='XGB Scores')
    if len(xgb_anomaly_indices) > 0:
        axes[2, 1].scatter(xgb_anomaly_indices, xgb_scores[xgb_anomalies == 1], 
                          color='darkorange', s=15, alpha=0.8, label='XGB Anomalies')
    axes[2, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    axes[2, 1].set_title(f'XGBoost Anomaly Probabilities', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Window Center Index', fontsize=11)
    axes[2, 1].set_ylabel('Anomaly Probability', fontsize=11)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Graph_Images/enhanced_anomaly_detection_{pair_name}_{current_time}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Graph_Images/enhanced_anomaly_detection_{pair_name}_{current_time}.png")

@safe_plot
def create_comprehensive_summary(all_results):
    """Create comprehensive summary plots"""
    print(f"\nCreating comprehensive summary...")
    
    pairs = list(all_results.keys())
    if not pairs:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Anomaly Detection Summary', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    iso_counts = [all_results[pair]['iso_anomalies'] for pair in pairs]
    xgb_counts = [all_results[pair]['xgb_anomalies'] for pair in pairs]
    overlap_counts = [all_results[pair]['overlap_anomalies'] for pair in pairs]
    iso_percentages = [all_results[pair]['iso_percentage'] for pair in pairs]
    xgb_percentages = [all_results[pair]['xgb_percentage'] for pair in pairs]
    
    # Plot 1: Anomaly counts comparison
    x = np.arange(len(pairs))
    width = 0.25
    
    axes[0, 0].bar(x - width, iso_counts, width, label='Isolation Forest', color='blue', alpha=0.7)
    axes[0, 0].bar(x, xgb_counts, width, label='XGBoost', color='orange', alpha=0.7)
    axes[0, 0].bar(x + width, overlap_counts, width, label='Overlap', color='purple', alpha=0.7)
    
    axes[0, 0].set_xlabel('Signal Pairs', fontsize=11)
    axes[0, 0].set_ylabel('Number of Anomalies', fontsize=11)
    axes[0, 0].set_title('Anomaly Detection Counts by Method', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(pairs)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Percentage comparison
    axes[0, 1].bar(x - width/2, iso_percentages, width, label='Isolation Forest', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, xgb_percentages, width, label='XGBoost', color='orange', alpha=0.7)
    
    axes[0, 1].set_xlabel('Signal Pairs', fontsize=11)
    axes[0, 1].set_ylabel('Anomaly Percentage (%)', fontsize=11)
    axes[0, 1].set_title('Anomaly Detection Percentages', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(pairs)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Method agreement
    if len(pairs) > 0:
        agreement_ratios = []
        for pair in pairs:
            total_unique = all_results[pair]['iso_anomalies'] + all_results[pair]['xgb_anomalies'] - all_results[pair]['overlap_anomalies']
            if total_unique > 0:
                agreement = all_results[pair]['overlap_anomalies'] / total_unique * 100
            else:
                agreement = 0
            agreement_ratios.append(agreement)
        
        axes[1, 0].bar(pairs, agreement_ratios, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Signal Pairs', fontsize=11)
        axes[1, 0].set_ylabel('Method Agreement (%)', fontsize=11)
        axes[1, 0].set_title('Detection Method Agreement', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overall statistics
    total_windows = [all_results[pair]['total_windows'] for pair in pairs]
    total_anomalies = [all_results[pair]['iso_anomalies'] + all_results[pair]['xgb_anomalies'] for pair in pairs]
    
    axes[1, 1].scatter(total_windows, total_anomalies, s=150, alpha=0.7, color='red')
    for i, pair in enumerate(pairs):
        axes[1, 1].annotate(pair, (total_windows[i], total_anomalies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    axes[1, 1].set_xlabel('Total Analysis Windows', fontsize=11)
    axes[1, 1].set_ylabel('Total Anomalies (IF + XGB)', fontsize=11)
    axes[1, 1].set_title('Windows vs Anomalies Detected', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Graph_Images/comprehensive_summary_{current_time}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Graph_Images/comprehensive_summary_{current_time}.png")

def save_enhanced_results(df_formatted, stats, all_results):
    """Save all enhanced results"""
    print(f"\nSaving enhanced results...")
    
    # Save formatted data summary
    summary_data = []
    for pair_name, results in all_results.items():
        summary_data.append({
            'pair_name': pair_name,
            'total_windows': results['total_windows'],
            'iso_anomalies': results['iso_anomalies'],
            'xgb_anomalies': results['xgb_anomalies'],
            'overlap_anomalies': results['overlap_anomalies'],
            'iso_percentage': results['iso_percentage'],
            'xgb_percentage': results['xgb_percentage'],
            'overlap_percentage': results['overlap_percentage']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"Data/csv_results/enhanced_anomaly_summary_{current_time}.csv", index=False)
    
    # Save detailed results
    with open(f"Data/json_results/enhanced_anomaly_results_{current_time}.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Save signal characteristics
    with open(f"Data/json_results/enhanced_signal_stats_{current_time}.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"  Saved: Data/csv_results/enhanced_anomaly_summary_{current_time}.csv")
    print(f"  Saved: Data/json_results/enhanced_anomaly_results_{current_time}.json")
    print(f"  Saved: Data/json_results/enhanced_signal_stats_{current_time}.json")

def main():
    """Main execution function for enhanced 20Hz anomaly detection"""
    print("=" * 70)
    print("ENHANCED 20HZ DATA FORMATTER AND ANOMALY DETECTION")
    print("=" * 70)
    start_time = time.time()
    
    # Step 1: Load and format the data
    print("\n" + "="*50)
    print("STEP 1: DATA FORMATTING AND PREPROCESSING")
    print("="*50)
    dataset_path = "Dataset/merged_data_20Hz.csv"
    df_formatted = load_and_format_20hz_data(dataset_path)
    
    # Step 2: Analyze signal characteristics
    print("\n" + "="*50)
    print("STEP 2: SIGNAL CHARACTERISTIC ANALYSIS")
    print("="*50)
    stats = analyze_signal_characteristics(df_formatted)
    
    # Step 3: Plot formatted signals
    print("\n" + "="*50)
    print("STEP 3: SIGNAL VISUALIZATION")
    print("="*50)
    plot_formatted_time_series(df_formatted)
    plot_voltage_current_relationships(df_formatted)
    
    # Step 4: Enhanced anomaly detection
    print("\n" + "="*50)
    print("STEP 4: ENHANCED ANOMALY DETECTION")
    print("="*50)
    all_results = analyze_signals_with_sliding_window(
        df_formatted, 
        window_size=100,    # Analyze 100-point windows (5 seconds at 20Hz)
        step_size=25,       # Move by 25 points (1.25 seconds)
        contamination=0.08  # Expect 8% anomalies
    )
    
    # Step 5: Create comprehensive summary
    print("\n" + "="*50)
    print("STEP 5: COMPREHENSIVE SUMMARY")
    print("="*50)
    create_comprehensive_summary(all_results)
    
    # Step 6: Save all results
    print("\n" + "="*50)
    print("STEP 6: SAVING RESULTS")
    print("="*50)
    save_enhanced_results(df_formatted, stats, all_results)
    
    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Processed {len(df_formatted)} total data points")
    print(f"Original records: {df_formatted['record_id'].nunique()}")
    print(f"Average points per record: {len(df_formatted) / df_formatted['record_id'].nunique():.1f}")
    
    for pair_name, results in all_results.items():
        print(f"\nSignal Pair {pair_name}:")
        print(f"  - Analysis windows: {results['total_windows']}")
        print(f"  - Isolation Forest: {results['iso_anomalies']} anomalies ({results['iso_percentage']:.1f}%)")
        print(f"  - XGBoost: {results['xgb_anomalies']} anomalies ({results['xgb_percentage']:.1f}%)")
        print(f"  - Method overlap: {results['overlap_anomalies']} anomalies ({results['overlap_percentage']:.1f}%)")
    
    print(f"\nGenerated files:")
    print(f"  - Formatted data CSV with {len(df_formatted)} rows")
    print(f"  - Signal visualization graphs")
    print(f"  - Enhanced anomaly detection plots")
    print(f"  - Comprehensive summary analysis")
    print(f"  - All results saved with timestamp: {current_time}")
    print("="*70)
    
    return df_formatted, stats, all_results

if __name__ == "__main__":
    df_formatted, stats, results = main()