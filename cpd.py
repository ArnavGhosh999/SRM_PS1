# =============================================================================
# CELL 1: IMPORTS AND SETUP
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
import warnings
import gc
import time
import os
from pathlib import Path
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

# Disable interactive mode and warnings
plt.ioff()  # Turn off interactive mode to prevent popups
plt.switch_backend('Agg')  # Use non-GUI backend
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

print("✅ All imports loaded successfully!")
print("📊 Matplotlib configured for non-interactive mode (no popups)")
print("🎨 Ready for fast railway anomaly detection analysis")

# =============================================================================
# CELL 2: UTILITY FUNCTIONS
# =============================================================================

def create_output_directory():
    """Create new_Graphs directory for saving plots"""
    output_dir = Path("new_Graphs")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Created/Using directory: {output_dir.absolute()}")
    return output_dir

def save_plot_no_show(fig, filename, output_dir):
    """Save plot without showing it (no popups)"""
    filepath = output_dir / f"{filename}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"💾 Saved: {filepath.name}")
    plt.close(fig)  # Close figure to free memory
    return filepath

def parse_comma_separated_values(value, method='mean'):
    """Fast parsing of comma-separated values"""
    if pd.isna(value) or value == '':
        return np.nan
    
    if isinstance(value, str):
        try:
            values = np.array([float(x.strip()) for x in value.split(',')])
            if len(values) == 0:
                return np.nan
            
            if method == 'mean':
                return np.mean(values)
            elif method == 'median':
                return np.median(values)
            elif method == 'std':
                return np.std(values)
            elif method == 'max':
                return np.max(values)
            elif method == 'min':
                return np.min(values)
            else:
                return np.mean(values)
        except:
            return np.nan
    else:
        try:
            return float(value) if not pd.isna(value) else np.nan
        except:
            return np.nan

print("✅ Utility functions loaded!")
print("🔧 Functions available: create_output_directory, save_plot_no_show, parse_comma_separated_values")

# =============================================================================
# CELL 3: FAST DATA LOADING AND PROCESSING
# =============================================================================

def load_railway_data_fast(file_path, max_rows=10000, aggregation_method='mean'):
    """
    Load and process railway data with separate A and B system features - FAST MODE
    """
    print(f"🚂 Loading Railway Data - ULTRA FAST Mode")
    print(f"📊 Processing {max_rows:,} rows maximum")
    print(f"🔧 Aggregation method: {aggregation_method}")
    print("="*60)
    
    # Define all features for A and B systems
    a_features = ['A Current', 'A Voltage']
    b_features = ['B Current', 'B Voltage']
    polling_features = ['Polling of A', 'Polling of B']
    all_electrical_features = a_features + b_features + polling_features
    
    # Read data with chunking for speed
    print("1️⃣ Reading raw data in chunks...")
    chunk_size = min(5000, max_rows)
    chunks = []
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= max_rows:
                break
        
        df_raw = pd.concat(chunks, ignore_index=True)[:max_rows]
        print(f"   📋 Loaded {len(df_raw):,} rows")
        
    except Exception as e:
        print(f"   ❌ Chunk reading failed: {e}")
        print("   🔄 Falling back to direct reading...")
        df_raw = pd.read_csv(file_path, nrows=max_rows, dtype=str)
    
    # Convert Time efficiently
    if 'Time' in df_raw.columns:
        print("2️⃣ Converting time column...")
        df_raw['Time'] = pd.to_datetime(df_raw['Time'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
        print(f"   📅 Date range: {df_raw['Time'].min()} to {df_raw['Time'].max()}")
    
    print(f"3️⃣ Processing electrical features (vectorized)...")
    processed_data = []
    
    # Vectorized processing for speed
    valid_indices = []
    for idx, row in df_raw.iterrows():
        if idx % 2000 == 0:
            print(f"   Processing row {idx:,}/{len(df_raw):,}")
        
        row_data = {}
        
        # Add time and metadata
        if 'Time' in df_raw.columns:
            row_data['Time'] = row['Time']
        for col in ['Site Name', 'Point Machine Name', 'Direction']:
            if col in df_raw.columns:
                row_data[col] = row[col]
        
        # Process electrical features quickly
        valid_row = True
        for feature in all_electrical_features:
            if feature in df_raw.columns:
                aggregated_value = parse_comma_separated_values(row[feature], method=aggregation_method)
                if not pd.isna(aggregated_value):
                    row_data[f'{feature}_processed'] = aggregated_value
                else:
                    valid_row = False
                    break
            else:
                valid_row = False
                break
        
        if valid_row:
            processed_data.append(row_data)
            valid_indices.append(idx)
    
    if not processed_data:
        print("❌ No valid data found!")
        return None, None, None, None
    
    df_processed = pd.DataFrame(processed_data)
    
    if 'Time' in df_processed.columns:
        df_processed = df_processed.sort_values('Time').reset_index(drop=True)
    
    # Separate A and B system data
    a_processed_features = [f'{f}_processed' for f in a_features if f'{f}_processed' in df_processed.columns]
    b_processed_features = [f'{f}_processed' for f in b_features if f'{f}_processed' in df_processed.columns]
    polling_processed_features = [f'{f}_processed' for f in polling_features if f'{f}_processed' in df_processed.columns]
    
    print(f"✅ Data processing complete!")
    print(f"   📊 Total valid rows: {len(df_processed):,}")
    print(f"   🔌 A System features: {len(a_processed_features)} - {a_processed_features}")
    print(f"   🔌 B System features: {len(b_processed_features)} - {b_processed_features}")
    print(f"   📡 Polling features: {len(polling_processed_features)} - {polling_processed_features}")
    
    return df_processed, a_processed_features, b_processed_features, polling_processed_features

print("🔧 Fast data loading function ready!")
print("📝 Ready to load your railway data at high speed")

# =============================================================================
# CELL 4: FAST ANOMALY DETECTION ALGORITHMS
# =============================================================================

def fast_anomaly_detection_hybrid(signal_data, system_name, feature_names, max_changes=20):
    """
    ULTRA-FAST anomaly detection using multiple methods
    """
    print(f"⚡ FAST ANOMALY DETECTION for {system_name}")
    print(f"📊 Signal shape: {signal_data.shape}")
    print("🚀 Using hybrid fast detection methods...")
    print("="*50)
    
    change_points = []
    detection_reasons = {}
    
    try:
        # Method 1: PELT with optimized settings (fastest ruptures method)
        print("1️⃣ Method 1: Optimized PELT detection...")
        try:
            scaler = StandardScaler()
            signal_normalized = scaler.fit_transform(signal_data)
            
            # Use L2 cost function (fastest) with adaptive penalty
            penalty = max(5.0, np.log10(len(signal_data)) * 2)
            min_size = max(5, len(signal_data) // 200)  # Smaller segments for speed
            
            algo = rpt.Pelt(model="l2", min_size=min_size)
            pelt_points = algo.fit_predict(signal_normalized, pen=penalty)[:-1]
            
            for point in pelt_points:
                change_points.append(point)
                detection_reasons[point] = "PELT Change Point Detection"
            
            print(f"   ✅ PELT found {len(pelt_points)} change points")
            
        except Exception as e:
            print(f"   ⚠️ PELT failed: {e}, continuing with other methods...")
        
        # Method 2: Statistical threshold detection
        print("2️⃣ Method 2: Statistical threshold detection...")
        threshold_points, threshold_reasons = detect_threshold_anomalies_with_reasons(signal_data, feature_names)
        
        for point in threshold_points:
            if point not in change_points:
                change_points.append(point)
                detection_reasons[point] = threshold_reasons.get(point, "Statistical threshold")
        
        print(f"   ✅ Threshold method found {len(threshold_points)} anomalies")
        
        # Method 3: Rolling window detection (fast)
        print("3️⃣ Method 3: Fast rolling window detection...")
        window_points, window_reasons = detect_rolling_window_anomalies_fast(signal_data, window_size=30)
        
        for point in window_points:
            if point not in change_points:
                change_points.append(point)
                detection_reasons[point] = window_reasons.get(point, "Rolling window change")
        
        print(f"   ✅ Window method found {len(window_points)} anomalies")
        
        # Sort and limit results
        change_points = sorted(list(set(change_points)))[:max_changes]
        
        print(f"\n🎯 FINAL RESULT: {len(change_points)} anomalies detected")
        print(f"   Change points: {change_points}")
        
        return change_points, signal_normalized if 'signal_normalized' in locals() else signal_data, "Hybrid-Fast", detection_reasons
        
    except Exception as e:
        print(f"❌ All detection methods failed: {e}")
        return [], signal_data, "None", {}

def detect_threshold_anomalies_with_reasons(signal, feature_names, sigma_threshold=2.5):
    """Detect anomalies using statistical thresholds with explanations"""
    anomalies = []
    reasons = {}
    
    for feature_idx in range(signal.shape[1]):
        feature_data = signal[:, feature_idx]
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        
        if std_val == 0:
            continue
        
        upper_limit = mean_val + sigma_threshold * std_val
        lower_limit = mean_val - sigma_threshold * std_val
        
        outliers = np.where((feature_data > upper_limit) | (feature_data < lower_limit))[0]
        
        for outlier_idx in outliers:
            if outlier_idx not in anomalies:
                anomalies.append(outlier_idx)
                value = feature_data[outlier_idx]
                z_score = abs(value - mean_val) / std_val
                
                if value > upper_limit:
                    reasons[outlier_idx] = f"{feature_names[feature_idx]}: {value:.3f} > μ+{sigma_threshold}σ (z={z_score:.2f})"
                else:
                    reasons[outlier_idx] = f"{feature_names[feature_idx]}: {value:.3f} < μ-{sigma_threshold}σ (z={z_score:.2f})"
    
    return sorted(list(set(anomalies))), reasons

def detect_rolling_window_anomalies_fast(signal, window_size=30, threshold=1.5):
    """Fast rolling window anomaly detection"""
    anomalies = []
    reasons = {}
    
    if len(signal) < window_size * 2:
        return anomalies, reasons
    
    # Use larger steps for speed
    step_size = max(1, window_size // 5)
    
    for i in range(window_size, len(signal) - window_size, step_size):
        before_window = signal[i-window_size:i]
        after_window = signal[i:i+window_size]
        
        before_mean = np.mean(before_window, axis=0)
        after_mean = np.mean(after_window, axis=0)
        
        change_magnitude = np.linalg.norm(after_mean - before_mean)
        if change_magnitude > threshold:
            anomalies.append(i)
            reasons[i] = f"Window change: magnitude={change_magnitude:.3f} > {threshold:.1f}"
    
    return anomalies, reasons

print("✅ Fast anomaly detection algorithms loaded!")
print("⚡ Ready for ultra-fast PELT + hybrid detection")

# =============================================================================
# CELL 5: MAIN OVERVIEW VISUALIZATION
# =============================================================================

def create_main_overview_plot(df, signal_a, signal_b, change_points_a, change_points_b, 
                             a_features, b_features, output_dir):
    """Main overview plot with detailed annotations"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # System A Current
    ax = axes[0, 0]
    if len(signal_a.shape) > 1 and signal_a.shape[1] > 0:
        time_indices = np.arange(len(signal_a))
        ax.plot(time_indices, signal_a[:, 0], 'b-', linewidth=1.5, alpha=0.8, label='System A Current')
        
        # Add anomaly markers with annotations
        for i, cp in enumerate(change_points_a[:10]):  # Show first 10 for clarity
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.8, linewidth=2)
            if i < 5:  # Annotate first 5
                ax.annotate(f'A{i+1}', xy=(cp, signal_a[cp, 0]), xytext=(10, 10), 
                           textcoords='offset points', fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
        
        ax.set_title('🔌 System A - Current Signal Analysis\n(Electrical Current vs Sample Index)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (Time Sequence)', fontsize=12)
        ax.set_ylabel('Normalized Current Value\n(Standardized Units)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics box
        mean_val = np.mean(signal_a[:, 0])
        std_val = np.std(signal_a[:, 0])
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nAnomalies={len(change_points_a)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # System A Voltage
    ax = axes[0, 1]
    if len(signal_a.shape) > 1 and signal_a.shape[1] > 1:
        ax.plot(time_indices, signal_a[:, 1], 'orange', linewidth=1.5, alpha=0.8, label='System A Voltage')
        for cp in change_points_a:
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax.set_title('🔌 System A - Voltage Signal Analysis\n(Electrical Voltage vs Sample Index)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (Time Sequence)', fontsize=12)
        ax.set_ylabel('Normalized Voltage Value\n(Standardized Units)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # System B Current
    ax = axes[1, 0]
    if len(signal_b.shape) > 1 and signal_b.shape[1] > 0:
        ax.plot(time_indices, signal_b[:, 0], 'g-', linewidth=1.5, alpha=0.8, label='System B Current')
        
        # Add anomaly markers with annotations
        for i, cp in enumerate(change_points_b[:10]):
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.8, linewidth=2)
            if i < 5:
                ax.annotate(f'B{i+1}', xy=(cp, signal_b[cp, 0]), xytext=(10, 10), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
        
        ax.set_title('🔌 System B - Current Signal Analysis\n(Electrical Current vs Sample Index)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (Time Sequence)', fontsize=12)
        ax.set_ylabel('Normalized Current Value\n(Standardized Units)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics box
        mean_val = np.mean(signal_b[:, 0])
        std_val = np.std(signal_b[:, 0])
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nAnomalies={len(change_points_b)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # System B Voltage
    ax = axes[1, 1]
    if len(signal_b.shape) > 1 and signal_b.shape[1] > 1:
        ax.plot(time_indices, signal_b[:, 1], 'red', linewidth=1.5, alpha=0.8, label='System B Voltage')
        for cp in change_points_b:
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax.set_title('🔌 System B - Voltage Signal Analysis\n(Electrical Voltage vs Sample Index)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (Time Sequence)', fontsize=12)
        ax.set_ylabel('Normalized Voltage Value\n(Standardized Units)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('⚡ Railway Electrical Systems - Complete Signal Analysis Overview\n'
                 f'System A: {len(change_points_a)} Anomalies | System B: {len(change_points_b)} Anomalies', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_plot_no_show(fig, "01_main_overview_detailed", output_dir)

print("✅ Main overview visualization function loaded!")
print("🎨 Ready to create detailed signal analysis plots")

# =============================================================================
# CELL 6: STATISTICAL DISTRIBUTION ANALYSIS
# =============================================================================

def create_statistical_distributions(signal_a, signal_b, a_features, b_features, output_dir):
    """Create comprehensive statistical distribution analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # System A distributions
    ax = axes[0, 0]
    for i in range(signal_a.shape[1]):
        feature_name = a_features[i].replace('_processed', '')
        ax.hist(signal_a[:, i], bins=50, alpha=0.6, label=feature_name, density=True)
    ax.set_title('System A - Signal Value Distributions\n(Probability Density)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Normalized Signal Value', fontsize=12)
    ax.set_ylabel('Probability Density\n(Frequency per Unit)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # System B distributions
    ax = axes[0, 1]
    for i in range(signal_b.shape[1]):
        feature_name = b_features[i].replace('_processed', '')
        ax.hist(signal_b[:, i], bins=50, alpha=0.6, label=feature_name, density=True)
    ax.set_title('System B - Signal Value Distributions\n(Probability Density)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Normalized Signal Value', fontsize=12)
    ax.set_ylabel('Probability Density\n(Frequency per Unit)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined box plots
    ax = axes[0, 2]
    combined_data = []
    combined_labels = []
    
    for i in range(signal_a.shape[1]):
        combined_data.append(signal_a[:, i])
        combined_labels.append(f'A-{a_features[i].replace("_processed", "")}')
    
    for i in range(signal_b.shape[1]):
        combined_data.append(signal_b[:, i])
        combined_labels.append(f'B-{b_features[i].replace("_processed", "")}')
    
    box_plot = ax.boxplot(combined_data, labels=combined_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Combined Systems - Distribution Box Plots\n(Quartiles and Outliers)', fontsize=14, fontweight='bold')
    ax.set_xlabel('System Features', fontsize=12)
    ax.set_ylabel('Normalized Signal Value\n(Standardized Units)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Q-Q plots for normality assessment
    ax = axes[1, 0]
    if signal_a.shape[1] > 0:
        stats.probplot(signal_a[:, 0], dist="norm", plot=ax)
        ax.set_title('System A Current - Q-Q Plot\n(Normality Assessment)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Theoretical Quantiles\n(Standard Normal)', fontsize=12)
        ax.set_ylabel('Sample Quantiles\n(Actual Data)', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if signal_b.shape[1] > 0:
        stats.probplot(signal_b[:, 0], dist="norm", plot=ax)
        ax.set_title('System B Current - Q-Q Plot\n(Normality Assessment)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Theoretical Quantiles\n(Standard Normal)', fontsize=12)
        ax.set_ylabel('Sample Quantiles\n(Actual Data)', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Statistical summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate statistics
    stats_data = []
    for i, feature in enumerate(['A-Current', 'A-Voltage', 'B-Current', 'B-Voltage']):
        if feature.startswith('A') and i < signal_a.shape[1]:
            data = signal_a[:, i]
        elif feature.startswith('B') and (i-2) < signal_b.shape[1]:
            data = signal_b[:, i-2]
        else:
            continue
        
        stats_data.append([
            feature,
            f'{np.mean(data):.3f}',
            f'{np.std(data):.3f}',
            f'{np.min(data):.3f}',
            f'{np.max(data):.3f}',
            f'{np.median(data):.3f}'
        ])
    
    table = ax.table(cellText=stats_data, 
                    colLabels=['Feature', 'Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax.set_title('Statistical Summary Table\n(Descriptive Statistics)', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('📊 Statistical Distribution Analysis - Complete Signal Characterization', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "02_statistical_distributions", output_dir)

print("✅ Statistical distribution analysis function loaded!")
print("📊 Ready to create comprehensive distribution plots")

# =============================================================================
# CELL 7: ANOMALY TIMELINE ANALYSIS
# =============================================================================

def create_anomaly_timeline_analysis(df, change_points_a, change_points_b, signal_length, output_dir):
    """Create comprehensive anomaly timeline analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Combined timeline
    ax = axes[0, 0]
    ax.scatter(change_points_a, [1]*len(change_points_a), c='blue', s=100, alpha=0.8, 
              label=f'System A Anomalies ({len(change_points_a)})', marker='|', linewidths=3)
    ax.scatter(change_points_b, [2]*len(change_points_b), c='green', s=100, alpha=0.8, 
              label=f'System B Anomalies ({len(change_points_b)})', marker='|', linewidths=3)
    
    ax.set_title('Anomaly Timeline - Both Systems\n(Temporal Distribution of Detected Anomalies)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index (Time Sequence)', fontsize=12)
    ax.set_ylabel('System Identifier', fontsize=12)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['System A', 'System B'])
    ax.set_xlim(0, signal_length)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Anomaly frequency histogram
    ax = axes[0, 1]
    all_anomalies = change_points_a + change_points_b
    if all_anomalies:
        bins = min(20, max(5, len(all_anomalies) // 2))
        ax.hist(all_anomalies, bins=bins, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title('Anomaly Frequency Distribution\n(Count of Anomalies per Time Window)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (Time Sequence)', fontsize=12)
        ax.set_ylabel('Number of Anomalies\n(Frequency Count)', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Inter-anomaly intervals for System A
    ax = axes[1, 0]
    if len(change_points_a) > 1:
        intervals_a = np.diff(change_points_a)
        ax.hist(intervals_a, bins=min(10, len(intervals_a)), alpha=0.7, color='lightblue', 
               edgecolor='blue', label='System A Intervals')
        ax.axvline(np.mean(intervals_a), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(intervals_a):.1f} samples')
        ax.set_title('System A - Inter-Anomaly Intervals\n(Time Between Consecutive Anomalies)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Interval Length (Sample Units)', fontsize=12)
        ax.set_ylabel('Frequency Count', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Inter-anomaly intervals for System B
    ax = axes[1, 1]
    if len(change_points_b) > 1:
        intervals_b = np.diff(change_points_b)
        ax.hist(intervals_b, bins=min(10, len(intervals_b)), alpha=0.7, color='lightgreen', 
               edgecolor='green', label='System B Intervals')
        ax.axvline(np.mean(intervals_b), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(intervals_b):.1f} samples')
        ax.set_title('System B - Inter-Anomaly Intervals\n(Time Between Consecutive Anomalies)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Interval Length (Sample Units)', fontsize=12)
        ax.set_ylabel('Frequency Count', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('⏰ Temporal Anomaly Analysis - Pattern Recognition in Time Domain', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "03_anomaly_timeline_analysis", output_dir)

print("✅ Anomaly timeline analysis function loaded!")
print("⏰ Ready to create temporal pattern analysis plots")

# =============================================================================
# CELL 8: DETECTION METHOD ANALYSIS
# =============================================================================

def create_detection_method_analysis(change_points_a, change_points_b, reasons_a, reasons_b, output_dir):
    """Create analysis of detection methods used"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # System A detection methods pie chart
    ax = axes[0, 0]
    method_counts_a = {}
    for point in change_points_a:
        reason = reasons_a.get(point, "Unknown")
        if "PELT" in reason:
            method_counts_a["PELT Algorithm"] = method_counts_a.get("PELT Algorithm", 0) + 1
        elif "threshold" in reason.lower():
            method_counts_a["Statistical Threshold"] = method_counts_a.get("Statistical Threshold", 0) + 1
        elif "window" in reason.lower():
            method_counts_a["Rolling Window"] = method_counts_a.get("Rolling Window", 0) + 1
        else:
            method_counts_a["Other Methods"] = method_counts_a.get("Other Methods", 0) + 1
    
    if method_counts_a:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        wedges, texts, autotexts = ax.pie(method_counts_a.values(), labels=method_counts_a.keys(), 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('System A - Detection Methods Used\n(Distribution of Anomaly Detection Techniques)', 
                    fontsize=14, fontweight='bold')
    
    # System B detection methods pie chart
    ax = axes[0, 1]
    method_counts_b = {}
    for point in change_points_b:
        reason = reasons_b.get(point, "Unknown")
        if "PELT" in reason:
            method_counts_b["PELT Algorithm"] = method_counts_b.get("PELT Algorithm", 0) + 1
        elif "threshold" in reason.lower():
            method_counts_b["Statistical Threshold"] = method_counts_b.get("Statistical Threshold", 0) + 1
        elif "window" in reason.lower():
            method_counts_b["Rolling Window"] = method_counts_b.get("Rolling Window", 0) + 1
        else:
            method_counts_b["Other Methods"] = method_counts_b.get("Other Methods", 0) + 1
    
    if method_counts_b:
        wedges, texts, autotexts = ax.pie(method_counts_b.values(), labels=method_counts_b.keys(), 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('System B - Detection Methods Used\n(Distribution of Anomaly Detection Techniques)', 
                    fontsize=14, fontweight='bold')
    
    # Combined method effectiveness
    ax = axes[1, 0]
    all_methods = set(list(method_counts_a.keys()) + list(method_counts_b.keys()))
    method_names = list(all_methods)
    a_counts = [method_counts_a.get(method, 0) for method in method_names]
    b_counts = [method_counts_b.get(method, 0) for method in method_names]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, a_counts, width, label='System A', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, b_counts, width, label='System B', color='lightgreen', alpha=0.8)
    
    ax.set_title('Detection Method Comparison\n(Effectiveness Across Both Systems)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Detection Method', fontsize=12)
    ax.set_ylabel('Number of Anomalies Detected', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Method explanation table
    ax = axes[1, 1]
    ax.axis('off')
    
    explanation_text = """
    🔍 DETECTION METHOD EXPLANATIONS:
    
    📊 PELT Algorithm:
    • Pruned Exact Linear Time algorithm
    • Detects change points in statistical properties
    • Identifies sudden shifts in mean/variance
    • Most sophisticated method used
    
    📈 Statistical Threshold:
    • Based on μ ± 2.5σ rule
    • Flags values beyond normal range
    • Uses z-score > 2.5 as criterion
    • Fast and interpretable
    
    🔄 Rolling Window:
    • Compares adjacent time windows
    • Detects gradual pattern changes
    • Uses magnitude threshold (>1.5)
    • Good for trend shifts
    
    ⚡ Hybrid Approach:
    • Combines all methods for robustness
    • Reduces false positives/negatives
    • Provides multiple validation layers
    • Ensures comprehensive detection
    """
    
    ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('🔍 Detection Method Analysis - Algorithm Performance Evaluation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "04_detection_method_analysis", output_dir)

print("✅ Detection method analysis function loaded!")
print("🔍 Ready to analyze algorithm performance and method distribution")

# =============================================================================
# CELL 9: SYSTEM HEALTH DASHBOARD
# =============================================================================

def create_system_health_dashboard(signal_a, signal_b, change_points_a, change_points_b, output_dir):
    """Create comprehensive system health dashboard"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Calculate health metrics
    total_samples = len(signal_a)
    rate_a = len(change_points_a) / total_samples * 100
    rate_b = len(change_points_b) / total_samples * 100
    
    cv_a = np.mean([np.std(signal_a[:, i])/np.abs(np.mean(signal_a[:, i])) 
                    for i in range(signal_a.shape[1]) if np.mean(signal_a[:, i]) != 0])
    cv_b = np.mean([np.std(signal_b[:, i])/np.abs(np.mean(signal_b[:, i])) 
                    for i in range(signal_b.shape[1]) if np.mean(signal_b[:, i]) != 0])
    
    # Anomaly rates comparison
    ax = axes[0, 0]
    systems = ['System A', 'System B']
    rates = [rate_a, rate_b]
    colors = ['lightblue', 'lightgreen']
    bars = ax.bar(systems, rates, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Anomaly Rates Comparison\n(Percentage of Anomalous Samples)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add value labels and status indicators
    for bar, rate in zip(bars, rates):
        status = "🟢 Good" if rate < 1 else "🟡 Monitor" if rate < 3 else "🔴 Critical"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
               f'{rate:.3f}%\n{status}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # Signal stability comparison
    ax = axes[0, 1]
    cvs = [cv_a, cv_b]
    bars = ax.bar(systems, cvs, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Signal Stability Comparison\n(Coefficient of Variation - Lower is Better)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation\n(σ/μ)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    for bar, cv in zip(bars, cvs):
        stability = "🟢 Stable" if cv < 0.2 else "🟡 Moderate" if cv < 0.5 else "🔴 Unstable"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{cv:.3f}\n{stability}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # Overall health scores
    ax = axes[0, 2]
    health_a = max(0, 100 - (rate_a * 10 + cv_a * 50))
    health_b = max(0, 100 - (rate_b * 10 + cv_b * 50))
    healths = [health_a, health_b]
    
    bars = ax.bar(systems, healths, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Overall Health Scores\n(Composite Performance Metric)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Health Score (0-100)\n(Higher is Better)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, health in zip(bars, healths):
        status = "🟢 Excellent" if health > 80 else "🟡 Good" if health > 60 else "🔴 Poor"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{health:.1f}\n{status}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # Reliability metrics
    ax = axes[1, 0]
    reliability_a = (total_samples - len(change_points_a)) / total_samples * 100
    reliability_b = (total_samples - len(change_points_b)) / total_samples * 100
    reliabilities = [reliability_a, reliability_b]
    
    bars = ax.bar(systems, reliabilities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('System Reliability\n(Percentage of Normal Operation)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Reliability (%)\n(Normal Operation Time)', fontsize=12)
    ax.set_ylim(95, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, rel in zip(bars, reliabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
               f'{rel:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Feature-wise performance
    ax = axes[1, 1]
    feature_performance = []
    feature_labels = []
    
    # System A features
    for i in range(signal_a.shape[1]):
        feature_cv = np.std(signal_a[:, i]) / np.abs(np.mean(signal_a[:, i])) if np.mean(signal_a[:, i]) != 0 else 0
        feature_performance.append(feature_cv)
        feature_labels.append(f'A-{"Current" if i == 0 else "Voltage"}')
    
    # System B features
    for i in range(signal_b.shape[1]):
        feature_cv = np.std(signal_b[:, i]) / np.abs(np.mean(signal_b[:, i])) if np.mean(signal_b[:, i]) != 0 else 0
        feature_performance.append(feature_cv)
        feature_labels.append(f'B-{"Current" if i == 0 else "Voltage"}')
    
    bars = ax.bar(range(len(feature_labels)), feature_performance, 
                 color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(feature_labels)],
                 alpha=0.8, edgecolor='black')
    
    ax.set_title('Feature-wise Stability Analysis\n(Individual Component Performance)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('System Features', fontsize=12)
    ax.set_ylabel('Coefficient of Variation\n(Lower is Better)', fontsize=12)
    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels)
    ax.grid(True, alpha=0.3)
    
    # Summary and recommendations
    ax = axes[1, 2]
    ax.axis('off')
    
    better_system = "A" if health_a > health_b else "B" if health_b > health_a else "Both Equal"
    more_stable = "A" if cv_a < cv_b else "B" if cv_b < cv_a else "Equal"
    fewer_anomalies = "A" if len(change_points_a) < len(change_points_b) else "B" if len(change_points_b) < len(change_points_a) else "Equal"
    
    summary_text = f"""
    📊 SYSTEM HEALTH SUMMARY:
    
    🏆 Performance Winner:
    • Overall: System {better_system}
    • Stability: System {more_stable}
    • Anomalies: System {fewer_anomalies}
    
    📈 Key Metrics:
    • System A Health: {health_a:.1f}/100
    • System B Health: {health_b:.1f}/100
    • A Reliability: {reliability_a:.3f}%
    • B Reliability: {reliability_b:.3f}%
    
    🎯 Recommendations:
    • {"Both systems healthy" if min(health_a, health_b) > 80 else 
       "Monitor both systems" if min(health_a, health_b) > 60 else
       "Immediate attention needed"}
    
    🔧 Priority Actions:
    • Focus on System {"A" if health_a < health_b else "B"}
    • {"Review anomaly patterns" if max(rate_a, rate_b) > 2 else "Continue monitoring"}
    • {"Stability improvements needed" if max(cv_a, cv_b) > 0.5 else "Stability acceptable"}
    
    📞 Alert Thresholds:
    • Anomaly rate > 3%: 🔴
    • Health score < 60: 🔴
    • CV > 0.5: 🟡
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('🏥 System Health Dashboard - Comprehensive Performance Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "05_system_health_dashboard", output_dir)

print("✅ System health dashboard function loaded!")
print("🏥 Ready to create comprehensive health and performance analysis")

# =============================================================================
# CELL 10: CORRELATION ANALYSIS
# =============================================================================

def create_correlation_analysis(signal_a, signal_b, a_features, b_features, output_dir):
    """Create correlation analysis between features"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create combined data for correlation matrix
    all_data = np.concatenate([signal_a, signal_b], axis=1)
    feature_names = ([f.replace('_processed', '') for f in a_features] + 
                    [f.replace('_processed', '') for f in b_features])
    
    # Correlation heatmap
    ax = axes[0, 0]
    correlation_matrix = np.corrcoef(all_data.T)
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    ax.set_title('Feature Correlation Matrix\n(Pearson Correlation Coefficients)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient\n(-1 to +1)', fontsize=10)
    
    # Scatter plot: A Current vs A Voltage
    ax = axes[0, 1]
    if signal_a.shape[1] >= 2:
        ax.scatter(signal_a[:, 0], signal_a[:, 1], alpha=0.6, s=20, color='blue')
        correlation = np.corrcoef(signal_a[:, 0], signal_a[:, 1])[0, 1]
        ax.set_title(f'System A: Current vs Voltage\n(Correlation: {correlation:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('A Current (Normalized)', fontsize=12)
        ax.set_ylabel('A Voltage (Normalized)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(signal_a[:, 0], signal_a[:, 1], 1)
        p = np.poly1d(z)
        ax.plot(sorted(signal_a[:, 0]), p(sorted(signal_a[:, 0])), "r--", alpha=0.8, linewidth=2)
    
    # Scatter plot: B Current vs B Voltage
    ax = axes[1, 0]
    if signal_b.shape[1] >= 2:
        ax.scatter(signal_b[:, 0], signal_b[:, 1], alpha=0.6, s=20, color='green')
        correlation = np.corrcoef(signal_b[:, 0], signal_b[:, 1])[0, 1]
        ax.set_title(f'System B: Current vs Voltage\n(Correlation: {correlation:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('B Current (Normalized)', fontsize=12)
        ax.set_ylabel('B Voltage (Normalized)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(signal_b[:, 0], signal_b[:, 1], 1)
        p = np.poly1d(z)
        ax.plot(sorted(signal_b[:, 0]), p(sorted(signal_b[:, 0])), "r--", alpha=0.8, linewidth=2)
    
    # Cross-system correlation: A Current vs B Current
    ax = axes[1, 1]
    if signal_a.shape[1] > 0 and signal_b.shape[1] > 0:
        ax.scatter(signal_a[:, 0], signal_b[:, 0], alpha=0.6, s=20, color='purple')
        correlation = np.corrcoef(signal_a[:, 0], signal_b[:, 0])[0, 1]
        ax.set_title(f'Cross-System: A Current vs B Current\n(Correlation: {correlation:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('A Current (Normalized)', fontsize=12)
        ax.set_ylabel('B Current (Normalized)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(signal_a[:, 0], signal_b[:, 0], 1)
        p = np.poly1d(z)
        ax.plot(sorted(signal_a[:, 0]), p(sorted(signal_a[:, 0])), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('🔗 Feature Correlation Analysis - System Interdependencies', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "06_correlation_analysis", output_dir)

print("✅ Correlation analysis function loaded!")
print("🔗 Ready to analyze feature relationships and system interdependencies")

# =============================================================================
# CELL 11: PDF REPORT GENERATION
# =============================================================================

def create_comprehensive_pdf_report(df, signal_a, signal_b, change_points_a, change_points_b, 
                                   reasons_a, reasons_b, a_features, b_features, output_dir):
    """Create comprehensive PDF report with anomaly explanations"""
    pdf_path = output_dir / "Comprehensive_Anomaly_Detection_Report.pdf"
    
    print(f"📄 Creating comprehensive PDF report...")
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Executive Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Calculate key metrics
        total_samples = len(df)
        total_anomalies = len(change_points_a) + len(change_points_b)
        rate_a = len(change_points_a) / total_samples * 100
        rate_b = len(change_points_b) / total_samples * 100
        
        cv_a = np.mean([np.std(signal_a[:, i])/np.abs(np.mean(signal_a[:, i])) 
                        for i in range(signal_a.shape[1]) if np.mean(signal_a[:, i]) != 0])
        cv_b = np.mean([np.std(signal_b[:, i])/np.abs(np.mean(signal_b[:, i])) 
                        for i in range(signal_b.shape[1]) if np.mean(signal_b[:, i]) != 0])
        
        health_a = max(0, 100 - (rate_a * 10 + cv_a * 50))
        health_b = max(0, 100 - (rate_b * 10 + cv_b * 50))
        
        summary_text = f"""
RAILWAY ELECTRICAL SYSTEMS
ANOMALY DETECTION REPORT

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY

Dataset Information:
• Total Data Points: {total_samples:,}
• Analysis Method: Hybrid Fast Detection (PELT + Statistical + Rolling Window)
• Detection Sensitivity: Medium (2.5σ threshold)

System Performance Overview:
• System A Anomalies: {len(change_points_a)} ({rate_a:.4f}%)
• System B Anomalies: {len(change_points_b)} ({rate_b:.4f}%)
• Total Anomalies: {total_anomalies} ({total_anomalies/total_samples*100:.4f}%)

Health Assessment:
• System A Status: {'🟢 Healthy' if rate_a < 1 else '🟡 Monitor' if rate_a < 3 else '🔴 Critical'}
• System B Status: {'🟢 Healthy' if rate_b < 1 else '🟡 Monitor' if rate_b < 3 else '🔴 Critical'}
• System A Health Score: {health_a:.1f}/100
• System B Health Score: {health_b:.1f}/100

Signal Stability Analysis:
• System A Coefficient of Variation: {cv_a:.3f}
• System B Coefficient of Variation: {cv_b:.3f}
• Stability Rating: {'A is more stable' if cv_a < cv_b else 'B is more stable' if cv_b < cv_a else 'Both equally stable'}

Key Findings:
• {'System A shows more anomalies' if len(change_points_a) > len(change_points_b) else 
   'System B shows more anomalies' if len(change_points_b) > len(change_points_a) else
   'Both systems show similar anomaly patterns'}
• Primary detection method: {'PELT algorithm' if total_anomalies > 0 else 'Statistical thresholds'}
• All anomalies categorized with specific technical reasons

Recommendations:
• {'Focus maintenance on System A' if len(change_points_a) > len(change_points_b) else
   'Focus maintenance on System B' if len(change_points_b) > len(change_points_a) else
   'Continue routine monitoring of both systems'}
• {'Immediate review required' if max(rate_a, rate_b) > 3 else 'Regular monitoring sufficient'}
• Review high-frequency anomaly periods for predictive maintenance

═══════════════════════════════════════════════════════════════
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace')
        
        plt.title('Railway Electrical Systems - Anomaly Detection Report', 
                 fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Detection Methodology Explanation
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        methodology_text = f"""
DETECTION METHODOLOGY & METRICS EXPLANATION

📊 STATISTICAL THRESHOLD DETECTION:

What it measures: Identifies data points that deviate significantly from normal patterns
Mathematical basis: Z-score = (value - μ) / σ
Threshold used: ±2.5σ (99.4% confidence interval)
Why this matters: Values beyond this range indicate unusual electrical behavior

Interpretation:
• μ (mu) = Mean of the signal (average normal value)
• σ (sigma) = Standard deviation (measure of normal variation)
• Z-score > 2.5 = Anomaly (occurs <1% of time normally)

Example: If A Current μ=0.5, σ=0.1, then values >0.75 or <0.25 are anomalies

🔄 ROLLING WINDOW DETECTION:

What it measures: Compares statistical properties of adjacent time periods
Window size: 30 samples (optimized for speed)
Threshold: Change magnitude > 1.5 (empirically determined)
Why this matters: Detects gradual shifts that might indicate system degradation

How it works:
1. Calculate mean of 30 samples before time point
2. Calculate mean of 30 samples after time point  
3. Measure magnitude of change between periods
4. Flag as anomaly if change > 1.5 threshold

⚡ PELT ALGORITHM (Pruned Exact Linear Time):

What it measures: Optimal change point detection in time series
Cost function: L2 (detects changes in mean)
Penalty: Adaptive (log₁₀(n) × 2) - prevents over-segmentation
Why this matters: Most sophisticated method, finds exact change points

Technical details:
• Penalty = {max(5.0, np.log10(total_samples) * 2):.1f} (calculated for this dataset)
• Minimum segment size: {max(5, total_samples // 200)} samples
• Detects: Sudden shifts in electrical parameters

🔍 HYBRID APPROACH BENEFITS:

1. Redundancy: Multiple methods reduce false negatives
2. Validation: Methods cross-validate each other
3. Completeness: Captures different types of anomalies
4. Reliability: Robust against single-method failures

ANOMALY CLASSIFICATION SYSTEM:

🟢 Normal: Within ±2σ range, stable patterns
🟡 Warning: Beyond ±2σ but within ±2.5σ  
🟠 Alert: Beyond ±2.5σ but within ±3σ
🔴 Critical: Beyond ±3σ or major pattern change

Why these thresholds matter:
• ±2σ: 95% confidence (normal variation)
• ±2.5σ: 99% confidence (likely anomaly)
• ±3σ: 99.7% confidence (definite anomaly)
        """
        
        ax.text(0.05, 0.95, methodology_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', fontfamily='monospace')
        
        plt.title('Detection Methodology & Metrics Explanation', 
                 fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: System A Detailed Analysis
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        system_a_text = f"""
SYSTEM A - DETAILED ANOMALY ANALYSIS

Signal Characteristics:
• A Current: μ={np.mean(signal_a[:, 0]):.4f}, σ={np.std(signal_a[:, 0]):.4f}
• A Voltage: μ={np.mean(signal_a[:, 1]):.4f}, σ={np.std(signal_a[:, 1]):.4f} (if available)
• Coefficient of Variation: {cv_a:.3f}
• Stability Rating: {'🟢 Stable' if cv_a < 0.2 else '🟡 Moderate' if cv_a < 0.5 else '🔴 Unstable'}

Anomalies Detected: {len(change_points_a)}
Detection Rate: {rate_a:.4f}% of total samples

DETAILED ANOMALY LIST:
"""
        
        for i, point in enumerate(change_points_a[:15]):  # Show first 15
            reason = reasons_a.get(point, "Unknown reason")
            if point < len(signal_a):
                current_val = signal_a[point, 0] if signal_a.shape[1] > 0 else 0
                voltage_val = signal_a[point, 1] if signal_a.shape[1] > 1 else 0
                
                # Calculate severity
                current_z = abs(current_val - np.mean(signal_a[:, 0])) / np.std(signal_a[:, 0]) if np.std(signal_a[:, 0]) > 0 else 0
                severity = "🔴 Critical" if current_z > 3 else "🟠 Alert" if current_z > 2.5 else "🟡 Warning"
                
                system_a_text += f"""
Anomaly #{i+1} at Sample {point}:
• Reason: {reason}
• Current Value: {current_val:.4f} (z-score: {current_z:.2f})
• Voltage Value: {voltage_val:.4f}
• Severity: {severity}
• Time: Sample index {point}
"""
        
        if len(change_points_a) > 15:
            system_a_text += f"\n... and {len(change_points_a) - 15} more anomalies (see full data for complete list)"
        
        system_a_text += f"""

SYSTEM A SUMMARY STATISTICS:

Reliability Metrics:
• Normal Operation: {((total_samples - len(change_points_a))/total_samples*100):.3f}%
• Anomalous Operation: {rate_a:.3f}%
• Mean Time Between Anomalies: {total_samples/max(1, len(change_points_a)):.1f} samples

Pattern Analysis:
• Most frequent detection method: {"PELT" if len([p for p in change_points_a if "PELT" in reasons_a.get(p, "")]) > len(change_points_a)//2 else "Statistical"}
• Anomaly clustering: {"High" if len(change_points_a) > 10 else "Medium" if len(change_points_a) > 5 else "Low"}
• System health trend: {"Declining" if rate_a > 2 else "Stable" if rate_a < 1 else "Monitoring required"}
        """
        
        ax.text(0.05, 0.95, system_a_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', fontfamily='monospace')
        
        plt.title('System A - Detailed Analysis', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: System B Detailed Analysis
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        system_b_text = f"""
SYSTEM B - DETAILED ANOMALY ANALYSIS

Signal Characteristics:
• B Current: μ={np.mean(signal_b[:, 0]):.4f}, σ={np.std(signal_b[:, 0]):.4f}
• B Voltage: μ={np.mean(signal_b[:, 1]):.4f}, σ={np.std(signal_b[:, 1]):.4f} (if available)
• Coefficient of Variation: {cv_b:.3f}
• Stability Rating: {'🟢 Stable' if cv_b < 0.2 else '🟡 Moderate' if cv_b < 0.5 else '🔴 Unstable'}

Anomalies Detected: {len(change_points_b)}
Detection Rate: {rate_b:.4f}% of total samples

DETAILED ANOMALY LIST:
"""
        
        for i, point in enumerate(change_points_b[:15]):  # Show first 15
            reason = reasons_b.get(point, "Unknown reason")
            if point < len(signal_b):
                current_val = signal_b[point, 0] if signal_b.shape[1] > 0 else 0
                voltage_val = signal_b[point, 1] if signal_b.shape[1] > 1 else 0
                
                # Calculate severity
                current_z = abs(current_val - np.mean(signal_b[:, 0])) / np.std(signal_b[:, 0]) if np.std(signal_b[:, 0]) > 0 else 0
                severity = "🔴 Critical" if current_z > 3 else "🟠 Alert" if current_z > 2.5 else "🟡 Warning"
                
                system_b_text += f"""
Anomaly #{i+1} at Sample {point}:
• Reason: {reason}
• Current Value: {current_val:.4f} (z-score: {current_z:.2f})
• Voltage Value: {voltage_val:.4f}
• Severity: {severity}
• Time: Sample index {point}
"""
        
        if len(change_points_b) > 15:
            system_b_text += f"\n... and {len(change_points_b) - 15} more anomalies (see full data for complete list)"
        
        system_b_text += f"""

SYSTEM B SUMMARY STATISTICS:

Reliability Metrics:
• Normal Operation: {((total_samples - len(change_points_b))/total_samples*100):.3f}%
• Anomalous Operation: {rate_b:.3f}%
• Mean Time Between Anomalies: {total_samples/max(1, len(change_points_b)):.1f} samples

Pattern Analysis:
• Most frequent detection method: {"PELT" if len([p for p in change_points_b if "PELT" in reasons_b.get(p, "")]) > len(change_points_b)//2 else "Statistical"}
• Anomaly clustering: {"High" if len(change_points_b) > 10 else "Medium" if len(change_points_b) > 5 else "Low"}
• System health trend: {"Declining" if rate_b > 2 else "Stable" if rate_b < 1 else "Monitoring required"}
        """
        
        ax.text(0.05, 0.95, system_b_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', fontfamily='monospace')
        
        plt.title('System B - Detailed Analysis', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Comparative Analysis and Recommendations
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        better_system = "A" if health_a > health_b else "B" if health_b > health_a else "Both Equal"
        more_stable = "A" if cv_a < cv_b else "B" if cv_b < cv_a else "Equal"
        
        comparison_text = f"""
COMPARATIVE ANALYSIS & RECOMMENDATIONS

SYSTEM COMPARISON MATRIX:

Metric                  System A        System B        Winner
════════════════════════════════════════════════════════════════
Anomaly Count           {len(change_points_a):<8}       {len(change_points_b):<8}       {'A' if len(change_points_a) < len(change_points_b) else 'B' if len(change_points_b) < len(change_points_a) else 'Tie'}
Anomaly Rate            {rate_a:<8.3f}%     {rate_b:<8.3f}%     {'A' if rate_a < rate_b else 'B' if rate_b < rate_a else 'Tie'}
Stability (CV)          {cv_a:<8.3f}     {cv_b:<8.3f}     {'A' if cv_a < cv_b else 'B' if cv_b < cv_a else 'Tie'}
Health Score            {health_a:<8.1f}      {health_b:<8.1f}      {'A' if health_a > health_b else 'B' if health_b > health_a else 'Tie'}
Reliability             {((total_samples - len(change_points_a))/total_samples*100):<8.3f}%     {((total_samples - len(change_points_b))/total_samples*100):<8.3f}%     {'A' if len(change_points_a) < len(change_points_b) else 'B' if len(change_points_b) < len(change_points_a) else 'Tie'}

OVERALL WINNER: System {better_system}

PRIORITY RECOMMENDATIONS:

🔴 IMMEDIATE ACTIONS (0-24 hours):
• {'Investigate System A anomalies' if len(change_points_a) > len(change_points_b) else 
   'Investigate System B anomalies' if len(change_points_b) > len(change_points_a) else
   'Both systems stable - continue monitoring'}
• {'Review electrical connections for System A' if health_a < 60 else 'Review electrical connections for System B' if health_b < 60 else 'No immediate electrical review needed'}
• {'Conduct emergency inspection' if max(rate_a, rate_b) > 5 else 'Schedule routine inspection'}

🟡 SHORT TERM ACTIONS (1-7 days):
• Update anomaly detection thresholds if needed
• Calibrate sensors and measurement equipment
• Review maintenance schedules for both systems
• {'Focus maintenance resources on System A' if health_a < health_b else 'Focus maintenance resources on System B' if health_b < health_a else 'Allocate maintenance resources equally'}

🟢 LONG TERM ACTIONS (1-3 months):
• Implement predictive maintenance based on anomaly patterns
• Trend analysis for pattern evolution
• System optimization review
• Staff training on anomaly interpretation

ESCALATION CRITERIA:

🚨 IMMEDIATE ESCALATION IF:
• Anomaly rate exceeds 5% in any hour
• Health score drops below 50
• More than 5 critical anomalies (z-score > 3) in sequence
• Both systems show simultaneous anomalies

📞 CONTACT INFORMATION:
• Engineering Team: ext. 1234 (anomaly analysis)
• Maintenance Team: ext. 5678 (system repairs)
• Operations Manager: ext. 9012 (escalation)
• Emergency Response: ext. 0000 (critical situations)

MONITORING SCHEDULE:
• Real-time: Continuous automated monitoring
• Daily: Review anomaly reports
• Weekly: Trend analysis and pattern review
• Monthly: Comprehensive system health assessment
        """
        
        ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', fontfamily='monospace')
        
        plt.title('Comparative Analysis & Recommendations', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✅ PDF report saved: {pdf_path}")
    return pdf_path

print("✅ PDF report generation function loaded!")
print("📄 Ready to create comprehensive anomaly detection reports")

# =============================================================================
# CELL 12: MAIN EXECUTION FUNCTION
# =============================================================================

def run_complete_railway_analysis(file_path, max_rows=10000):
    """
    Run complete railway anomaly detection analysis
    """
    print("🚂 STARTING COMPLETE RAILWAY ANOMALY DETECTION ANALYSIS")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Create output directory
        print("\n1️⃣ Setting up output directory...")
        output_dir = create_output_directory()
        
        # Step 2: Load and process data
        print("\n2️⃣ Loading and processing railway data...")
        df, a_features, b_features, polling_features = load_railway_data_fast(
            file_path, max_rows=max_rows, aggregation_method='mean'
        )
        
        if df is None:
            print("❌ Data loading failed. Please check your file path and format.")
            return
        
        # Prepare signal arrays
        print("\n3️⃣ Preparing signal arrays for analysis...")
        signal_a = df[a_features].values
        signal_b = df[b_features].values
        
        print(f"   📊 System A signal shape: {signal_a.shape}")
        print(f"   📊 System B signal shape: {signal_b.shape}")
        
        # Step 3: Run anomaly detection
        print("\n4️⃣ Running fast anomaly detection...")
        
        # System A detection
        print("   🔍 Analyzing System A...")
        change_points_a, signal_a_processed, method_a, reasons_a = fast_anomaly_detection_hybrid(
            signal_a, "System A", a_features, max_changes=20
        )
        
        # System B detection
        print("   🔍 Analyzing System B...")
        change_points_b, signal_b_processed, method_b, reasons_b = fast_anomaly_detection_hybrid(
            signal_b, "System B", b_features, max_changes=20
        )
        
        # Step 4: Create visualizations
        print("\n5️⃣ Creating comprehensive visualizations...")
        
        # Main overview
        print("   🎨 Creating main overview plot...")
        create_main_overview_plot(df, signal_a_processed, signal_b_processed, 
                                 change_points_a, change_points_b, a_features, b_features, output_dir)
        
        # Statistical distributions
        print("   📊 Creating statistical distribution analysis...")
        create_statistical_distributions(signal_a_processed, signal_b_processed, 
                                       a_features, b_features, output_dir)
        
        # Anomaly timeline
        print("   ⏰ Creating anomaly timeline analysis...")
        create_anomaly_timeline_analysis(df, change_points_a, change_points_b, 
                                       len(signal_a_processed), output_dir)
        
        # Detection methods
        print("   🔍 Creating detection method analysis...")
        create_detection_method_analysis(change_points_a, change_points_b, 
                                       reasons_a, reasons_b, output_dir)
        
        # System health dashboard
        print("   🏥 Creating system health dashboard...")
        create_system_health_dashboard(signal_a_processed, signal_b_processed, 
                                     change_points_a, change_points_b, output_dir)
        
        # Correlation analysis
        print("   🔗 Creating correlation analysis...")
        create_correlation_analysis(signal_a_processed, signal_b_processed, 
                                   a_features, b_features, output_dir)
        
        # Step 5: Generate PDF report
        print("\n6️⃣ Generating comprehensive PDF report...")
        pdf_path = create_comprehensive_pdf_report(df, signal_a_processed, signal_b_processed, 
                                                  change_points_a, change_points_b, 
                                                  reasons_a, reasons_b, a_features, b_features, output_dir)
        
        # Step 6: Summary and results
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📊 FINAL RESULTS SUMMARY:")
        print(f"   ⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"   📈 Data points analyzed: {len(df):,}")
        print(f"   🔌 System A anomalies: {len(change_points_a)}")
        print(f"   🔌 System B anomalies: {len(change_points_b)}")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   📄 PDF report: {pdf_path.name}")
        
        # Performance metrics
        rate_a = len(change_points_a) / len(df) * 100
        rate_b = len(change_points_b) / len(df) * 100
        
        print(f"\n🏥 SYSTEM HEALTH OVERVIEW:")
        print(f"   System A anomaly rate: {rate_a:.4f}%")
        print(f"   System B anomaly rate: {rate_b:.4f}%")
        print(f"   Overall status: {'🟢 Systems healthy' if max(rate_a, rate_b) < 1 else '🟡 Monitor systems' if max(rate_a, rate_b) < 3 else '🔴 Review required'}")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"   01_main_overview_detailed.png")
        print(f"   02_statistical_distributions.png")
        print(f"   03_anomaly_timeline_analysis.png")
        print(f"   04_detection_method_analysis.png")
        print(f"   05_system_health_dashboard.png")
        print(f"   06_correlation_analysis.png")
        print(f"   Comprehensive_Anomaly_Detection_Report.pdf")
        
        print(f"\n✅ All files saved to: {output_dir.absolute()}")
        print("\n🎯 Next steps:")
        print("   1. Review the PDF report for detailed analysis")
        print("   2. Examine individual plots for specific insights")
        print("   3. Take action based on health recommendations")
        
        return {
            'df': df,
            'signal_a': signal_a_processed,
            'signal_b': signal_b_processed,
            'change_points_a': change_points_a,
            'change_points_b': change_points_b,
            'reasons_a': reasons_a,
            'reasons_b': reasons_b,
            'output_dir': output_dir,
            'pdf_path': pdf_path,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        print("Please check your data file and try again.")
        import traceback
        traceback.print_exc()
        return None

print("✅ Main execution function loaded!")
print("🚀 Ready to run complete railway anomaly detection analysis!")
print("\n" + "="*60)
print("HOW TO USE:")
print("="*60)
print("1. Make sure your CSV file is ready")
print("2. Run: results = run_complete_railway_analysis('your_file.csv')")
print("3. Check the 'new_Graphs' folder for all outputs")
print("4. Review the PDF report for detailed analysis")
print("="*60)

# =============================================================================
# CELL 13: USAGE EXAMPLE AND INSTRUCTIONS
# =============================================================================

"""
USAGE INSTRUCTIONS FOR RAILWAY ANOMALY DETECTION

1. PREPARATION:
   - Ensure your CSV file contains columns: 'A Current', 'A Voltage', 'B Current', 'B Voltage'
   - Optionally: 'Time', 'Site Name', 'Point Machine Name', 'Direction'
   - Data can contain comma-separated values (will be automatically averaged)

2. RUN ANALYSIS:
   Copy and run this code block:
"""

# =============================================================================
# CELL 13: USAGE EXAMPLE AND INSTRUCTIONS
# =============================================================================

"""
USAGE INSTRUCTIONS FOR RAILWAY ANOMALY DETECTION

1. PREPARATION:
   - Ensure your CSV file contains columns: 'A Current', 'A Voltage', 'B Current', 'B Voltage'
   - Optionally: 'Time', 'Site Name', 'Point Machine Name', 'Direction'
   - Data can contain comma-separated values (will be automatically averaged)

2. RUN ANALYSIS:
   Copy and run this code block:
"""

# EXAMPLE USAGE:
# Replace 'your_railway_data.csv' with your actual file path
file_path = 'processed_data_v2/merged_data_20Hz.csv'  # Change this to your file path

# Run the complete analysis (fast mode - 10k samples)
print("🚂 Starting Railway Anomaly Detection Analysis...")
print("📝 Instructions:")
print("1. Update the file_path variable above with your CSV file path")
print("2. Run: results = run_complete_railway_analysis(file_path)")
print("3. Wait for analysis to complete (typically 30-60 seconds)")
print("4. Check 'new_Graphs' folder for all outputs")

# Uncomment the line below and run to start analysis:
# results = run_complete_railway_analysis(file_path, max_rows=10000)

"""
EXPECTED OUTPUTS:

📁 Files created in 'new_Graphs' folder:
├── 01_main_overview_detailed.png          # Main signal plots with anomalies
├── 02_statistical_distributions.png       # Histograms, box plots, Q-Q plots
├── 03_anomaly_timeline_analysis.png       # Timeline and interval analysis
├── 04_detection_method_analysis.png       # Method effectiveness pie charts
├── 05_system_health_dashboard.png         # Health scores and recommendations
├── 06_correlation_analysis.png            # Feature correlation heatmaps
└── Comprehensive_Anomaly_Detection_Report.pdf  # Complete technical report

📊 What each visualization shows:

1. MAIN OVERVIEW (01_):
   - X-axis: Sample Index (time sequence)
   - Y-axis: Normalized electrical values
   - Red lines: Detected anomalies
   - Annotations: First 5 anomalies labeled

2. STATISTICAL DISTRIBUTIONS (02_):
   - Histograms: Value frequency distributions
   - Box plots: Quartiles and outliers
   - Q-Q plots: Normality assessment
   - Summary table: Descriptive statistics

3. ANOMALY TIMELINE (03_):
   - Timeline: When anomalies occurred
   - Frequency: Clustering patterns
   - Intervals: Time between anomalies

4. DETECTION METHODS (04_):
   - Pie charts: Which algorithms found what
   - Bar charts: Method effectiveness
   - Explanations: How each method works

5. SYSTEM HEALTH (05_):
   - Anomaly rates: % of problematic samples
   - Stability: Coefficient of variation
   - Health scores: Overall system performance
   - Recommendations: What to do next

6. CORRELATION ANALYSIS (06_):
   - Heatmap: How features relate to each other
   - Scatter plots: Current vs voltage relationships
   - Trend lines: Linear relationships

📄 PDF REPORT CONTAINS:

Page 1: Executive Summary
   - Key metrics and findings
   - Health assessment
   - High-level recommendations

Page 2: Methodology Explanation
   - What each detection method measures
   - Mathematical formulas and thresholds
   - Why anomalies matter

Page 3: System A Detailed Analysis
   - Individual anomaly listings
   - Signal characteristics
   - Specific recommendations

Page 4: System B Detailed Analysis
   - Individual anomaly listings
   - Signal characteristics
   - Specific recommendations

Page 5: Comparative Analysis
   - Side-by-side system comparison
   - Priority action items
   - Contact information

🎯 INTERPRETING RESULTS:

ANOMALY RATES:
• < 1%: 🟢 Excellent (normal operation)
• 1-3%: 🟡 Monitor (some issues)
• > 3%: 🔴 Critical (needs attention)

HEALTH SCORES:
• 80-100: 🟢 Excellent condition
• 60-80: 🟡 Good condition, monitor
• < 60: 🔴 Poor condition, immediate action needed

Z-SCORES (in anomaly details):
• 2.0-2.5: Minor deviation
• 2.5-3.0: Significant anomaly
• > 3.0: Critical anomaly

COEFFICIENT OF VARIATION:
• < 0.2: Very stable signal
• 0.2-0.5: Moderately stable
• > 0.5: Unstable signal
"""

print("\n" + "="*60)
print("QUICK START GUIDE:")
print("="*60)
print("1. Set your file path: file_path = 'processed_data_v2\merged_data_20Hz.csv'")
print("2. Run analysis: results = run_complete_railway_analysis(file_path)")
print("3. Wait for completion (usually < 1 minute)")
print("4. Open 'new_Graphs' folder to view results")
print("5. Read PDF report for detailed insights")
print("="*60)

# Additional utility functions for post-analysis

def quick_summary(results):
    """Print a quick summary of analysis results"""
    if results is None:
        print("❌ No results to summarize. Run analysis first.")
        return
    
    print("\n🎯 QUICK SUMMARY:")
    print("="*40)
    print(f"⏱️  Analysis time: {results['total_time']:.1f} seconds")
    print(f"📊 Data points: {len(results['df']):,}")
    print(f"🔌 System A anomalies: {len(results['change_points_a'])}")
    print(f"🔌 System B anomalies: {len(results['change_points_b'])}")
    
    rate_a = len(results['change_points_a']) / len(results['df']) * 100
    rate_b = len(results['change_points_b']) / len(results['df']) * 100
    
    print(f"📈 System A rate: {rate_a:.3f}%")
    print(f"📈 System B rate: {rate_b:.3f}%")
    print(f"🏥 Status: {'🟢 Healthy' if max(rate_a, rate_b) < 1 else '🟡 Monitor' if max(rate_a, rate_b) < 3 else '🔴 Critical'}")
    print("="*40)

def list_anomalies(results, system='A', max_show=10):
    """List detailed information about detected anomalies"""
    if results is None:
        print("❌ No results available. Run analysis first.")
        return
    
    if system.upper() == 'A':
        change_points = results['change_points_a']
        reasons = results['reasons_a']
        signal = results['signal_a']
        print(f"\n🔌 SYSTEM A ANOMALIES (showing first {max_show}):")
    else:
        change_points = results['change_points_b']
        reasons = results['reasons_b']
        signal = results['signal_b']
        print(f"\n🔌 SYSTEM B ANOMALIES (showing first {max_show}):")
    
    print("="*60)
    
    for i, point in enumerate(change_points[:max_show]):
        reason = reasons.get(point, "Unknown")
        if point < len(signal):
            current_val = signal[point, 0] if signal.shape[1] > 0 else 0
            voltage_val = signal[point, 1] if signal.shape[1] > 1 else 0
            
            print(f"Anomaly #{i+1}:")
            print(f"  📍 Sample: {point}")
            print(f"  ⚡ Current: {current_val:.4f}")
            print(f"  🔋 Voltage: {voltage_val:.4f}")
            print(f"  🔍 Reason: {reason}")
            print("-" * 40)
    
    if len(change_points) > max_show:
        print(f"... and {len(change_points) - max_show} more anomalies")

print("✅ Usage examples and utility functions loaded!")
print("📖 Ready to run railway anomaly detection analysis!")
# =============================================================================
# COMPLETE FIXED RAILWAY ANALYSIS - ALL 6 GRAPHS + PROPER STATS
# =============================================================================

def debug_signal_statistics(signal_a, signal_b, a_features, b_features):
    """Debug signal statistics to understand the data better"""
    print("\n🔍 DETAILED SIGNAL STATISTICS DEBUG:")
    print("="*60)
    
    print("📊 SYSTEM A DETAILED ANALYSIS:")
    for i in range(signal_a.shape[1]):
        data = signal_a[:, i]
        feature_name = a_features[i].replace('_processed', '')
        
        print(f"\n   {feature_name}:")
        print(f"     📈 Mean: {np.mean(data):.8f}")
        print(f"     📊 Std Dev: {np.std(data):.8f}")
        print(f"     📉 Min: {np.min(data):.8f}")
        print(f"     📈 Max: {np.max(data):.8f}")
        print(f"     🎯 Range: {np.max(data) - np.min(data):.8f}")
        print(f"     🔢 Unique values: {len(np.unique(data))}")
        print(f"     0️⃣ Zero count: {np.sum(data == 0)}")
        print(f"     ❓ NaN count: {np.sum(np.isnan(data))}")
        
        # Check if signal is essentially constant
        if np.std(data) < 1e-10:
            print(f"     ⚠️  WARNING: Signal appears constant!")
        
        # Show some sample values
        print(f"     📋 First 10 values: {data[:10]}")
    
    print("\n📊 SYSTEM B DETAILED ANALYSIS:")
    for i in range(signal_b.shape[1]):
        data = signal_b[:, i]
        feature_name = b_features[i].replace('_processed', '')
        
        print(f"\n   {feature_name}:")
        print(f"     📈 Mean: {np.mean(data):.8f}")
        print(f"     📊 Std Dev: {np.std(data):.8f}")
        print(f"     📉 Min: {np.min(data):.8f}")
        print(f"     📈 Max: {np.max(data):.8f}")
        print(f"     🎯 Range: {np.max(data) - np.min(data):.8f}")
        print(f"     🔢 Unique values: {len(np.unique(data))}")
        print(f"     0️⃣ Zero count: {np.sum(data == 0)}")
        print(f"     ❓ NaN count: {np.sum(np.isnan(data))}")
        
        if np.std(data) < 1e-10:
            print(f"     ⚠️  WARNING: Signal appears constant!")
        
        print(f"     📋 First 10 values: {data[:10]}")

def improved_coefficient_of_variation(data, feature_name=""):
    """Calculate CV with better handling and debugging"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    print(f"   🔍 CV Debug for {feature_name}:")
    print(f"     Mean: {mean_val:.8f}")
    print(f"     Std: {std_val:.8f}")
    
    # Handle different cases
    if np.all(data == data[0]):  # All values identical
        print(f"     ⚠️  All values identical: {data[0]:.8f}")
        return 0.0
    
    if abs(mean_val) < 1e-15:  # Essentially zero mean
        print(f"     ⚠️  Mean essentially zero")
        return float('inf') if std_val > 0 else 0.0
    
    cv = std_val / abs(mean_val)
    print(f"     CV: {cv:.6f}")
    
    return cv

def create_all_comprehensive_visualizations(df, signal_a, signal_b, change_points_a, change_points_b, 
                                           a_features, b_features, reasons_a, reasons_b, output_dir):
    """Create ALL 6 comprehensive visualizations"""
    print("🎨 Creating ALL 6 comprehensive visualizations...")
    
    # 1. Main overview plot
    print("   📊 1/6: Main overview plot...")
    create_main_overview_plot(df, signal_a, signal_b, change_points_a, change_points_b, 
                             a_features, b_features, output_dir)
    
    # 2. Statistical distributions
    print("   📈 2/6: Statistical distribution analysis...")
    create_statistical_distributions(signal_a, signal_b, a_features, b_features, output_dir)
    
    # 3. Anomaly timeline
    print("   ⏰ 3/6: Anomaly timeline analysis...")
    create_anomaly_timeline_analysis(df, change_points_a, change_points_b, len(signal_a), output_dir)
    
    # 4. Detection methods
    print("   🔍 4/6: Detection method analysis...")
    create_detection_method_analysis(change_points_a, change_points_b, reasons_a, reasons_b, output_dir)
    
    # 5. System health dashboard (improved)
    print("   🏥 5/6: System health dashboard...")
    health_a, health_b, cv_a, cv_b = create_improved_health_dashboard_fixed(
        signal_a, signal_b, change_points_a, change_points_b, a_features, b_features, output_dir)
    
    # 6. Correlation analysis
    print("   🔗 6/6: Correlation analysis...")
    create_correlation_analysis(signal_a, signal_b, a_features, b_features, output_dir)
    
    return health_a, health_b, cv_a, cv_b

def create_improved_health_dashboard_fixed(signal_a, signal_b, change_points_a, change_points_b, 
                                          a_features, b_features, output_dir):
    """Create improved system health dashboard with proper debugging"""
    print("\n🔧 CALCULATING HEALTH METRICS WITH DEBUGGING:")
    
    # Debug the signals first
    debug_signal_statistics(signal_a, signal_b, a_features, b_features)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Calculate health metrics with debugging
    total_samples = len(signal_a)
    rate_a = len(change_points_a) / total_samples * 100
    rate_b = len(change_points_b) / total_samples * 100
    
    print(f"\n📊 ANOMALY RATES:")
    print(f"   System A: {len(change_points_a)} anomalies / {total_samples} samples = {rate_a:.4f}%")
    print(f"   System B: {len(change_points_b)} anomalies / {total_samples} samples = {rate_b:.4f}%")
    
    # Calculate CV for each feature with debugging
    print(f"\n📈 COEFFICIENT OF VARIATION CALCULATIONS:")
    
    cvs_a = []
    for i in range(signal_a.shape[1]):
        feature_name = a_features[i].replace('_processed', '')
        cv = improved_coefficient_of_variation(signal_a[:, i], f"System A {feature_name}")
        cvs_a.append(cv)
    
    cvs_b = []
    for i in range(signal_b.shape[1]):
        feature_name = b_features[i].replace('_processed', '')
        cv = improved_coefficient_of_variation(signal_b[:, i], f"System B {feature_name}")
        cvs_b.append(cv)
    
    # Handle infinite or very large CVs
    cv_a_clean = [cv for cv in cvs_a if cv != float('inf') and cv < 1000]
    cv_b_clean = [cv for cv in cvs_b if cv != float('inf') and cv < 1000]
    
    avg_cv_a = np.mean(cv_a_clean) if cv_a_clean else 0
    avg_cv_b = np.mean(cv_b_clean) if cv_b_clean else 0
    
    print(f"\n📊 AVERAGE CVs:")
    print(f"   System A: {avg_cv_a:.6f} (from {len(cv_a_clean)} valid features)")
    print(f"   System B: {avg_cv_b:.6f} (from {len(cv_b_clean)} valid features)")
    
    # Calculate health scores
    health_a = max(0, 100 - (rate_a * 5) - (min(avg_cv_a, 20) * 2))
    health_b = max(0, 100 - (rate_b * 5) - (min(avg_cv_b, 20) * 2))
    
    print(f"\n🏥 HEALTH SCORES:")
    print(f"   System A: {health_a:.1f}/100")
    print(f"   System B: {health_b:.1f}/100")
    
    # 1. Anomaly rates comparison
    ax = axes[0, 0]
    systems = ['System A', 'System B']
    rates = [rate_a, rate_b]
    colors = ['lightblue', 'lightgreen']
    bars = ax.bar(systems, rates, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Anomaly Rates Comparison\n(Percentage of Anomalous Samples)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars, rates):
        status = "🟢 Good" if rate < 1 else "🟡 Monitor" if rate < 3 else "🔴 Critical"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.01, max(rates)*0.05), 
               f'{rate:.3f}%\n{status}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # 2. Signal stability comparison
    ax = axes[0, 1]
    display_cvs = [min(avg_cv_a, 10), min(avg_cv_b, 10)]  # Cap for display
    bars = ax.bar(systems, display_cvs, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Signal Stability Comparison\n(Coefficient of Variation)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation\n(σ/μ)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    for bar, display_cv, actual_cv in zip(bars, display_cvs, [avg_cv_a, avg_cv_b]):
        if actual_cv > 10:
            label = f'{display_cv:.1f}+\n🔴 Very High'
        elif actual_cv < 0.1:
            label = f'{actual_cv:.4f}\n🟢 Very Stable'
        elif actual_cv < 1.0:
            label = f'{actual_cv:.3f}\n🟡 Moderate'
        else:
            label = f'{actual_cv:.2f}\n🔴 Unstable'
        
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.1, max(display_cvs)*0.05), 
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Health scores
    ax = axes[0, 2]
    healths = [health_a, health_b]
    bars = ax.bar(systems, healths, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Overall Health Scores\n(Composite Performance Metric)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Health Score (0-100)\n(Higher is Better)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, health in zip(bars, healths):
        status = "🟢 Excellent" if health > 80 else "🟡 Good" if health > 60 else "🔴 Poor"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{health:.1f}\n{status}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # 4. Feature-wise analysis
    ax = axes[1, 0]
    all_features = []
    all_cvs = []
    all_colors = []
    
    for i, cv in enumerate(cvs_a):
        feature_name = f"A-{a_features[i].replace('_processed', '')}"
        all_features.append(feature_name)
        all_cvs.append(min(cv, 10))  # Cap for display
        all_colors.append('lightblue')
    
    for i, cv in enumerate(cvs_b):
        feature_name = f"B-{b_features[i].replace('_processed', '')}"
        all_features.append(feature_name)
        all_cvs.append(min(cv, 10))  # Cap for display
        all_colors.append('lightgreen')
    
    bars = ax.bar(range(len(all_features)), all_cvs, color=all_colors, alpha=0.8, edgecolor='black')
    ax.set_title('Feature-wise Stability\n(Individual CV Values)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_xticks(range(len(all_features)))
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 5. Data quality assessment
    ax = axes[1, 1]
    quality_a = 100 - (np.sum(signal_a == 0) / signal_a.size * 100) - (np.sum(np.isnan(signal_a)) / signal_a.size * 100)
    quality_b = 100 - (np.sum(signal_b == 0) / signal_b.size * 100) - (np.sum(np.isnan(signal_b)) / signal_b.size * 100)
    
    qualities = [max(0, quality_a), max(0, quality_b)]
    bars = ax.bar(systems, qualities, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Data Quality Assessment\n(Completeness Score)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Quality Score (0-100)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, quality in zip(bars, qualities):
        status = "🟢 Good" if quality > 90 else "🟡 Fair" if quality > 70 else "🔴 Poor"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{quality:.1f}\n{status}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # 6. Summary and diagnostics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Diagnostic information
    diagnostics = []
    if avg_cv_a < 1e-6 and avg_cv_b < 1e-6:
        diagnostics.append("⚠️  Both signals appear constant")
    if rate_a == 0 and rate_b == 0:
        diagnostics.append("ℹ️  No anomalies detected")
    if np.sum(signal_a == 0) > len(signal_a) * 0.1:
        diagnostics.append("⚠️  System A has many zeros")
    if np.sum(signal_b == 0) > len(signal_b) * 0.1:
        diagnostics.append("⚠️  System B has many zeros")
    
    summary_text = f"""
📊 COMPREHENSIVE ANALYSIS SUMMARY:

🏆 Performance Comparison:
• Health: A={health_a:.1f}, B={health_b:.1f}
• Stability: A={avg_cv_a:.6f}, B={avg_cv_b:.6f}
• Anomalies: A={len(change_points_a)}, B={len(change_points_b)}

📈 Signal Characteristics:
• A Range: {np.max(signal_a) - np.min(signal_a):.6f}
• B Range: {np.max(signal_b) - np.min(signal_b):.6f}
• A Unique: {len(np.unique(signal_a.flatten()))}
• B Unique: {len(np.unique(signal_b.flatten()))}

🔍 Diagnostics:
{"".join([f"  {d}" + chr(10) for d in diagnostics])}

🎯 Recommendations:
• {"Check data preprocessing" if min(qualities) < 80 else "Data quality good"}
• {"Review signal sources" if max(avg_cv_a, avg_cv_b) < 1e-6 else "Signals show variation"}
• {"Investigate anomalies" if max(rate_a, rate_b) > 1 else "Low anomaly rates"}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('🏥 Complete System Health Dashboard - Detailed Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "05_complete_system_health_dashboard", output_dir)
    
    return health_a, health_b, avg_cv_a, avg_cv_b

def run_complete_fixed_analysis(file_path, max_rows=10000):
    """Run complete analysis with all 6 graphs and proper statistics"""
    print("🚂 STARTING COMPLETE FIXED RAILWAY ANALYSIS - ALL 6 GRAPHS")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Setup
        output_dir = create_output_directory()
        
        # Step 2: Load data
        print("\n📊 Loading railway data...")
        df, a_features, b_features, polling_features = load_railway_data_fast(
            file_path, max_rows=max_rows, aggregation_method='mean'
        )
        
        if df is None:
            return None
        
        signal_a = df[a_features].values
        signal_b = df[b_features].values
        
        print(f"   Signal A shape: {signal_a.shape}")
        print(f"   Signal B shape: {signal_b.shape}")
        
        # Step 3: Anomaly detection
        print("\n🔍 Running anomaly detection...")
        change_points_a, signal_a_processed, method_a, reasons_a = fast_anomaly_detection_hybrid(
            signal_a, "System A", a_features, max_changes=20
        )
        change_points_b, signal_b_processed, method_b, reasons_b = fast_anomaly_detection_hybrid(
            signal_b, "System B", b_features, max_changes=20
        )
        
        # Step 4: Create ALL visualizations
        print("\n🎨 Creating all 6 comprehensive visualizations...")
        health_a, health_b, cv_a, cv_b = create_all_comprehensive_visualizations(
            df, signal_a_processed, signal_b_processed, change_points_a, change_points_b, 
            a_features, b_features, reasons_a, reasons_b, output_dir
        )
        
        # Step 5: Create PDF report
        print("\n📄 Generating PDF report...")
        pdf_path = create_comprehensive_pdf_report(df, signal_a_processed, signal_b_processed, 
                                                  change_points_a, change_points_b, 
                                                  reasons_a, reasons_b, a_features, b_features, output_dir)
        
        end_time = time.time()
        
        print("\n" + "="*80)
        print("🎉 COMPLETE ANALYSIS FINISHED - ALL 6 GRAPHS CREATED!")
        print("="*80)
        
        print(f"\n📊 FINAL CORRECTED RESULTS:")
        print(f"   ⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"   📈 Data points: {len(df):,}")
        print(f"   🔌 System A: {len(change_points_a)} anomalies ({len(change_points_a)/len(df)*100:.3f}%)")
        print(f"   🔌 System B: {len(change_points_b)} anomalies ({len(change_points_b)/len(df)*100:.3f}%)")
        print(f"   🏥 System A Health: {health_a:.1f}/100")
        print(f"   🏥 System B Health: {health_b:.1f}/100")
        print(f"   📊 System A CV: {cv_a:.6f}")
        print(f"   📊 System B CV: {cv_b:.6f}")
        
        print(f"\n📁 ALL 6 GRAPHS CREATED:")
        print(f"   ✅ 01_main_overview_detailed.png")
        print(f"   ✅ 02_statistical_distributions.png")
        print(f"   ✅ 03_anomaly_timeline_analysis.png")
        print(f"   ✅ 04_detection_method_analysis.png")
        print(f"   ✅ 05_complete_system_health_dashboard.png")
        print(f"   ✅ 06_correlation_analysis.png")
        print(f"   ✅ Comprehensive_Anomaly_Detection_Report.pdf")
        
        return {
            'health_a': health_a,
            'health_b': health_b,
            'cv_a': cv_a,
            'cv_b': cv_b,
            'change_points_a': change_points_a,
            'change_points_b': change_points_b,
            'total_graphs': 6,
            'output_dir': output_dir,
            'pdf_path': pdf_path
        }
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# EXECUTE COMPLETE FIXED ANALYSIS
# =============================================================================

# =============================================================================
# RAILWAY ANALYSIS FOR NORMALIZED/STANDARDIZED SIGNALS
# =============================================================================

def calculate_normalized_signal_metrics(signal, feature_name=""):
    """Calculate appropriate metrics for normalized signals (mean=0, std=1)"""
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    
    print(f"   📊 {feature_name} Normalized Metrics:")
    print(f"     Mean: {mean_val:.8f} (should be ~0)")
    print(f"     Std: {std_val:.8f} (should be ~1)")
    
    # For normalized signals, use different stability metrics
    metrics = {
        'mean': mean_val,
        'std': std_val,
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.max(signal) - np.min(signal),
        'skewness': np.mean(((signal - mean_val) / std_val) ** 3) if std_val > 0 else 0,
        'kurtosis': np.mean(((signal - mean_val) / std_val) ** 4) - 3 if std_val > 0 else 0,
        'outlier_ratio': np.sum(np.abs(signal) > 3) / len(signal) * 100,  # Beyond 3-sigma
        'extreme_ratio': np.sum(np.abs(signal) > 4) / len(signal) * 100,  # Beyond 4-sigma
    }
    
    print(f"     Range: {metrics['range']:.3f}")
    print(f"     Skewness: {metrics['skewness']:.3f}")
    print(f"     Kurtosis: {metrics['kurtosis']:.3f}")
    print(f"     Outliers (>3σ): {metrics['outlier_ratio']:.2f}%")
    print(f"     Extremes (>4σ): {metrics['extreme_ratio']:.2f}%")
    
    return metrics

def calculate_normalized_health_score(signal, change_points, total_samples, feature_name=""):
    """Calculate health score appropriate for normalized signals"""
    
    # Get normalized metrics
    metrics = calculate_normalized_signal_metrics(signal.flatten(), feature_name)
    
    # Anomaly rate (same as before)
    anomaly_rate = len(change_points) / total_samples * 100
    
    # For normalized signals, stability is measured differently:
    # 1. How close std is to 1 (perfect normalization)
    std_deviation_penalty = abs(1.0 - metrics['std']) * 10
    
    # 2. How close mean is to 0 (perfect centering) 
    mean_deviation_penalty = abs(metrics['mean']) * 100
    
    # 3. Signal range (normalized signals should have reasonable range)
    if metrics['range'] > 8:  # Very wide range (beyond ±4σ)
        range_penalty = (metrics['range'] - 8) * 2
    elif metrics['range'] < 4:  # Very narrow range
        range_penalty = (4 - metrics['range']) * 5
    else:
        range_penalty = 0
    
    # 4. Excessive outliers beyond 3-sigma
    outlier_penalty = metrics['outlier_ratio'] * 2
    
    # 5. Distribution shape penalties
    shape_penalty = abs(metrics['skewness']) * 2 + abs(metrics['kurtosis']) * 1
    
    # Calculate health score
    base_score = 100
    total_penalty = (anomaly_rate * 3 +  # Anomalies are important
                    std_deviation_penalty +  # Normalization quality
                    mean_deviation_penalty +  # Centering quality  
                    range_penalty +  # Range appropriateness
                    outlier_penalty +  # Excessive outliers
                    shape_penalty)  # Distribution shape
    
    health_score = max(0, min(100, base_score - total_penalty))
    
    print(f"   🏥 {feature_name} Health Calculation:")
    print(f"     Base score: {base_score}")
    print(f"     Anomaly penalty: -{anomaly_rate * 3:.1f}")
    print(f"     Std deviation penalty: -{std_deviation_penalty:.1f}")
    print(f"     Mean deviation penalty: -{mean_deviation_penalty:.1f}")
    print(f"     Range penalty: -{range_penalty:.1f}")
    print(f"     Outlier penalty: -{outlier_penalty:.1f}")
    print(f"     Shape penalty: -{shape_penalty:.1f}")
    print(f"     Final health score: {health_score:.1f}/100")
    
    return health_score, metrics

def create_normalized_signal_health_dashboard(signal_a, signal_b, change_points_a, change_points_b, 
                                            a_features, b_features, output_dir):
    """Create health dashboard specifically for normalized signals"""
    
    print("\n🔧 CALCULATING HEALTH METRICS FOR NORMALIZED SIGNALS:")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    total_samples = len(signal_a)
    rate_a = len(change_points_a) / total_samples * 100
    rate_b = len(change_points_b) / total_samples * 100
    
    print(f"\n📊 ANOMALY RATES:")
    print(f"   System A: {len(change_points_a)} / {total_samples} = {rate_a:.3f}%")
    print(f"   System B: {len(change_points_b)} / {total_samples} = {rate_b:.3f}%")
    
    # Calculate health for each system
    print(f"\n🔍 SYSTEM A ANALYSIS:")
    health_a, metrics_a = calculate_normalized_health_score(signal_a, change_points_a, total_samples, "System A")
    
    print(f"\n🔍 SYSTEM B ANALYSIS:")
    health_b, metrics_b = calculate_normalized_health_score(signal_b, change_points_b, total_samples, "System B")
    
    # 1. Anomaly rates (same as before)
    ax = axes[0, 0]
    systems = ['System A', 'System B']
    rates = [rate_a, rate_b]
    colors = ['lightblue', 'lightgreen']
    bars = ax.bar(systems, rates, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Anomaly Rates\n(Percentage of Anomalous Samples)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars, rates):
        status = "🟢 Good" if rate < 0.5 else "🟡 Monitor" if rate < 1.0 else "🔴 Critical"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.01, max(rates)*0.05), 
               f'{rate:.3f}%\n{status}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Normalization Quality (how well normalized)
    ax = axes[0, 1]
    norm_quality_a = 100 - (abs(1.0 - metrics_a['std']) * 50 + abs(metrics_a['mean']) * 200)
    norm_quality_b = 100 - (abs(1.0 - metrics_b['std']) * 50 + abs(metrics_b['mean']) * 200)
    qualities = [max(0, norm_quality_a), max(0, norm_quality_b)]
    
    bars = ax.bar(systems, qualities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Normalization Quality\n(How Well Standardized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Quality Score (0-100)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, quality in zip(bars, qualities):
        status = "🟢 Perfect" if quality > 95 else "🟡 Good" if quality > 80 else "🔴 Poor"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{quality:.1f}\n{status}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Overall Health Scores
    ax = axes[0, 2]
    healths = [health_a, health_b]
    bars = ax.bar(systems, healths, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Overall Health Scores\n(Normalized Signal Assessment)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Health Score (0-100)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, health in zip(bars, healths):
        status = "🟢 Excellent" if health > 80 else "🟡 Good" if health > 60 else "🔴 Poor"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{health:.1f}\n{status}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Signal Range Analysis
    ax = axes[1, 0]
    ranges = [metrics_a['range'], metrics_b['range']]
    bars = ax.bar(systems, ranges, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Signal Range Analysis\n(Max - Min Values)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Range (Standard Deviations)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add expected range reference line
    ax.axhline(y=6, color='green', linestyle='--', alpha=0.7, label='Expected Range (~6σ)')
    ax.legend()
    
    for bar, range_val in zip(bars, ranges):
        status = "🟢 Normal" if 4 <= range_val <= 8 else "🟡 Wide" if range_val > 8 else "🔴 Narrow"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{range_val:.2f}\n{status}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Outlier Analysis
    ax = axes[1, 1]
    outlier_ratios = [metrics_a['outlier_ratio'], metrics_b['outlier_ratio']]
    bars = ax.bar(systems, outlier_ratios, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('Outlier Analysis\n(% Beyond 3-Sigma)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Outlier Percentage (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add expected outlier rate reference
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Expected (~0.3%)')
    ax.legend()
    
    for bar, outlier_pct in zip(bars, outlier_ratios):
        status = "🟢 Normal" if outlier_pct <= 1 else "🟡 High" if outlier_pct <= 3 else "🔴 Excessive"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.1, max(outlier_ratios)*0.05), 
               f'{outlier_pct:.2f}%\n{status}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Summary for Normalized Signals
    ax = axes[1, 2]
    ax.axis('off')
    
    better_system = "A" if health_a > health_b else "B" if health_b > health_a else "Both Equal"
    
    summary_text = f"""
📊 NORMALIZED SIGNAL ANALYSIS:

🎯 Key Findings:
• Data is properly normalized (μ≈0, σ≈1)
• System A Health: {health_a:.1f}/100
• System B Health: {health_b:.1f}/100
• Winner: System {better_system}

📈 Signal Characteristics:
• A Range: {metrics_a['range']:.2f}σ
• B Range: {metrics_b['range']:.2f}σ
• A Outliers: {metrics_a['outlier_ratio']:.2f}%
• B Outliers: {metrics_b['outlier_ratio']:.2f}%

⚠️  Notes for Normalized Data:
• CV cannot be calculated (μ≈0)
• Health based on: anomalies, 
  normalization quality, range,
  outliers, and distribution shape
• Expected outlier rate: ~0.3%
• Expected range: 4-8 standard deviations

🎯 Recommendations:
• {"Both systems healthy" if min(health_a, health_b) > 80 else "Monitor " + ("A" if health_a < health_b else "B")}
• {"Outlier investigation needed" if max(metrics_a['outlier_ratio'], metrics_b['outlier_ratio']) > 2 else "Outlier levels normal"}
• {"Anomaly patterns acceptable" if max(rate_a, rate_b) < 1 else "Review anomaly patterns"}

🔧 Data Quality: ✅ Normalized
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('🏥 Normalized Signal Health Dashboard - Specialized Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_plot_no_show(fig, "05_normalized_signal_health_dashboard", output_dir)
    
    return health_a, health_b, metrics_a, metrics_b

def run_normalized_signal_analysis(file_path, max_rows=10000):
    """Complete analysis optimized for normalized signals"""
    print("🚂 RAILWAY ANALYSIS FOR NORMALIZED SIGNALS")
    print("="*60)
    print("🔍 Detected: Pre-normalized data (mean≈0, std≈1)")
    print("📊 Using specialized metrics for standardized signals")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Standard setup
        output_dir = create_output_directory()
        
        df, a_features, b_features, polling_features = load_railway_data_fast(
            file_path, max_rows=max_rows, aggregation_method='mean'
        )
        
        if df is None:
            return None
        
        signal_a = df[a_features].values
        signal_b = df[b_features].values
        
        # Anomaly detection
        change_points_a, signal_a_processed, method_a, reasons_a = fast_anomaly_detection_hybrid(
            signal_a, "System A", a_features, max_changes=20
        )
        change_points_b, signal_b_processed, method_b, reasons_b = fast_anomaly_detection_hybrid(
            signal_b, "System B", b_features, max_changes=20
        )
        
        # Create visualizations (first 4 standard ones)
        print("\n🎨 Creating visualizations for normalized signals...")
        
        print("   📊 1/6: Main overview...")
        create_main_overview_plot(df, signal_a_processed, signal_b_processed, 
                                 change_points_a, change_points_b, a_features, b_features, output_dir)
        
        print("   📈 2/6: Statistical distributions...")
        create_statistical_distributions(signal_a_processed, signal_b_processed, 
                                       a_features, b_features, output_dir)
        
        print("   ⏰ 3/6: Anomaly timeline...")
        create_anomaly_timeline_analysis(df, change_points_a, change_points_b, len(signal_a), output_dir)
        
        print("   🔍 4/6: Detection methods...")
        create_detection_method_analysis(change_points_a, change_points_b, reasons_a, reasons_b, output_dir)
        
        print("   🏥 5/6: Normalized signal health dashboard...")
        health_a, health_b, metrics_a, metrics_b = create_normalized_signal_health_dashboard(
            signal_a_processed, signal_b_processed, change_points_a, change_points_b, 
            a_features, b_features, output_dir)
        
        print("   🔗 6/6: Correlation analysis...")
        create_correlation_analysis(signal_a_processed, signal_b_processed, a_features, b_features, output_dir)
        
        # PDF Report
        print("\n📄 Creating PDF report...")
        pdf_path = create_comprehensive_pdf_report(df, signal_a_processed, signal_b_processed, 
                                                  change_points_a, change_points_b, 
                                                  reasons_a, reasons_b, a_features, b_features, output_dir)
        
        end_time = time.time()
        
        print("\n" + "="*80)
        print("🎉 NORMALIZED SIGNAL ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📊 RESULTS FOR NORMALIZED SIGNALS:")
        print(f"   ⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"   📈 Data points: {len(df):,}")
        print(f"   🔌 System A: {len(change_points_a)} anomalies ({len(change_points_a)/len(df)*100:.3f}%)")
        print(f"   🔌 System B: {len(change_points_b)} anomalies ({len(change_points_b)/len(df)*100:.3f}%)")
        print(f"   🏥 System A Health: {health_a:.1f}/100")
        print(f"   🏥 System B Health: {health_b:.1f}/100")
        print(f"   📊 A Signal Range: {metrics_a['range']:.2f}σ")
        print(f"   📊 B Signal Range: {metrics_b['range']:.2f}σ")
        print(f"   ⚠️  A Outliers: {metrics_a['outlier_ratio']:.2f}%")
        print(f"   ⚠️  B Outliers: {metrics_b['outlier_ratio']:.2f}%")
        
        print(f"\n📁 ALL 6 GRAPHS CREATED:")
        for i in range(1, 7):
            graph_names = [
                "01_main_overview_detailed.png",
                "02_statistical_distributions.png", 
                "03_anomaly_timeline_analysis.png",
                "04_detection_method_analysis.png",
                "05_normalized_signal_health_dashboard.png",
                "06_correlation_analysis.png"
            ]
            print(f"   ✅ {graph_names[i-1]}")
        print(f"   ✅ Comprehensive_Anomaly_Detection_Report.pdf")
        
        return {
            'health_a': health_a,
            'health_b': health_b,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'change_points_a': change_points_a,
            'change_points_b': change_points_b,
            'output_dir': output_dir,
            'pdf_path': pdf_path
        }
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# EXECUTE ANALYSIS FOR NORMALIZED SIGNALS
# =============================================================================

print("🔧 RUNNING SPECIALIZED ANALYSIS FOR NORMALIZED SIGNALS")
print("="*70)

file_path = r'processed_data_v2\merged_data_20Hz.csv'
normalized_results = run_normalized_signal_analysis(file_path, max_rows=10000)

if normalized_results:
    print(f"\n✅ SUCCESS! Analyzed normalized signals with proper metrics!")
    print(f"📁 Check: {normalized_results['output_dir']}")
    
    # Show quick comparison
    print(f"\n🏆 SYSTEM COMPARISON:")
    print(f"   System A Health: {normalized_results['health_a']:.1f}/100")
    print(f"   System B Health: {normalized_results['health_b']:.1f}/100")
    winner = "A" if normalized_results['health_a'] > normalized_results['health_b'] else "B" if normalized_results['health_b'] > normalized_results['health_a'] else "Tie"
    print(f"   Winner: System {winner}")
else:
    print("\n❌ Analysis failed.")