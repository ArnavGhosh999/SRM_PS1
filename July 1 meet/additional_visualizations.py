import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_array_string(x):
    """Parse array string to numpy array"""
    if isinstance(x, str):
        try:
            return np.fromiter((float(i) for i in x.split(',')), dtype=float)
        except Exception:
            return np.array([])
    else:
        return np.array([])

def parse_array_column(col):
    """Parse array column using ThreadPoolExecutor for speed"""
    with ThreadPoolExecutor() as executor:
        return list(executor.map(parse_array_string, col))

def load_sample_data(file_path, sample_size=500):
    """Load a sample of data for analysis"""
    print(f"Loading sample of {sample_size} rows from {file_path}...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Parse array columns
    features = ['A Current', 'A Voltage', 'B Current', 'B Voltage']
    for feat in features:
        print(f"Parsing {feat}...")
        df[feat] = parse_array_column(df[feat])
    
    return df, features

def create_correlation_analysis(df, features, save_dir='.'):
    """Create correlation analysis between different measurements"""
    print("Creating correlation analysis...")
    
    # Extract mean values for correlation analysis
    correlation_data = {}
    for feature in features:
        means = []
        for arr in df[feature]:
            if len(arr) > 0:
                means.append(np.mean(arr))
            else:
                means.append(np.nan)
        correlation_data[feature] = means
    
    corr_df = pd.DataFrame(correlation_data)
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Electrical Measurements')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features[i+1:], i+1):
            if plot_idx < 4:
                ax = axes[plot_idx]
                
                # Get valid data points
                valid_mask = ~(np.isnan(correlation_data[feat1]) | np.isnan(correlation_data[feat2]))
                x_data = np.array(correlation_data[feat1])[valid_mask]
                y_data = np.array(correlation_data[feat2])[valid_mask]
                
                if len(x_data) > 0:
                    ax.scatter(x_data, y_data, alpha=0.6, s=20)
                    ax.set_xlabel(feat1)
                    ax.set_ylabel(feat2)
                    ax.set_title(f'{feat1} vs {feat2}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add correlation coefficient
                    if len(x_data) > 1:
                        corr_coef, p_value = pearsonr(x_data, y_data)
                        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                               transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Correlation analysis saved")

def create_power_analysis(df, features, save_dir='.'):
    """Create power analysis (P = V * I)"""
    print("Creating power analysis...")
    
    # Calculate power for each measurement
    power_data = {}
    
    # A Power = A Voltage * A Current
    a_power = []
    for i in range(len(df)):
        a_voltage = df['A Voltage'].iloc[i]
        a_current = df['A Current'].iloc[i]
        
        if len(a_voltage) > 0 and len(a_current) > 0:
            # Pad to same length
            min_len = min(len(a_voltage), len(a_current))
            power = a_voltage[:min_len] * a_current[:min_len]
            a_power.append(power)
        else:
            a_power.append(np.array([]))
    
    # B Power = B Voltage * B Current
    b_power = []
    for i in range(len(df)):
        b_voltage = df['B Voltage'].iloc[i]
        b_current = df['B Current'].iloc[i]
        
        if len(b_voltage) > 0 and len(b_current) > 0:
            # Pad to same length
            min_len = min(len(b_voltage), len(b_current))
            power = b_voltage[:min_len] * b_current[:min_len]
            b_power.append(power)
        else:
            b_power.append(np.array([]))
    
    power_data['A Power'] = a_power
    power_data['B Power'] = b_power
    
    # Create power analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Power time series samples
    ax1 = axes[0, 0]
    sample_indices = np.random.choice(len(df), min(3, len(df)), replace=False)
    for idx in sample_indices:
        power_series = a_power[idx]
        if len(power_series) > 0:
            ax1.plot(power_series, alpha=0.7, linewidth=1, label=f'Sample {idx}')
    ax1.set_title('A Power - Sample Time Series')
    ax1.set_xlabel('Time Points')
    ax1.set_ylabel('Power (W)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for idx in sample_indices:
        power_series = b_power[idx]
        if len(power_series) > 0:
            ax2.plot(power_series, alpha=0.7, linewidth=1, label=f'Sample {idx}')
    ax2.set_title('B Power - Sample Time Series')
    ax2.set_xlabel('Time Points')
    ax2.set_ylabel('Power (W)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Power distribution
    ax3 = axes[1, 0]
    a_power_means = [np.mean(p) for p in a_power if len(p) > 0]
    b_power_means = [np.mean(p) for p in b_power if len(p) > 0]
    
    ax3.hist(a_power_means, alpha=0.7, label='A Power', bins=30)
    ax3.hist(b_power_means, alpha=0.7, label='B Power', bins=30)
    ax3.set_title('Power Distribution')
    ax3.set_xlabel('Mean Power (W)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Power comparison
    ax4 = axes[1, 1]
    power_comparison = []
    labels = []
    
    for i in range(len(df)):
        a_p = a_power[i]
        b_p = b_power[i]
        
        if len(a_p) > 0 and len(b_p) > 0:
            min_len = min(len(a_p), len(b_p))
            a_mean = np.mean(a_p[:min_len])
            b_mean = np.mean(b_p[:min_len])
            power_comparison.append([a_mean, b_mean])
            labels.append(f'Record {i}')
    
    if power_comparison:
        power_comparison = np.array(power_comparison)
        ax4.scatter(power_comparison[:, 0], power_comparison[:, 1], alpha=0.6)
        ax4.plot([0, max(power_comparison[:, 0].max(), power_comparison[:, 1].max())], 
                [0, max(power_comparison[:, 0].max(), power_comparison[:, 1].max())], 
                'r--', alpha=0.5, label='Equal Power Line')
        ax4.set_xlabel('A Power (W)')
        ax4.set_ylabel('B Power (W)')
        ax4.set_title('A Power vs B Power')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/power_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Power analysis saved")

def create_trend_analysis(df, features, save_dir='.'):
    """Create trend analysis over time"""
    print("Creating trend analysis...")
    
    # Convert Time column to datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Sort by time
    df_sorted = df.sort_values('Time')
    
    # Calculate trends for each feature
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Calculate mean values over time
        times = []
        means = []
        stds = []
        
        for idx, row in df_sorted.iterrows():
            arr = row[feature]
            if len(arr) > 0:
                times.append(row['Time'])
                means.append(np.mean(arr))
                stds.append(np.std(arr))
        
        if times:
            times = pd.to_datetime(times)
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot with error bars
            ax.errorbar(times, means, yerr=stds, fmt='o-', alpha=0.7, markersize=3, capsize=2)
            ax.set_title(f'{feature} - Trend Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Mean Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(means) > 1:
                z = np.polyfit(range(len(means)), means, 1)
                p = np.poly1d(z)
                ax.plot(times, p(range(len(means))), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Trend analysis saved")

def create_direction_analysis(df, features, save_dir='.'):
    """Analyze data by direction (Normal vs Reverse)"""
    print("Creating direction analysis...")
    
    # Group by direction
    normal_data = df[df['Direction'] == 'Normal']
    reverse_data = df[df['Direction'] == 'Reverse']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Extract means for each direction
        normal_means = []
        reverse_means = []
        
        for arr in normal_data[feature]:
            if len(arr) > 0:
                normal_means.append(np.mean(arr))
        
        for arr in reverse_data[feature]:
            if len(arr) > 0:
                reverse_means.append(np.mean(arr))
        
        # Create box plot
        data_to_plot = [normal_means, reverse_means]
        labels = ['Normal', 'Reverse']
        
        if any(len(d) > 0 for d in data_to_plot):
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f'{feature} by Direction')
            ax.set_ylabel('Mean Value')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/direction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Direction analysis saved")

def main():
    file_path = 'merged_data_20Hz.csv'
    
    # Load sample data
    df, features = load_sample_data(file_path, sample_size=500)
    
    print(f"Loaded {len(df)} rows with features: {features}")
    
    # Create output directory
    import os
    output_dir = 'additional_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate additional visualizations
    create_correlation_analysis(df, features, output_dir)
    create_power_analysis(df, features, output_dir)
    create_trend_analysis(df, features, output_dir)
    create_direction_analysis(df, features, output_dir)
    
    print(f"\nAdditional analysis complete! Check the '{output_dir}' directory for generated plots.")
    print("Generated files:")
    print("- correlation_heatmap.png")
    print("- correlation_scatter_plots.png")
    print("- power_analysis.png")
    print("- trend_analysis.png")
    print("- direction_analysis.png")

if __name__ == "__main__":
    main() 