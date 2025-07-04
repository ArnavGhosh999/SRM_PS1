import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis, percentileofscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
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

def load_sample_data(file_path, sample_percentage=2.0):
    """Load a percentage of data for analysis"""
    # Calculate number of rows for 2% of data
    total_rows = 1678506  # Known total rows
    sample_size = int(total_rows * sample_percentage / 100)
    
    print(f"Loading {sample_percentage}% of data ({sample_size:,} rows) from {file_path}...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Parse array columns
    features = ['A Current', 'A Voltage', 'B Current', 'B Voltage']
    for feat in features:
        print(f"Parsing {feat}...")
        df[feat] = parse_array_column(df[feat])
    
    return df, features

def extract_comprehensive_stats(arr):
    """Extract comprehensive statistics from array"""
    if len(arr) == 0:
        return [0] * 15  # 15 different statistical measures
    
    # Basic statistics
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    median_val = np.median(arr)
    
    # Higher order moments
    skewness = skew(arr) if len(arr) > 2 else 0
    kurtosis_val = kurtosis(arr) if len(arr) > 2 else 0
    
    # Percentiles
    p25 = np.percentile(arr, 25)
    p75 = np.percentile(arr, 75)
    p90 = np.percentile(arr, 90)
    p95 = np.percentile(arr, 95)
    
    # Range and IQR
    range_val = max_val - min_val
    iqr = p75 - p25
    
    # Coefficient of variation
    cv = std_val / mean_val if mean_val != 0 else 0
    
    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(arr**2))
    
    # Peak to peak
    peak_to_peak = max_val - min_val
    
    return [mean_val, std_val, min_val, max_val, median_val, 
            skewness, kurtosis_val, p25, p75, p90, p95, 
            range_val, iqr, cv, rms, peak_to_peak]

def create_comprehensive_clustering(df, features, save_dir='.'):
    """Create comprehensive clustering analysis with detailed statistics"""
    print("Creating comprehensive clustering analysis...")
    
    for feature in features:
        print(f"Processing comprehensive clustering for {feature}...")
        
        # Filter arrays with sufficient length
        arr_lengths = [len(x) for x in df[feature]]
        min_length = 50
        valid_indices = [i for i, length in enumerate(arr_lengths) if length >= min_length]
        
        if len(valid_indices) < 20:
            print(f"Not enough valid samples for {feature}")
            continue
        
        # Get valid arrays (limit to 200 for performance)
        valid_arrays = [df[feature].iloc[i] for i in valid_indices[:200]]
        
        # Pad arrays to same length
        max_len = max(len(arr) for arr in valid_arrays)
        padded_arrays = []
        for arr in valid_arrays:
            if len(arr) < max_len:
                padded = np.pad(arr, (0, max_len - len(arr)), 'constant')
            else:
                padded = arr[:max_len]
            padded_arrays.append(padded)
        
        # Convert to feature matrix
        feature_matrix = np.array(padded_arrays)
        
        # Standardize
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(10, feature_matrix_scaled.shape[1]))
        reduced_features = pca.fit_transform(feature_matrix_scaled)
        
        # Clustering
        n_clusters = min(6, len(reduced_features))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=64)
        clusters = kmeans.fit_predict(reduced_features)
        
        # Extract comprehensive statistics for each cluster
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) > 0:
                cluster_arrays = [padded_arrays[j] for j in cluster_indices]
                cluster_stats[i] = extract_comprehensive_stats(np.concatenate(cluster_arrays))
        
        # Create comprehensive visualization
        create_cluster_comprehensive_plots(feature, cluster_stats, padded_arrays, clusters, save_dir)

def create_cluster_comprehensive_plots(feature, cluster_stats, arrays, clusters, save_dir):
    """Create comprehensive plots for each cluster"""
    
    # Statistical measures names
    stat_names = ['Mean', 'Std', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis', 
                  'P25', 'P75', 'P90', 'P95', 'Range', 'IQR', 'CV', 'RMS', 'Peak-to-Peak']
    
    n_clusters = len(cluster_stats)
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Time series patterns for each cluster
    for i in range(n_clusters):
        ax = plt.subplot(4, 4, i+1)
        cluster_indices = np.where(clusters == i)[0]
        
        if len(cluster_indices) > 0:
            cluster_arrays = [arrays[j] for j in cluster_indices]
            
            # Plot mean and std
            mean_series = np.mean(cluster_arrays, axis=0)
            std_series = np.std(cluster_arrays, axis=0)
            
            ax.plot(mean_series, 'b-', linewidth=2, label='Mean')
            ax.fill_between(range(len(mean_series)), 
                          mean_series - std_series, 
                          mean_series + std_series, 
                          alpha=0.3, color='blue')
            
            ax.set_title(f'Cluster {i} ({len(cluster_indices)} samples)')
            ax.set_xlabel('Time Points')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 2. Statistical comparison across clusters
    # Mean comparison
    ax1 = plt.subplot(4, 4, 5)
    means = [cluster_stats[i][0] for i in range(n_clusters)]
    ax1.bar(range(n_clusters), means, alpha=0.8)
    ax1.set_title('Mean Values by Cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Mean')
    ax1.set_xticks(range(n_clusters))
    
    # Standard deviation comparison
    ax2 = plt.subplot(4, 4, 6)
    stds = [cluster_stats[i][1] for i in range(n_clusters)]
    ax2.bar(range(n_clusters), stds, alpha=0.8, color='orange')
    ax2.set_title('Standard Deviation by Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Std Dev')
    ax2.set_xticks(range(n_clusters))
    
    # Range comparison
    ax3 = plt.subplot(4, 4, 7)
    ranges = [cluster_stats[i][12] for i in range(n_clusters)]
    ax3.bar(range(n_clusters), ranges, alpha=0.8, color='green')
    ax3.set_title('Range by Cluster')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Range')
    ax3.set_xticks(range(n_clusters))
    
    # RMS comparison
    ax4 = plt.subplot(4, 4, 8)
    rms_vals = [cluster_stats[i][14] for i in range(n_clusters)]
    ax4.bar(range(n_clusters), rms_vals, alpha=0.8, color='red')
    ax4.set_title('RMS by Cluster')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('RMS')
    ax4.set_xticks(range(n_clusters))
    
    # 3. Statistical heatmap
    ax5 = plt.subplot(4, 4, 9)
    stats_matrix = np.array([cluster_stats[i] for i in range(n_clusters)])
    im = ax5.imshow(stats_matrix, cmap='viridis', aspect='auto')
    ax5.set_title('Statistical Measures Heatmap')
    ax5.set_xlabel('Statistical Measures')
    ax5.set_ylabel('Cluster')
    ax5.set_xticks(range(len(stat_names)))
    ax5.set_xticklabels(stat_names, rotation=45, ha='right')
    ax5.set_yticks(range(n_clusters))
    plt.colorbar(im, ax=ax5)
    
    # 4. Distribution comparison
    ax6 = plt.subplot(4, 4, 10)
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            cluster_arrays = [arrays[j] for j in cluster_indices]
            all_values = np.concatenate(cluster_arrays)
            ax6.hist(all_values, alpha=0.6, label=f'Cluster {i}', bins=30)
    ax6.set_title('Value Distribution by Cluster')
    ax6.set_xlabel('Value')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    
    # 5. Percentile comparison
    ax7 = plt.subplot(4, 4, 11)
    percentiles = ['P25', 'P75', 'P90', 'P95']
    p_indices = [7, 8, 9, 10]  # Indices for percentiles
    
    x = np.arange(len(percentiles))
    width = 0.8 / n_clusters
    
    for i in range(n_clusters):
        p_values = [cluster_stats[i][j] for j in p_indices]
        ax7.bar(x + i*width, p_values, width, label=f'Cluster {i}', alpha=0.8)
    
    ax7.set_title('Percentile Comparison')
    ax7.set_xlabel('Percentile')
    ax7.set_ylabel('Value')
    ax7.set_xticks(x + width * (n_clusters-1) / 2)
    ax7.set_xticklabels(percentiles)
    ax7.legend()
    
    # 6. Skewness and Kurtosis
    ax8 = plt.subplot(4, 4, 12)
    skewness_vals = [cluster_stats[i][5] for i in range(n_clusters)]
    kurtosis_vals = [cluster_stats[i][6] for i in range(n_clusters)]
    
    x_pos = np.arange(n_clusters)
    width = 0.35
    
    ax8.bar(x_pos - width/2, skewness_vals, width, label='Skewness', alpha=0.8)
    ax8.bar(x_pos + width/2, kurtosis_vals, width, label='Kurtosis', alpha=0.8)
    ax8.set_title('Skewness and Kurtosis by Cluster')
    ax8.set_xlabel('Cluster')
    ax8.set_ylabel('Value')
    ax8.set_xticks(x_pos)
    ax8.legend()
    
    # 7. Coefficient of Variation
    ax9 = plt.subplot(4, 4, 13)
    cv_vals = [cluster_stats[i][13] for i in range(n_clusters)]
    ax9.bar(range(n_clusters), cv_vals, alpha=0.8, color='purple')
    ax9.set_title('Coefficient of Variation by Cluster')
    ax9.set_xlabel('Cluster')
    ax9.set_ylabel('CV')
    ax9.set_xticks(range(n_clusters))
    
    # 8. Peak-to-Peak comparison
    ax10 = plt.subplot(4, 4, 14)
    ptp_vals = [cluster_stats[i][15] for i in range(n_clusters)]
    ax10.bar(range(n_clusters), ptp_vals, alpha=0.8, color='brown')
    ax10.set_title('Peak-to-Peak by Cluster')
    ax10.set_xlabel('Cluster')
    ax10.set_ylabel('Peak-to-Peak')
    ax10.set_xticks(range(n_clusters))
    
    # 9. IQR comparison
    ax11 = plt.subplot(4, 4, 15)
    iqr_vals = [cluster_stats[i][12] for i in range(n_clusters)]
    ax11.bar(range(n_clusters), iqr_vals, alpha=0.8, color='teal')
    ax11.set_title('Interquartile Range by Cluster')
    ax11.set_xlabel('Cluster')
    ax11.set_ylabel('IQR')
    ax11.set_xticks(range(n_clusters))
    
    # 10. Cluster size distribution
    ax12 = plt.subplot(4, 4, 16)
    cluster_sizes = [len(np.where(clusters == i)[0]) for i in range(n_clusters)]
    ax12.pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(n_clusters)], autopct='%1.1f%%')
    ax12.set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{feature.replace(" ", "_")}_comprehensive_clusters.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive clustering for {feature} saved")

def create_detailed_statistical_analysis(df, features, save_dir='.'):
    """Create detailed statistical analysis with 10+ factors"""
    print("Creating detailed statistical analysis...")
    
    # Extract comprehensive statistics for each feature
    stats_data = {}
    for feature in features:
        print(f"Processing detailed statistics for {feature}...")
        stats = []
        for arr in df[feature]:
            if len(arr) > 0:
                stats.append(extract_comprehensive_stats(arr))
        
        if stats:
            stat_names = ['Mean', 'Std', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis', 
                         'P25', 'P75', 'P90', 'P95', 'Range', 'IQR', 'CV', 'RMS', 'Peak-to-Peak']
            stats_data[feature] = pd.DataFrame(stats, columns=stat_names)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    axes = axes.flatten()
    
    stat_names = ['Mean', 'Std', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis', 
                  'P25', 'P75', 'P90', 'P95', 'Range', 'IQR', 'CV', 'RMS', 'Peak-to-Peak']
    
    for i, stat in enumerate(stat_names[:16]):  # Show first 16 statistics
        ax = axes[i]
        data_to_plot = []
        labels = []
        
        for feature in features:
            if feature in stats_data:
                data_to_plot.append(stats_data[feature][stat].dropna())
                labels.append(feature)
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f'{stat} Distribution')
            ax.set_ylabel(stat)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/detailed_statistical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Detailed statistical analysis saved")

def create_feature_comparison_analysis(df, features, save_dir='.'):
    """Create comprehensive feature comparison analysis"""
    print("Creating feature comparison analysis...")
    
    # Extract statistics for comparison
    comparison_data = {}
    for feature in features:
        means = []
        stds = []
        ranges = []
        rms_vals = []
        cv_vals = []
        
        for arr in df[feature]:
            if len(arr) > 0:
                means.append(np.mean(arr))
                stds.append(np.std(arr))
                ranges.append(np.max(arr) - np.min(arr))
                rms_vals.append(np.sqrt(np.mean(arr**2)))
                cv_vals.append(np.std(arr) / np.mean(arr) if np.mean(arr) != 0 else 0)
        
        comparison_data[feature] = {
            'Mean': means,
            'Std': stds,
            'Range': ranges,
            'RMS': rms_vals,
            'CV': cv_vals
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Mean comparison
    ax1 = axes[0, 0]
    for feature in features:
        ax1.hist(comparison_data[feature]['Mean'], alpha=0.7, label=feature, bins=30)
    ax1.set_title('Mean Values Distribution')
    ax1.set_xlabel('Mean Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 2. Standard deviation comparison
    ax2 = axes[0, 1]
    for feature in features:
        ax2.hist(comparison_data[feature]['Std'], alpha=0.7, label=feature, bins=30)
    ax2.set_title('Standard Deviation Distribution')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Range comparison
    ax3 = axes[0, 2]
    for feature in features:
        ax3.hist(comparison_data[feature]['Range'], alpha=0.7, label=feature, bins=30)
    ax3.set_title('Range Distribution')
    ax3.set_xlabel('Range')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. RMS comparison
    ax4 = axes[1, 0]
    for feature in features:
        ax4.hist(comparison_data[feature]['RMS'], alpha=0.7, label=feature, bins=30)
    ax4.set_title('RMS Distribution')
    ax4.set_xlabel('RMS')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    # 5. Coefficient of variation comparison
    ax5 = axes[1, 1]
    for feature in features:
        ax5.hist(comparison_data[feature]['CV'], alpha=0.7, label=feature, bins=30)
    ax5.set_title('Coefficient of Variation Distribution')
    ax5.set_xlabel('CV')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    
    # 6. Statistical summary table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for feature in features:
        summary_data.append([
            feature,
            f"{np.mean(comparison_data[feature]['Mean']):.2f}",
            f"{np.mean(comparison_data[feature]['Std']):.2f}",
            f"{np.mean(comparison_data[feature]['Range']):.2f}",
            f"{np.mean(comparison_data[feature]['RMS']):.2f}",
            f"{np.mean(comparison_data[feature]['CV']):.2f}"
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Feature', 'Mean', 'Std', 'Range', 'RMS', 'CV'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax6.set_title('Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature comparison analysis saved")

def main():
    file_path = 'merged_data_20Hz.csv'
    
    # Load 2% of data
    df, features = load_sample_data(file_path, sample_percentage=2.0)
    
    print(f"Loaded {len(df):,} rows with features: {features}")
    print(f"Data shape: {df.shape}")
    
    # Create output directory
    import os
    output_dir = 'comprehensive_analysis_2percent'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive analysis
    create_comprehensive_clustering(df, features, output_dir)
    create_detailed_statistical_analysis(df, features, output_dir)
    create_feature_comparison_analysis(df, features, output_dir)
    
    print(f"\nComprehensive analysis complete! Check the '{output_dir}' directory for generated plots.")
    print("Generated files:")
    print("- [Feature]_comprehensive_clusters.png (for each feature)")
    print("- detailed_statistical_distributions.png")
    print("- feature_comparison_analysis.png")
    print(f"\nAnalysis performed on {len(df):,} rows ({2.0}% of total dataset)")

if __name__ == "__main__":
    main() 