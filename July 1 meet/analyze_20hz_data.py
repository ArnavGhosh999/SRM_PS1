import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ast
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis
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

def load_sample_data(file_path, sample_size=1000):
    """Load a sample of data for analysis"""
    print(f"Loading sample of {sample_size} rows from {file_path}...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Parse array columns
    features = ['A Current', 'A Voltage', 'B Current', 'B Voltage']
    for feat in features:
        print(f"Parsing {feat}...")
        df[feat] = parse_array_column(df[feat])
    
    return df, features

def extract_basic_stats(arr):
    """Extract basic statistics from array"""
    if len(arr) == 0:
        return [0, 0, 0, 0, 0, 0]
    
    return [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        skew(arr) if len(arr) > 2 else 0,
        kurtosis(arr) if len(arr) > 2 else 0
    ]

def create_time_series_plots(df, features, save_dir='.'):
    """Create time series plots for sample data"""
    print("Creating time series plots...")
    
    # Select a few representative samples
    sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        for idx in sample_indices:
            arr = df[feature].iloc[idx]
            if len(arr) > 0:
                ax.plot(arr, alpha=0.7, linewidth=1)
        
        ax.set_title(f'{feature} - Sample Time Series')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/time_series_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Time series plots saved as 'time_series_samples.png'")

def create_statistical_analysis(df, features, save_dir='.'):
    """Create statistical analysis plots"""
    print("Creating statistical analysis plots...")
    
    # Extract statistics for each feature
    stats_data = {}
    for feature in features:
        print(f"Processing statistics for {feature}...")
        stats = []
        for arr in df[feature]:
            if len(arr) > 0:
                stats.append(extract_basic_stats(arr))
        
        if stats:
            stats_data[feature] = pd.DataFrame(stats, 
                                             columns=['Mean', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis'])
    
    # Create subplots for each statistic
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    stat_names = ['Mean', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis']
    
    for i, stat in enumerate(stat_names):
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
    plt.savefig(f'{save_dir}/statistical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Statistical analysis saved as 'statistical_distributions.png'")

def create_clustering_analysis(df, features, save_dir='.'):
    """Create clustering analysis for time series patterns"""
    print("Creating clustering analysis...")
    
    for feature in features:
        print(f"Clustering {feature}...")
        
        # Filter arrays with sufficient length
        arr_lengths = [len(x) for x in df[feature]]
        min_length = 50
        valid_indices = [i for i, length in enumerate(arr_lengths) if length >= min_length]
        
        if len(valid_indices) < 10:
            print(f"Not enough valid samples for {feature}")
            continue
        
        # Get valid arrays
        valid_arrays = [df[feature].iloc[i] for i in valid_indices[:100]]  # Limit to 100 samples
        
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
        pca = PCA(n_components=min(5, feature_matrix_scaled.shape[1]))
        reduced_features = pca.fit_transform(feature_matrix_scaled)
        
        # Clustering
        n_clusters = min(5, len(reduced_features))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=32)
        clusters = kmeans.fit_predict(reduced_features)
        
        # Plot clusters
        fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))
        if n_clusters == 1:
            axes = [axes]
        
        for i in range(n_clusters):
            ax = axes[i]
            cluster_indices = np.where(clusters == i)[0]
            
            if len(cluster_indices) > 0:
                cluster_arrays = [padded_arrays[j] for j in cluster_indices]
                
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
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{feature.replace(" ", "_")}_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Clustering for {feature} saved")

def create_summary_statistics(df, features, save_dir='.'):
    """Create summary statistics table and plots"""
    print("Creating summary statistics...")
    
    summary_stats = {}
    
    for feature in features:
        print(f"Calculating summary for {feature}...")
        
        # Calculate array lengths
        lengths = [len(arr) for arr in df[feature]]
        non_empty = [l for l in lengths if l > 0]
        
        if non_empty:
            summary_stats[feature] = {
                'Total_Records': len(df),
                'Non_Empty_Arrays': len(non_empty),
                'Empty_Arrays': len(lengths) - len(non_empty),
                'Min_Length': min(non_empty),
                'Max_Length': max(non_empty),
                'Mean_Length': np.mean(non_empty),
                'Std_Length': np.std(non_empty)
            }
    
    # Create summary plot
    if summary_stats:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Array length distribution
        ax1 = axes[0, 0]
        for feature in features:
            if feature in summary_stats:
                lengths = [len(arr) for arr in df[feature] if len(arr) > 0]
                ax1.hist(lengths, alpha=0.7, label=feature, bins=30)
        ax1.set_title('Array Length Distribution')
        ax1.set_xlabel('Array Length')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Non-empty vs empty arrays
        ax2 = axes[0, 1]
        features_list = list(summary_stats.keys())
        non_empty_counts = [summary_stats[f]['Non_Empty_Arrays'] for f in features_list]
        empty_counts = [summary_stats[f]['Empty_Arrays'] for f in features_list]
        
        x = np.arange(len(features_list))
        width = 0.35
        
        ax2.bar(x - width/2, non_empty_counts, width, label='Non-Empty', alpha=0.8)
        ax2.bar(x + width/2, empty_counts, width, label='Empty', alpha=0.8)
        ax2.set_title('Array Completeness')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(features_list, rotation=45)
        ax2.legend()
        
        # Mean values comparison
        ax3 = axes[1, 0]
        mean_values = []
        for feature in features:
            if feature in summary_stats:
                values = []
                for arr in df[feature]:
                    if len(arr) > 0:
                        values.extend(arr)
                if values:
                    mean_values.append(np.mean(values))
                else:
                    mean_values.append(0)
            else:
                mean_values.append(0)
        
        ax3.bar(features, mean_values, alpha=0.8)
        ax3.set_title('Mean Values Across All Data')
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Mean Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # Data quality heatmap
        ax4 = axes[1, 1]
        quality_data = []
        for feature in features:
            if feature in summary_stats:
                quality_data.append([
                    summary_stats[feature]['Non_Empty_Arrays'] / summary_stats[feature]['Total_Records'],
                    summary_stats[feature]['Mean_Length'] / summary_stats[feature]['Max_Length']
                ])
            else:
                quality_data.append([0, 0])
        
        im = ax4.imshow(quality_data, cmap='viridis', aspect='auto')
        ax4.set_title('Data Quality Heatmap')
        ax4.set_xlabel('Completeness | Length Ratio')
        ax4.set_ylabel('Features')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Completeness', 'Length Ratio'])
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels(features)
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Summary statistics saved as 'summary_statistics.png'")

def main():
    file_path = 'merged_data_20Hz.csv'
    
    # Load sample data
    df, features = load_sample_data(file_path, sample_size=1000)
    
    print(f"Loaded {len(df)} rows with features: {features}")
    print(f"Data shape: {df.shape}")
    
    # Create output directory
    import os
    output_dir = 'analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    create_time_series_plots(df, features, output_dir)
    create_statistical_analysis(df, features, output_dir)
    create_clustering_analysis(df, features, output_dir)
    create_summary_statistics(df, features, output_dir)
    
    print(f"\nAll analysis complete! Check the '{output_dir}' directory for generated plots.")
    print("Generated files:")
    print("- time_series_samples.png")
    print("- statistical_distributions.png")
    print("- [Feature]_clusters.png (for each feature)")
    print("- summary_statistics.png")

if __name__ == "__main__":
    main() 