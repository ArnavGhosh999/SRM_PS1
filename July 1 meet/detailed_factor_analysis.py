import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis, percentileofscore
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

def extract_detailed_stats(arr):
    """Extract detailed statistics from array"""
    if len(arr) == 0:
        return {}
    
    stats = {}
    
    # 1. Basic Statistics
    stats['Mean'] = np.mean(arr)
    stats['Median'] = np.median(arr)
    
    # Safer mode calculation
    try:
        rounded_arr = np.round(arr, 2)
        unique, counts = np.unique(rounded_arr, return_counts=True)
        mode_idx = np.argmax(counts)
        stats['Mode'] = unique[mode_idx]
    except:
        stats['Mode'] = np.mean(arr)  # Fallback to mean if mode calculation fails
    
    stats['Min'] = np.min(arr)
    stats['Max'] = np.max(arr)
    stats['Range'] = np.max(arr) - np.min(arr)
    
    # 2. Dispersion Statistics
    stats['Std'] = np.std(arr)
    stats['Variance'] = np.var(arr)
    stats['CV'] = np.std(arr) / np.mean(arr) if np.mean(arr) != 0 else 0  # Coefficient of Variation
    stats['IQR'] = np.percentile(arr, 75) - np.percentile(arr, 25)
    stats['MAD'] = np.median(np.abs(arr - np.median(arr)))  # Median Absolute Deviation
    
    # 3. Percentiles
    stats['P10'] = np.percentile(arr, 10)
    stats['P25'] = np.percentile(arr, 25)
    stats['P50'] = np.percentile(arr, 50)
    stats['P75'] = np.percentile(arr, 75)
    stats['P90'] = np.percentile(arr, 90)
    stats['P95'] = np.percentile(arr, 95)
    stats['P99'] = np.percentile(arr, 99)
    
    # 4. Shape Statistics
    stats['Skewness'] = skew(arr) if len(arr) > 2 else 0
    stats['Kurtosis'] = kurtosis(arr) if len(arr) > 2 else 0
    
    # 5. Energy and Power Statistics
    stats['RMS'] = np.sqrt(np.mean(arr**2))  # Root Mean Square
    stats['Peak_to_Peak'] = np.max(arr) - np.min(arr)
    stats['Peak'] = np.max(np.abs(arr))
    stats['Crest_Factor'] = np.max(np.abs(arr)) / np.sqrt(np.mean(arr**2)) if np.mean(arr**2) != 0 else 0
    
    # 6. Additional Statistical Measures
    stats['Zero_Crossings'] = np.sum(np.diff(np.sign(arr)) != 0)
    stats['Mean_Absolute'] = np.mean(np.abs(arr))
    stats['Geometric_Mean'] = np.exp(np.mean(np.log(np.abs(arr) + 1e-10)))
    stats['Harmonic_Mean'] = len(arr) / np.sum(1 / (np.abs(arr) + 1e-10))
    
    return stats

def create_individual_factor_analysis(df, features, save_dir='.'):
    """Create detailed analysis for each individual factor"""
    print("Creating individual factor analysis...")
    
    # Define the factors to analyze
    factors = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Range', 'Skewness', 'Kurtosis', 
               'RMS', 'Peak_to_Peak', 'CV', 'IQR', 'P25', 'P75', 'P90', 'P95']
    
    for feature in features:
        print(f"Processing individual factors for {feature}...")
        
        # Extract statistics for all arrays
        all_stats = []
        for arr in df[feature]:
            if len(arr) > 0:
                stats = extract_detailed_stats(arr)
                all_stats.append(stats)
        
        if not all_stats:
            continue
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(all_stats)
        
        # Create individual factor plots
        create_factor_individual_plots(feature, stats_df, factors, save_dir)

def create_factor_individual_plots(feature, stats_df, factors, save_dir):
    """Create individual plots for each factor"""
    
    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle(f'Detailed Factor Analysis for {feature}', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for i, factor in enumerate(factors[:16]):  # Show first 16 factors
        if factor in stats_df.columns:
            ax = axes[i]
            
            # Create histogram
            ax.hist(stats_df[factor].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{factor} Distribution', fontweight='bold')
            ax.set_xlabel(factor)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = stats_df[factor].mean()
            std_val = stats_df[factor].std()
            median_val = stats_df[factor].median()
            
            textstr = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{feature.replace(" ", "_")}_individual_factors.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual factors for {feature} saved")

def create_factor_comparison_analysis(df, features, save_dir='.'):
    """Create comparison analysis between different factors"""
    print("Creating factor comparison analysis...")
    
    # Extract statistics for all features
    all_stats = {}
    for feature in features:
        print(f"Extracting statistics for {feature}...")
        feature_stats = []
        for arr in df[feature]:
            if len(arr) > 0:
                stats = extract_detailed_stats(arr)
                feature_stats.append(stats)
        
        if feature_stats:
            all_stats[feature] = pd.DataFrame(feature_stats)
    
    # Define key factors for comparison
    key_factors = ['Mean', 'Std', 'Min', 'Max', 'Range', 'Skewness', 'Kurtosis', 
                   'RMS', 'Peak_to_Peak', 'CV', 'IQR', 'P25', 'P75', 'P90', 'P95']
    
    # Create comparison plots
    create_factor_comparison_plots(all_stats, key_factors, save_dir)

def create_factor_comparison_plots(all_stats, key_factors, save_dir):
    """Create comparison plots for factors across features"""
    
    # Create large figure
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle('Factor Comparison Across All Features', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for i, factor in enumerate(key_factors[:16]):
        ax = axes[i]
        
        # Collect data for this factor across all features
        data_to_plot = []
        labels = []
        
        for feature in all_stats.keys():
            if factor in all_stats[feature].columns:
                data_to_plot.append(all_stats[feature][factor].dropna())
                labels.append(feature)
        
        if data_to_plot:
            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_title(f'{factor} Comparison', fontweight='bold')
            ax.set_ylabel(factor)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/factor_comparison_across_features.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Factor comparison across features saved")

def create_statistical_summary_tables(df, features, save_dir='.'):
    """Create detailed statistical summary tables"""
    print("Creating statistical summary tables...")
    
    # Extract comprehensive statistics
    summary_data = {}
    for feature in features:
        print(f"Creating summary for {feature}...")
        
        feature_stats = []
        for arr in df[feature]:
            if len(arr) > 0:
                stats = extract_detailed_stats(arr)
                feature_stats.append(stats)
        
        if feature_stats:
            stats_df = pd.DataFrame(feature_stats)
            summary_data[feature] = stats_df.describe()
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Statistical Summary Tables', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature in summary_data:
            ax = axes[i]
            ax.axis('off')
            
            # Create table
            table_data = summary_data[feature].round(3)
            table = ax.table(cellText=table_data.values,
                           rowLabels=table_data.index,
                           colLabels=table_data.columns,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            ax.set_title(f'{feature} Statistical Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/statistical_summary_tables.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Statistical summary tables saved")

def create_correlation_heatmap_detailed(df, features, save_dir='.'):
    """Create detailed correlation heatmap for all factors"""
    print("Creating detailed correlation heatmap...")
    
    # Extract all statistics
    all_data = {}
    for feature in features:
        feature_data = {}
        for arr in df[feature]:
            if len(arr) > 0:
                stats = extract_detailed_stats(arr)
                for key, value in stats.items():
                    if key not in feature_data:
                        feature_data[key] = []
                    feature_data[key].append(value)
        
        # Calculate means for each factor
        for key in feature_data:
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(np.mean(feature_data[key]))
    
    # Create correlation matrix
    if all_data:
        corr_df = pd.DataFrame(all_data, index=features)
        corr_matrix = corr_df.T.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        plt.title('Correlation Matrix of Statistical Factors Across Features', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/detailed_correlation_heatmap.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Detailed correlation heatmap saved")

def main():
    file_path = 'merged_data_20Hz.csv'
    
    # Load 2% of data
    df, features = load_sample_data(file_path, sample_percentage=2.0)
    
    print(f"Loaded {len(df):,} rows with features: {features}")
    print(f"Data shape: {df.shape}")
    
    # Create output directory
    import os
    output_dir = 'detailed_factor_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate detailed factor analysis
    create_individual_factor_analysis(df, features, output_dir)
    create_factor_comparison_analysis(df, features, output_dir)
    create_statistical_summary_tables(df, features, output_dir)
    create_correlation_heatmap_detailed(df, features, output_dir)
    
    print(f"\nDetailed factor analysis complete! Check the '{output_dir}' directory for generated plots.")
    print("Generated files:")
    print("- [Feature]_individual_factors.png (for each feature)")
    print("- factor_comparison_across_features.png")
    print("- statistical_summary_tables.png")
    print("- detailed_correlation_heatmap.png")
    print(f"\nAnalysis performed on {len(df):,} rows ({2.0}% of total dataset)")
    print("Each feature analyzed with 16+ different statistical factors!")

if __name__ == "__main__":
    main() 