import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import ast
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Helper to parse array columns efficiently

def parse_array_string(x):
    if isinstance(x, str):
        try:
            return np.fromiter((float(i) for i in x.split(',')), dtype=float)
        except Exception:
            return np.array([])
    else:
        return np.array([])

def parse_array_column(col):
    # Use ThreadPoolExecutor for faster parsing on large columns
    with ThreadPoolExecutor() as executor:
        return list(executor.map(parse_array_string, col))

def load_and_parse_arrays(file_path, n_rows=None):
    # n_rows=None loads all rows
    df = pd.read_csv(file_path, nrows=n_rows)
    features = ['A Current', 'A Voltage', 'B Current', 'B Voltage']
    for feat in features:
        df[feat] = parse_array_column(df[feat])
    return df, features

def extract_features(arr):
    # arr: 1D numpy array
    features = [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        skew(arr),
        kurtosis(arr)
    ]
    # FFT features (first 5 magnitudes, excluding DC)
    fft_vals = np.abs(np.fft.rfft(arr))
    fft_features = fft_vals[1:6] if len(fft_vals) > 6 else np.pad(fft_vals[1:], (0, 5-len(fft_vals[1:])), 'constant')
    features.extend(fft_features)
    return features

def pad_array(arr, target_len):
    # Pad or truncate array to target_len
    if len(arr) >= target_len:
        return arr[:target_len]
    else:
        return np.pad(arr, (0, target_len - len(arr)), constant_values=np.nan)

def cluster_and_plot(df, feature, n_clusters=10, save_path='A_current_clusters.png'):
    # No filtering: use all rows
    arrs_list = df[feature].tolist()
    max_len = max(len(x) for x in arrs_list)
    arrs = np.stack([pad_array(x, max_len) for x in arrs_list])
    # Feature extraction (ignore NaNs in stats)
    def safe_extract_features(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            arr = np.zeros(1)
        return extract_features(arr)
    feature_matrix = np.array([safe_extract_features(arr) for arr in arrs])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    pca = PCA(n_components=min(5, feature_matrix_scaled.shape[1]))
    reduced_features = pca.fit_transform(feature_matrix_scaled)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    clusters = kmeans.fit_predict(reduced_features)
    # Plot enhanced cluster comparison
    fig, axes = plt.subplots(2, (n_clusters+1)//2, figsize=(24, 10))
    axes = axes.flatten()
    fig.suptitle(f'{feature} Cluster Comparison (All Data)', fontsize=22)
    for i in range(n_clusters):
        ax = axes[i]
        idxs = np.where(clusters == i)[0]
        n_members = len(idxs)
        if n_members == 0:
            ax.axis('off')
            continue
        cluster_arrs = arrs[idxs]
        # Ignore NaNs in stats
        mean_series = np.nanmean(cluster_arrs, axis=0)
        median_series = np.nanmedian(cluster_arrs, axis=0)
        std_series = np.nanstd(cluster_arrs, axis=0)
        min_series = np.nanmin(cluster_arrs, axis=0)
        max_series = np.nanmax(cluster_arrs, axis=0)
        ax.plot(mean_series, color='blue', label='Mean', linewidth=2)
        ax.plot(median_series, color='orange', linestyle='--', label='Median', linewidth=2)
        ax.fill_between(range(len(mean_series)), min_series, max_series, color='blue', alpha=0.08, label='Min-Max Range')
        ax.fill_between(range(len(mean_series)), mean_series-std_series, mean_series+std_series, color='green', alpha=0.15, label='±1 Std Dev')
        if n_members > 1:
            sample_indices = np.random.choice(n_members, min(5, n_members), replace=False)
            for si in sample_indices:
                ax.plot(cluster_arrs[si], color='gray', alpha=0.3, linewidth=1, label='Sample' if si==sample_indices[0] else None)
        ax.set_title(f'Cluster {i}\n{n_members} members ({100*n_members/len(arrs):.2f}%)')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8)
    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def main():
    file_path = '/media/umeshgjh/New Volume/Energy7 Internship/June 20 meet/merged_data_20Hz.csv'
    df, features = load_and_parse_arrays(file_path, n_rows=None)  # Load all rows

    for feature in features:
        cluster_and_plot(df, feature, 10, f'{feature.replace(" ", "_").lower()}_clusters.png')
    print('All cluster images saved.')

if __name__ == "__main__":
    main()
