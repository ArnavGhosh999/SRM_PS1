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

def cluster_and_plot(df, feature, n_clusters=10, save_path='A_current_clusters.png'):
    arr_lengths = pd.Series([len(x) for x in df[feature]])
    min_valid_length = 100
    filtered_df = df[arr_lengths >= min_valid_length]
    if filtered_df.empty:
        print(f'No valid time-series found for {feature} with length >= {min_valid_length}')
        return
    filtered_df = filtered_df.reset_index(drop=True)
    valid_lengths = pd.Series([len(x) for x in filtered_df[feature]])
    common_len = valid_lengths.mode()[0]
    filtered_df = filtered_df[valid_lengths == common_len]
    arrs = np.stack(filtered_df[feature].values)
    # Feature extraction
    feature_matrix = np.array([extract_features(arr) for arr in arrs])
    # Replace NaN and inf values with 0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    # Dimensionality reduction
    pca = PCA(n_components=min(5, feature_matrix_scaled.shape[1]))
    reduced_features = pca.fit_transform(feature_matrix_scaled)
    # Cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    clusters = kmeans.fit_predict(reduced_features)
    # Plot mean time series for each cluster
    fig, axes = plt.subplots(2, (n_clusters+1)//2, figsize=(20, 8))
    axes = axes.flatten()
    fig.suptitle(f'{feature} Mean Time Series for Clusters', fontsize=20)
    for i in range(n_clusters):
        ax = axes[i]
        idxs = np.where(clusters == i)[0]
        n_members = len(idxs)
        if n_members == 0:
            ax.axis('off')
            continue
        mean_series = arrs[idxs].mean(axis=0)
        ax.plot(mean_series, color='blue', label='Mean')
        ax.fill_between(range(len(mean_series)),
                        arrs[idxs].min(axis=0),
                        arrs[idxs].max(axis=0),
                        color='blue', alpha=0.1, label='Min-Max Range')
        ax.set_title(f'Cluster {i}\n{n_members} members ({100*n_members/len(arrs):.2f}%)')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.close()

def main():
    file_path = '/media/umeshgjh/New Volume/Energy7 Internship/May 7/merged_data_20Hz (copy).csv'
    df, features = load_and_parse_arrays(file_path, n_rows=None)  # Load all rows

    # Diagnostics for B Current and B Voltage
    for feature in ['B Current', 'B Voltage']:
        arr_lengths = pd.Series([len(x) for x in df[feature]])
        print(f"{feature} array length value counts:")
        print(arr_lengths.value_counts())
        empty_count = (arr_lengths == 0).sum()
        print(f"{feature} empty arrays: {empty_count}\n")

    cluster_and_plot(df, 'A Current', 10, 'A_current_clusters.png')
    cluster_and_plot(df, 'A Voltage', 10, 'A_voltage_clusters.png')
    cluster_and_plot(df, 'B Current', 10, 'B_current_clusters.png')
    cluster_and_plot(df, 'B Voltage', 10, 'B_voltage_clusters.png')
    print('All cluster images saved.')

if __name__ == "__main__":
    main()
