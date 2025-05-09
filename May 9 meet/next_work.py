import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import ast

# Helper to parse array columns
def parse_array_column(col):
    def parse(x):
        if isinstance(x, str):
            try:
                return np.array([float(i) for i in x.split(',')])
            except Exception:
                return np.array([])
        else:
            return np.array([])
    return col.apply(parse)

def load_and_parse_arrays(file_path, n_rows=25500):
    df = pd.read_csv(file_path, nrows=n_rows)
    features = ['A Current', 'A Voltage', 'B Current', 'B Voltage']
    for feat in features:
        df[feat] = parse_array_column(df[feat])
    return df, features

def cluster_and_plot(df, feature, n_clusters=25, save_path='A_current_clusters.png'):
    arr_lengths = df[feature].apply(lambda x: len(x))
    min_valid_length = 100
    filtered_df = df[arr_lengths >= min_valid_length]
    if filtered_df.empty:
        print(f'No valid time-series found for {feature} with length >= {min_valid_length}')
        return
    valid_lengths = filtered_df[feature].apply(lambda x: len(x))
    common_len = valid_lengths.mode()[0]
    filtered_df = filtered_df[valid_lengths == common_len]
    arrs = np.stack(filtered_df[feature].values)
    # Cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    clusters = kmeans.fit_predict(arrs)
    # Plot
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    fig.suptitle(f'{feature} Data for all Clusters', fontsize=20)
    for i in range(n_clusters):
        ax = axes[i // 5, i % 5]
        idxs = np.where(clusters == i)[0]
        for idx in idxs:
            ax.plot(arrs[idx], alpha=0.5)
        ax.set_title(f'Cluster {i}\n{100*len(idxs)/len(arrs):.2f}%')
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n_clusters, 25):
        axes[j // 5, j % 5].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.close()

def main():
    file_path = '/media/umeshgjh/New Volume/Energy7 Internship/May 7/merged_data_20Hz (copy).csv'
    df, features = load_and_parse_arrays(file_path, n_rows=25500)

    # Diagnostics for B Current and B Voltage
    for feature in ['B Current', 'B Voltage']:
        arr_lengths = df[feature].apply(lambda x: len(x))
        print(f"{feature} array length value counts:")
        print(arr_lengths.value_counts())
        empty_count = (arr_lengths == 0).sum()
        print(f"{feature} empty arrays: {empty_count}\n")

    cluster_and_plot(df, 'A Current', 25, 'A_current_clusters.png')
    cluster_and_plot(df, 'A Voltage', 25, 'A_voltage_clusters.png')
    cluster_and_plot(df, 'B Current', 25, 'B_current_clusters.png')
    cluster_and_plot(df, 'B Voltage', 25, 'B_voltage_clusters.png')
    print('All cluster images saved.')

if __name__ == "__main__":
    main()
