import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('merged_data_20Hz.csv', low_memory=False)

# 2. Clean and Preprocess Data
# Fix column names (remove newlines, extra spaces)
df.columns = [c.replace('\n', '').replace(' ', '').replace('TypeofA', 'Type of A').replace('TypeofB', 'Type of B') for c in df.columns]

# Drop columns with all NaN or irrelevant columns if any
df = df.dropna(axis=1, how='all')

# Identify sequence columns (those with long comma-separated values)
sequence_cols = ['ACurrent', 'AVoltage', 'BCurrent', 'BVoltage']

# Function to expand sequence columns into multiple features
def expand_sequence_column(series, prefix):
    # Split by comma, pad with zeros if needed
    expanded = series.fillna('0').apply(lambda x: [float(i) for i in str(x).split(',') if i.strip() != ''])
    max_len = expanded.apply(len).max()
    expanded = expanded.apply(lambda x: x + [0]*(max_len - len(x)))
    arr = np.vstack(expanded.values)
    return pd.DataFrame(arr, columns=[f'{prefix}_{i}' for i in range(arr.shape[1])])

# Expand all sequence columns
expanded_features = []
for col in sequence_cols:
    if col in df.columns:
        expanded_features.append(expand_sequence_column(df[col], col))

# Concatenate expanded features
if expanded_features:
    expanded_df = pd.concat(expanded_features, axis=1)
else:
    expanded_df = pd.DataFrame()

# Encode categorical columns
categorical_cols = ['SiteName', 'PointMachineName', 'Direction', 'TypeofA', 'TypeofB', 'source_dataset']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Select features for clustering
feature_cols = []
for col in categorical_cols:
    if col in df.columns:
        feature_cols.append(col)
if not expanded_df.empty:
    X = pd.concat([df[feature_cols].reset_index(drop=True), expanded_df.reset_index(drop=True)], axis=1)
else:
    X = df[feature_cols]

# Remove rows with NaN (if any remain)
X = X.dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Clustering and Anomaly Detection
# Try DBSCAN (density-based, good for anomaly detection)
dbscan = DBSCAN(eps=2, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

# Try KMeans for comparison
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Try Isolation Forest for anomaly scoring
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(X_scaled)

# 5. Insights and Visualization
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.title('DBSCAN Clusters')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap='tab10', s=10)
plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.subplot(1, 3, 2)
plt.title('KMeans Clusters')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', s=10)
plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.subplot(1, 3, 3)
plt.title('Isolation Forest Anomalies')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iso_labels, cmap='coolwarm', s=10)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.tight_layout()
plt.savefig('cluster_anomaly_insights.png')
plt.show()

# Print cluster/anomaly statistics
def print_stats(labels, name):
    unique, counts = np.unique(labels, return_counts=True)
    print(f'\n{name} label distribution:')
    for u, c in zip(unique, counts):
        print(f'  Label {u}: {c} samples')

print_stats(db_labels, 'DBSCAN')
print_stats(kmeans_labels, 'KMeans')
print_stats(iso_labels, 'Isolation Forest')

# Silhouette score for clustering quality (ignore -1 for DBSCAN noise)
if len(set(db_labels)) > 1 and -1 in db_labels:
    mask = db_labels != -1
    sil_db = silhouette_score(X_scaled[mask], db_labels[mask])
    print(f'\nDBSCAN Silhouette Score (excluding noise): {sil_db:.3f}')
if len(set(kmeans_labels)) > 1:
    sil_km = silhouette_score(X_scaled, kmeans_labels)
    print(f'KMeans Silhouette Score: {sil_km:.3f}')

# Feature importances for anomalies (Isolation Forest)
if hasattr(iso_forest, 'feature_importances_'):
    importances = iso_forest.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    print('\nTop 10 features for anomaly detection:')
    for i in idx:
        print(f'{X.columns[i]}: {importances[i]:.4f}')

print('\nCluster and anomaly detection complete. See cluster_anomaly_insights.png for visualization.') 