import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
import re
from tqdm import tqdm
warnings.filterwarnings('ignore')

class RailwaySignalAnalyzer:
    def __init__(self, data_path, sample_size=10000):
        self.sample_size = sample_size
        self.data = pd.read_csv(data_path)
        self.features = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def clean_value(self, val):
        """Clean and convert a value to float, handling corrupted data"""
        try:
            # Remove any non-numeric characters except decimal point and minus sign
            cleaned = re.sub(r'[^\d.-]', '', val)
            return float(cleaned)
        except (ValueError, TypeError):
            return None
        
    def preprocess_data(self):
        """Clean and preprocess the raw data"""
        print("Preprocessing data...")
        
        # Sample the data if it's too large
        if len(self.data) > self.sample_size:
            print(f"Sampling {self.sample_size} rows from the dataset...")
            self.data = self.data.sample(n=self.sample_size, random_state=42)
        
        # Convert time column to datetime
        self.data['Time'] = pd.to_datetime(self.data['Time'])
        
        # Convert string measurements to arrays, handling corrupted values
        for col in ['A Current', 'A Voltage', 'B Current', 'B Voltage']:
            self.data[col] = self.data[col].apply(
                lambda x: np.array([val for val in [self.clean_value(v) for v in x.split(',')] if val is not None])
            )
        
        # Remove rows where any measurement array is empty
        self.data = self.data[self.data['A Current'].apply(len) > 0]
        self.data = self.data[self.data['A Voltage'].apply(len) > 0]
        self.data = self.data[self.data['B Current'].apply(len) > 0]
        self.data = self.data[self.data['B Voltage'].apply(len) > 0]
        
        print(f"Data shape after preprocessing: {self.data.shape}")
        
    def extract_features(self):
        """Extract meaningful features from the time series data"""
        print("Extracting features...")
        
        features = []
        
        # Group by zone and point machine
        group_cols = ['Zone', 'Point Machine Name', 'Direction']
        grouped_data = self.data.groupby(group_cols)
        
        # Use tqdm for progress tracking
        for (zone, machine, direction), group in tqdm(grouped_data, desc="Processing groups"):
            # Calculate features for A and B measurements
            for prefix in ['A', 'B']:
                try:
                    # Take mean of measurements to reduce data size
                    current_data = np.array([np.mean(arr) for arr in group[f'{prefix} Current'].values])
                    voltage_data = np.array([np.mean(arr) for arr in group[f'{prefix} Voltage'].values])
                    
                    # Statistical features
                    stats_features = {
                        'zone': zone,
                        'machine': machine,
                        'direction': direction,
                        'type': prefix,
                        'mean_current': np.mean(current_data),
                        'std_current': np.std(current_data),
                        'mean_voltage': np.mean(voltage_data),
                        'std_voltage': np.std(voltage_data),
                        'max_current': np.max(current_data),
                        'max_voltage': np.max(voltage_data),
                        'min_current': np.min(current_data),
                        'min_voltage': np.min(voltage_data),
                        'current_skew': stats.skew(current_data),
                        'voltage_skew': stats.skew(voltage_data),
                        'current_kurtosis': stats.kurtosis(current_data),
                        'voltage_kurtosis': stats.kurtosis(voltage_data)
                    }
                    
                    # Peak features
                    current_peaks, _ = find_peaks(current_data)
                    voltage_peaks, _ = find_peaks(voltage_data)
                    
                    stats_features.update({
                        'num_current_peaks': len(current_peaks),
                        'num_voltage_peaks': len(voltage_peaks),
                        'mean_peak_current': np.mean(current_data[current_peaks]) if len(current_peaks) > 0 else 0,
                        'mean_peak_voltage': np.mean(voltage_data[voltage_peaks]) if len(voltage_peaks) > 0 else 0
                    })
                    
                    features.append(stats_features)
                except Exception as e:
                    print(f"Error processing {prefix} measurements for {zone} - {machine}: {str(e)}")
                    continue
            
        self.features = pd.DataFrame(features)
        print(f"Extracted {len(self.features.columns)} features")
        
    def prepare_data_for_clustering(self):
        """Prepare data for clustering by handling missing values and scaling"""
        # Get numeric columns
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        
        # Handle missing values
        X = self.features[numeric_cols]
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled, X_imputed.columns
        
    def perform_clustering(self, n_clusters=5):
        """Perform clustering on the extracted features"""
        print("Performing clustering...")
        
        # Prepare data for clustering
        X_scaled, _ = self.prepare_data_for_clustering()
        
        # Try different clustering algorithms
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Add cluster labels to features
        self.features['kmeans_cluster'] = kmeans_labels
        self.features['dbscan_cluster'] = dbscan_labels
        
        # Calculate silhouette scores
        kmeans_score = silhouette_score(X_scaled, kmeans_labels)
        dbscan_score = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else None
        
        print(f"K-means silhouette score: {kmeans_score:.3f}")
        if dbscan_score:
            print(f"DBSCAN silhouette score: {dbscan_score:.3f}")
            
        return kmeans_labels, dbscan_labels
        
    def visualize_results(self):
        """Visualize the clustering results"""
        print("Generating visualizations...")
        
        # Prepare data for visualization
        X_scaled, numeric_cols = self.prepare_data_for_clustering()
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot K-means results
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                       hue=self.features['kmeans_cluster'],
                       palette='viridis', ax=ax1)
        ax1.set_title('K-means Clustering')
        
        # Plot DBSCAN results
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                       hue=self.features['dbscan_cluster'],
                       palette='viridis', ax=ax2)
        ax2.set_title('DBSCAN Clustering')
        
        plt.tight_layout()
        plt.savefig('clustering_results.png')
        plt.close()
        
        # Generate cluster statistics
        cluster_stats = self.features.groupby('kmeans_cluster').agg({
            'mean_current': ['mean', 'std'],
            'mean_voltage': ['mean', 'std'],
            'num_current_peaks': ['mean', 'std']
        })
        
        print("\nCluster Statistics:")
        print(cluster_stats)
        
    def analyze_clusters(self):
        """Analyze and interpret the clustering results"""
        print("\nAnalyzing clusters...")
        
        # Analyze each cluster's characteristics
        for cluster in sorted(self.features['kmeans_cluster'].unique()):
            cluster_data = self.features[self.features['kmeans_cluster'] == cluster]
            
            print(f"\nCluster {cluster} Analysis:")
            print(f"Number of samples: {len(cluster_data)}")
            print(f"Average current: {cluster_data['mean_current'].mean():.2f} ± {cluster_data['std_current'].mean():.2f}")
            print(f"Average voltage: {cluster_data['mean_voltage'].mean():.2f} ± {cluster_data['std_voltage'].mean():.2f}")
            
            print("\nMost common zones in this cluster:")
            print(cluster_data['zone'].value_counts().head())
            
            print("\nMost common machines in this cluster:")
            print(cluster_data['machine'].value_counts().head())
            
            print("\nDirection distribution:")
            print(cluster_data['direction'].value_counts())

def main():
    try:
        # Initialize analyzer with a sample size of 10,000 rows
        analyzer = RailwaySignalAnalyzer('merged_data.csv', sample_size=10000)
        
        # Perform analysis steps
        analyzer.preprocess_data()
        analyzer.extract_features()
        analyzer.perform_clustering()
        analyzer.visualize_results()
        analyzer.analyze_clusters()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 