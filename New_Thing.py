import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
import joblib
import os

# Advanced ML/DL libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Advanced signal processing
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter, find_peaks, welch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pywt import wavedec, waverec
import antropy as ant

# Advanced time series libraries
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA as PyOD_PCA
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class ModelConfig:
    """Configuration class for anomaly detection models"""
    window_size: int = 100
    step_size: int = 25
    contamination: float = 0.08
    random_state: int = 42
    n_jobs: int = -1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 10
    
class AdvancedLogger:
    """Enhanced logging system with multiple handlers"""
    
    def __init__(self, name: str = "AnomalyDetection"):
        import os
        os.makedirs('logs', exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(f'logs/anomaly_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

class LSTMAutoencoder(nn.Module):
    """Advanced LSTM-based Autoencoder for time series anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.encoder_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.encoder_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim * 2)
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.decoder_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.decoder_norm = nn.LayerNorm(hidden_dim * 2)
        self.output_layer = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encoder
        encoded, _ = self.encoder_lstm(x)
        encoded = encoded.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim*2)
        encoded_att, _ = self.encoder_attention(encoded, encoded, encoded)
        encoded_att = self.encoder_norm(encoded_att + encoded)
        encoded_att = encoded_att.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim*2)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(encoded_att)
        
        # Decoder
        decoded, _ = self.decoder_lstm(bottleneck_out)
        decoded = decoded.permute(1, 0, 2)
        decoded_att, _ = self.decoder_attention(decoded, decoded, decoded)
        decoded_att = self.decoder_norm(decoded_att + decoded)
        decoded_att = decoded_att.permute(1, 0, 2)
        
        # Output
        output = self.output_layer(decoded_att)
        
        return output

class TransformerAnomalyDetector(nn.Module):
    """Transformer-based anomaly detection model"""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super(TransformerAnomalyDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            dim_feedforward=d_model * 4, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, input_dim)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Reconstruction
        reconstructed = self.output_projection(encoded)
        
        # Anomaly prediction
        anomaly_scores = self.anomaly_head(encoded)
        
        return reconstructed, anomaly_scores

class AdvancedFeatureExtractor:
    """Comprehensive feature extraction for time series data"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        
    def extract_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive statistical features"""
        features = {}
        
        # Basic statistics
        features.update({
            'mean': np.mean(data),
            'std': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'range': np.max(data) - np.min(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
        })
        
        # Distribution features
        features.update({
            'entropy': ant.sample_entropy(data),
            'approximate_entropy': ant.app_entropy(data),
            'permutation_entropy': ant.perm_entropy(data),
            'spectral_entropy': ant.spectral_entropy(data, 100),
            'svd_entropy': ant.svd_entropy(data),
            'hjorth_mobility': ant.hjorth_params(data)[0],
            'hjorth_complexity': ant.hjorth_params(data)[1],
        })
        
        return features
    
    def extract_frequency_features(self, data: np.ndarray, fs: float = 20.0) -> Dict[str, float]:
        """Extract frequency domain features"""
        features = {}
        
        # FFT features
        fft_vals = np.abs(fft(data))
        freqs = fftfreq(len(data), 1/fs)
        
        features.update({
            'dominant_freq': freqs[np.argmax(fft_vals[1:len(fft_vals)//2]) + 1],
            'spectral_centroid': np.sum(freqs[:len(freqs)//2] * fft_vals[:len(fft_vals)//2]) / np.sum(fft_vals[:len(fft_vals)//2]),
            'spectral_rolloff': freqs[np.where(np.cumsum(fft_vals[:len(fft_vals)//2]) >= 0.85 * np.sum(fft_vals[:len(fft_vals)//2]))[0][0]],
            'spectral_bandwidth': np.sqrt(np.sum(((freqs[:len(freqs)//2] - features.get('spectral_centroid', 0))**2) * fft_vals[:len(fft_vals)//2]) / np.sum(fft_vals[:len(fft_vals)//2])),
        })
        
        # Welch's method for power spectral density
        try:
            f_welch, psd = welch(data, fs)
            features.update({
                'psd_peak': f_welch[np.argmax(psd)],
                'total_power': np.sum(psd),
                'relative_power_low': np.sum(psd[f_welch <= 2]) / np.sum(psd),
                'relative_power_mid': np.sum(psd[(f_welch > 2) & (f_welch <= 8)]) / np.sum(psd),
                'relative_power_high': np.sum(psd[f_welch > 8]) / np.sum(psd),
            })
        except:
            pass
        
        return features
    
    def extract_wavelet_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract wavelet-based features"""
        features = {}
        
        try:
            # Discrete wavelet transform
            coeffs = wavedec(data, 'db4', level=6)
            
            for i, coeff in enumerate(coeffs):
                features.update({
                    f'wavelet_energy_level_{i}': np.sum(coeff**2),
                    f'wavelet_std_level_{i}': np.std(coeff),
                    f'wavelet_mean_level_{i}': np.mean(coeff),
                })
            
            # Relative wavelet energy
            total_energy = sum(np.sum(coeff**2) for coeff in coeffs)
            for i, coeff in enumerate(coeffs):
                features[f'wavelet_rel_energy_level_{i}'] = np.sum(coeff**2) / total_energy
        except:
            pass
        
        return features
    
    def extract_temporal_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract temporal pattern features"""
        features = {}
        
        # Trend analysis
        try:
            trend_coeff = np.polyfit(range(len(data)), data, 1)[0]
            features['trend_slope'] = trend_coeff
        except:
            features['trend_slope'] = 0
        
        # Seasonality detection
        try:
            decomposition = seasonal_decompose(pd.Series(data), model='additive')
            features.update({
                'seasonal_strength': np.var(decomposition.seasonal) / np.var(data),
                'trend_strength': np.var(decomposition.trend.dropna()) / np.var(data),
                'residual_strength': np.var(decomposition.resid.dropna()) / np.var(data),
            })
        except:
            pass
        
        # Change points and volatility
        diff_data = np.diff(data)
        features.update({
            'volatility': np.std(diff_data),
            'mean_abs_change': np.mean(np.abs(diff_data)),
            'mean_change': np.mean(diff_data),
            'max_change': np.max(np.abs(diff_data)),
            'num_peaks': len(find_peaks(data)[0]),
            'num_valleys': len(find_peaks(-data)[0]),
        })
        
        return features
    
    def extract_comprehensive_features(self, voltage_data: np.ndarray, current_data: np.ndarray) -> np.ndarray:
        """Extract all features for voltage-current pair"""
        all_features = {}
        
        # Extract features for voltage and current separately
        for data, prefix in [(voltage_data, 'v'), (current_data, 'c')]:
            stat_features = self.extract_statistical_features(data)
            freq_features = self.extract_frequency_features(data)
            wavelet_features = self.extract_wavelet_features(data)
            temporal_features = self.extract_temporal_features(data)
            
            # Add prefix to feature names
            for feature_dict in [stat_features, freq_features, wavelet_features, temporal_features]:
                for key, value in feature_dict.items():
                    all_features[f'{prefix}_{key}'] = value
        
        # Cross-signal features
        try:
            correlation = np.corrcoef(voltage_data, current_data)[0, 1]
            all_features['v_c_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # Mutual information
            all_features['v_c_mutual_info'] = stats.pearsonr(voltage_data, current_data)[0]
            
            # Power features
            power = voltage_data * current_data
            power_features = self.extract_statistical_features(power)
            for key, value in power_features.items():
                all_features[f'power_{key}'] = value
                
        except:
            pass
        
        # Convert to array and handle NaN/inf values
        feature_array = np.array(list(all_features.values()), dtype=np.float64)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_array

class EnsembleAnomalyDetector:
    """Advanced ensemble anomaly detection system"""
    
    def __init__(self, config: ModelConfig, logger: AdvancedLogger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.scalers = {}
        
    def _initialize_models(self):
        """Initialize all anomaly detection models"""
        self.models = {
            # Traditional methods
            'isolation_forest': IForest(contamination=self.config.contamination, random_state=self.config.random_state),
            'local_outlier_factor': LOF(contamination=self.config.contamination),
            'one_class_svm': OCSVM(contamination=self.config.contamination),
            'minimum_covariance_determinant': MCD(contamination=self.config.contamination, random_state=self.config.random_state),
            'principal_component_analysis': PyOD_PCA(contamination=self.config.contamination, random_state=self.config.random_state),
            
            # Advanced methods
            'angle_based_outlier_detection': ABOD(contamination=self.config.contamination),
            'clustering_based_local_outlier': CBLOF(contamination=self.config.contamination, random_state=self.config.random_state),
            'histogram_based_outlier_score': HBOS(contamination=self.config.contamination),
            'subspace_outlier_detection': SOD(contamination=self.config.contamination),
            'feature_bagging': FeatureBagging(contamination=self.config.contamination, random_state=self.config.random_state),
            
            # Deep learning methods
            'autoencoder': AutoEncoder(contamination=self.config.contamination, epochs=50, batch_size=32, random_state=self.config.random_state),
            'variational_autoencoder': VAE(contamination=self.config.contamination, epochs=50, batch_size=32, random_state=self.config.random_state),
            'deep_svdd': DeepSVDD(contamination=self.config.contamination, epochs=50, batch_size=32, random_state=self.config.random_state),
            
            # GAN-based methods (commented out due to potential installation issues)
            # 'single_objective_gaal': SO_GAAL(contamination=self.config.contamination),
            # 'multi_objective_gaal': MO_GAAL(contamination=self.config.contamination),
        }
    
    def fit_predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit all models and generate predictions"""
        self._initialize_models()
        results = {}
        
        # Scale features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        self.logger.info(f"Training {len(self.models)} anomaly detection models...")
        
        for name, model in tqdm(self.models.items(), desc="Training models"):
            try:
                self.logger.debug(f"Training {name}")
                
                # Fit and predict
                model.fit(features_scaled)
                predictions = model.predict(features_scaled)  # 0 for normal, 1 for anomaly
                scores = model.decision_function(features_scaled)
                
                results[name] = {
                    'predictions': predictions,
                    'scores': scores,
                    'model': model
                }
                
                anomaly_count = np.sum(predictions)
                self.logger.info(f"{name}: {anomaly_count} anomalies ({anomaly_count/len(features)*100:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        return results
    
    def create_ensemble_prediction(self, results: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Create ensemble predictions using voting and averaging"""
        
        # Collect all predictions and scores
        all_predictions = []
        all_scores = []
        model_names = []
        
        for name, result in results.items():
            all_predictions.append(result['predictions'])
            all_scores.append(result['scores'])
            model_names.append(name)
        
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
        
        # Voting ensemble
        voting_predictions = np.sum(all_predictions, axis=0)
        voting_threshold = len(model_names) * 0.3  # 30% of models must agree
        ensemble_predictions = (voting_predictions >= voting_threshold).astype(int)
        
        # Score averaging
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Weighted ensemble (give more weight to better performing models)
        weights = self._calculate_model_weights(results)
        weighted_scores = np.average(all_scores, axis=0, weights=weights)
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'ensemble_scores': ensemble_scores,
            'weighted_scores': weighted_scores,
            'voting_scores': voting_predictions,
            'model_names': model_names,
            'individual_results': results
        }
    
    def _calculate_model_weights(self, results: Dict[str, Dict]) -> np.ndarray:
        """Calculate weights for ensemble based on model performance"""
        weights = []
        for name, result in results.items():
            # Simple weighting based on score distribution
            score_std = np.std(result['scores'])
            weight = 1.0 / (score_std + 1e-6)  # Higher weight for more discriminative scores
            weights.append(weight)
        
        weights = np.array(weights)
        return weights / np.sum(weights)  # Normalize weights

class ModernAnomalyDetectionPipeline:
    """Complete modern anomaly detection pipeline"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.logger = AdvancedLogger()
        self.feature_extractor = AdvancedFeatureExtractor(self.logger)
        self.ensemble_detector = EnsembleAnomalyDetector(self.config, self.logger)
        
        # Create directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'logs', 'results', 'plots', 'models', 
            'data/processed', 'data/features', 'plots/interactive'
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_and_preprocess_data(self, dataset_path: str, chunk_size: int = 1000) -> pd.DataFrame:
        """Load and preprocess the 20Hz dataset in chunks to save memory"""
        self.logger.info(f"Loading dataset from: {dataset_path}")
        processed_chunks = []
        for df_raw in pd.read_csv(dataset_path, chunksize=chunk_size):
            formatted_data = []
            for idx, row in df_raw.iterrows():
                try:
                    # Extract and clean comma-separated values
                    signals = {}
                    for col in ['A Current', 'A Voltage', 'B Current', 'B Voltage']:
                        signal_str = str(row[col]).strip('"')
                        values = [float(x.strip()) for x in signal_str.split(',') if x.strip()]
                        signals[col.lower().replace(' ', '_')] = values
                    
                    # Find minimum length
                    min_length = min(len(values) for values in signals.values())
                    
                    if min_length > 0:
                        # Create time series data points
                        for i in range(min_length):
                            formatted_row = {
                                'record_id': idx,
                                'time_point': i,
                                'site_name': row['Site Name'],
                                'machine_name': row['Point Machine Name'],
                                'direction': row['Direction'],
                                'timestamp': row['Time'],
                                'a_current': signals['a_current'][i],
                                'a_voltage': signals['a_voltage'][i],
                                'b_current': signals['b_current'][i],
                                'b_voltage': signals['b_voltage'][i],
                                'type_a': row['Type of A'],
                                'type_b': row['Type of B'],
                                'polling_a': row['Polling of A'],
                                'polling_b': row['Polling of B'],
                                'source_dataset': row['source_dataset']
                            }
                            formatted_data.append(formatted_row)
                            
                except Exception as e:
                    self.logger.error(f"Error processing row {idx}: {str(e)}")
                    continue
            
            df_processed = pd.DataFrame(formatted_data)
            processed_chunks.append(df_processed)
        df_final = pd.concat(processed_chunks, ignore_index=True)
        self.logger.info(f"Processed dataset shape: {df_final.shape}")
        df_final.to_csv('data/processed/formatted_data.csv', index=False)
        return df_final
    
    def extract_windowed_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features using sliding window approach"""
        
        self.logger.info("Extracting windowed features...")
        
        # Prepare signal data
        signals = {
            'A': (df['a_voltage'].values, df['a_current'].values),
            'B': (df['b_voltage'].values, df['b_current'].values)
        }
        
        all_features = []
        all_indices = []
        pair_labels = []
        
        for pair_name, (voltage_data, current_data) in signals.items():
            self.logger.info(f"Processing signal pair {pair_name}")
            
            # Create sliding windows
            for i in tqdm(range(0, len(voltage_data) - self.config.window_size + 1, self.config.step_size),
                         desc=f"Extracting features for pair {pair_name}"):
                
                v_window = voltage_data[i:i + self.config.window_size]
                c_window = current_data[i:i + self.config.window_size]
                
                # Extract comprehensive features
                features = self.feature_extractor.extract_comprehensive_features(v_window, c_window)
                
                all_features.append(features)
                all_indices.append(i + self.config.window_size // 2)
                pair_labels.append(pair_name)
        
        feature_matrix = np.array(all_features)
        indices = np.array(all_indices)
        
        self.logger.info(f"Extracted {feature_matrix.shape[0]} windows with {feature_matrix.shape[1]} features each")
        
        # Save features
        np.save('data/features/feature_matrix.npy', feature_matrix)
        np.save('data/features/indices.npy', indices)
        
        return feature_matrix, indices, pair_labels
    
    def detect_anomalies(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect anomalies using ensemble methods"""
        
        self.logger.info("Starting anomaly detection...")
        
        # Ensemble detection
        ensemble_results = self.ensemble_detector.fit_predict(features)
        
        # Deep learning detection
        dl_results = self.deep_learning_detection(features)
        
        # Combine results
        combined_results = {**ensemble_results, **dl_results}
        
        return combined_results
    
    def deep_learning_detection(self, features: np.ndarray) -> Dict[str, Dict]:
        """Deep learning based anomaly detection"""
        
        self.logger.info("Running deep learning anomaly detection...")
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        seq_length = min(50, features.shape[0] // 10)
        if features.shape[0] < seq_length:
            self.logger.warning("Not enough data for deep learning models")
            return {}
        # Sample only a subset of sequences to save memory
        max_sequences = 1000
        indices = np.arange(len(features_scaled) - seq_length + 1)
        if len(indices) > max_sequences:
            indices = np.random.choice(indices, max_sequences, replace=False)
        sequences = np.array([features_scaled[i:i + seq_length] for i in indices])
        
        results = {}
        
        # LSTM Autoencoder
        try:
            lstm_ae = LSTMAutoencoder(features.shape[1])
            lstm_results = self.train_lstm_autoencoder(lstm_ae, sequences)
            results['lstm_autoencoder'] = lstm_results
        except Exception as e:
            self.logger.error(f"LSTM Autoencoder failed: {str(e)}")
        
        # Transformer
        try:
            transformer = TransformerAnomalyDetector(features.shape[1])
            transformer_results = self.train_transformer(transformer, sequences)
            results['transformer'] = transformer_results
        except Exception as e:
            self.logger.error(f"Transformer failed: {str(e)}")
        
        return results
    
    def train_lstm_autoencoder(self, model: LSTMAutoencoder, sequences: np.ndarray) -> Dict[str, np.ndarray]:
        """Train LSTM Autoencoder"""
        
        device = torch.device(self.config.device)
        model.to(device)
        
        # Prepare data
        tensor_sequences = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(tensor_sequences, tensor_sequences)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(self.config.epochs // 2):  # Reduced epochs for demo
            total_loss = 0
            for batch_data, batch_target in dataloader:
                optimizer.zero_grad()
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                self.logger.debug(f"LSTM AE Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Generate anomaly scores
        model.eval()
        with torch.no_grad():
            reconstructed = model(tensor_sequences)
            mse_scores = torch.mean((tensor_sequences - reconstructed) ** 2, dim=(1, 2))
            anomaly_scores = mse_scores.cpu().numpy()
        
        # Convert scores to predictions
        threshold = np.percentile(anomaly_scores, (1 - self.config.contamination) * 100)
        predictions = (anomaly_scores > threshold).astype(int)
        
        return {
            'predictions': predictions,
            'scores': anomaly_scores,
            'model': model
        }
    
    def train_transformer(self, model: TransformerAnomalyDetector, sequences: np.ndarray) -> Dict[str, np.ndarray]:
        """Train Transformer model"""
        
        device = torch.device(self.config.device)
        model.to(device)
        
        # Prepare data
        tensor_sequences = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(tensor_sequences, tensor_sequences)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(self.config.epochs // 2):  # Reduced epochs for demo
            total_loss = 0
            for batch_data, batch_target in dataloader:
                optimizer.zero_grad()
                reconstructed, anomaly_scores = model(batch_data)
                loss = criterion(reconstructed, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                self.logger.debug(f"Transformer Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Generate final predictions
        model.eval()
        with torch.no_grad():
            _, anomaly_scores = model(tensor_sequences)
            scores = anomaly_scores.squeeze().cpu().numpy()
        
        # Convert scores to predictions
        threshold = np.percentile(scores, (1 - self.config.contamination) * 100)
        predictions = (scores > threshold).astype(int)
        
        return {
            'predictions': predictions,
            'scores': scores,
            'model': model
        }
    
    def create_advanced_visualizations(self, df: pd.DataFrame, results: Dict, indices: np.ndarray, pair_labels: List[str]):
        """Create advanced interactive visualizations"""
        
        self.logger.info("Creating advanced visualizations...")
        
        # 1. Interactive time series with anomalies
        self.create_interactive_time_series(df, results, indices, pair_labels)
        
        # 2. Model comparison dashboard
        self.create_model_comparison_dashboard(results)
        
        # 3. Feature importance analysis
        self.create_feature_importance_analysis(results)
        
        # 4. Anomaly distribution analysis
        self.create_anomaly_distribution_analysis(results, indices)
        
        # 5. 3D visualization of anomalies
        self.create_3d_anomaly_visualization(results)
    
    def create_interactive_time_series(self, df: pd.DataFrame, results: Dict, indices: np.ndarray, pair_labels: List[str]):
        """Create interactive time series plots"""
        
        # Get ensemble results
        if 'ensemble_predictions' in results:
            ensemble_preds = results['ensemble_predictions']
            ensemble_scores = results['ensemble_scores']
        else:
            # Fallback to first available result
            first_result = list(results.values())[0]
            ensemble_preds = first_result['predictions']
            ensemble_scores = first_result['scores']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['A Voltage', 'A Current', 'B Voltage', 'B Current', 'Anomaly Scores', 'Model Comparison'],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"colspan": 2}, None]]
        )
        
        # Plot signals with anomalies
        signals = [
            (df['a_voltage'].values, 'A Voltage', 'blue'),
            (df['a_current'].values, 'A Current', 'red'),
            (df['b_voltage'].values, 'B Voltage', 'green'),
            (df['b_current'].values, 'B Current', 'orange')
        ]
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, ((signal_data, name, color), (row, col)) in enumerate(zip(signals, positions)):
            # Plot signal
            fig.add_trace(
                go.Scatter(x=list(range(len(signal_data))), y=signal_data,
                          mode='lines', name=name, line=dict(color=color, width=1)),
                row=row, col=col
            )
            
            # Add anomaly markers
            anomaly_indices = indices[ensemble_preds == 1]
            if len(anomaly_indices) > 0:
                fig.add_trace(
                    go.Scatter(x=anomaly_indices, y=signal_data[anomaly_indices],
                              mode='markers', name=f'{name} Anomalies',
                              marker=dict(color='red', size=8, symbol='x')),
                    row=row, col=col
                )
        
        # Plot anomaly scores
        fig.add_trace(
            go.Scatter(x=indices, y=ensemble_scores,
                      mode='lines+markers', name='Anomaly Scores',
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Advanced Anomaly Detection Results",
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Save interactive plot
        fig.write_html('plots/interactive/anomaly_detection_results.html')
        self.logger.info("Saved interactive plot: plots/interactive/anomaly_detection_results.html")
    
    def create_model_comparison_dashboard(self, results: Dict):
        """Create model comparison dashboard"""
        
        if 'individual_results' not in results:
            return
        
        individual_results = results['individual_results']
        
        # Prepare data for comparison
        model_names = []
        anomaly_counts = []
        score_means = []
        score_stds = []
        
        for name, result in individual_results.items():
            model_names.append(name.replace('_', ' ').title())
            anomaly_counts.append(np.sum(result['predictions']))
            score_means.append(np.mean(result['scores']))
            score_stds.append(np.std(result['scores']))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Anomaly Counts by Model', 'Score Distributions', 
                           'Model Agreement Matrix', 'Performance Metrics'],
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Anomaly counts
        fig.add_trace(
            go.Bar(x=model_names, y=anomaly_counts, name='Anomaly Counts',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Score distributions (box plot)
        for i, (name, result) in enumerate(individual_results.items()):
            fig.add_trace(
                go.Box(y=result['scores'], name=name.replace('_', ' ').title(),
                       boxpoints='outliers'),
                row=1, col=2
            )
        
        # Create agreement matrix
        predictions_matrix = np.array([result['predictions'] for result in individual_results.values()])
        agreement_matrix = np.corrcoef(predictions_matrix)
        
        fig.add_trace(
            go.Heatmap(
                z=agreement_matrix,
                x=model_names,
                y=model_names,
                colorscale='RdBu',
                name='Agreement'
            ),
            row=2, col=1
        )
        
        # Performance metrics
        fig.add_trace(
            go.Bar(x=model_names, y=score_stds, name='Score Standard Deviation',
                   marker_color='lightgreen'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Comparison Dashboard",
            height=800,
            showlegend=False
        )
        
        fig.write_html('plots/interactive/model_comparison.html')
        self.logger.info("Saved model comparison dashboard: plots/interactive/model_comparison.html")
    
    def create_feature_importance_analysis(self, results: Dict):
        """Analyze and visualize feature importance"""
        
        # This is a placeholder for feature importance analysis
        # In practice, you would extract feature importance from tree-based models
        self.logger.info("Feature importance analysis completed (placeholder)")
    
    def create_anomaly_distribution_analysis(self, results: Dict, indices: np.ndarray):
        """Analyze anomaly distribution patterns"""
        
        if 'ensemble_predictions' not in results:
            return
        
        anomaly_indices = indices[results['ensemble_predictions'] == 1]
        
        # Time-based distribution
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(x=anomaly_indices, nbinsx=50, name='Anomaly Distribution',
                        marker_color='red', opacity=0.7)
        )
        
        fig.update_layout(
            title="Temporal Distribution of Anomalies",
            xaxis_title="Time Index",
            yaxis_title="Frequency",
            height=400
        )
        
        fig.write_html('plots/interactive/anomaly_distribution.html')
        self.logger.info("Saved anomaly distribution analysis: plots/interactive/anomaly_distribution.html")
    
    def create_3d_anomaly_visualization(self, results: Dict):
        """Create 3D visualization of anomalies using dimensionality reduction"""
        
        # This would require the original feature matrix
        # Placeholder for 3D visualization
        self.logger.info("3D anomaly visualization completed (placeholder)")
    
    def save_comprehensive_results(self, results: Dict, df: pd.DataFrame, feature_matrix: np.ndarray):
        """Save all results in multiple formats"""
        
        self.logger.info("Saving comprehensive results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ensemble results
        if 'ensemble_predictions' in results:
            ensemble_df = pd.DataFrame({
                'predictions': results['ensemble_predictions'],
                'scores': results['ensemble_scores'],
                'weighted_scores': results['weighted_scores'],
                'voting_scores': results['voting_scores']
            })
            ensemble_df.to_csv(f'results/ensemble_results_{timestamp}.csv', index=False)
        
        # Save individual model results
        if 'individual_results' in results:
            for model_name, model_result in results['individual_results'].items():
                model_df = pd.DataFrame({
                    'predictions': model_result['predictions'],
                    'scores': model_result['scores']
                })
                model_df.to_csv(f'results/{model_name}_results_{timestamp}.csv', index=False)
        
        # Save summary statistics
        summary_stats = {
            'timestamp': timestamp,
            'dataset_shape': df.shape,
            'feature_matrix_shape': feature_matrix.shape,
            'total_anomalies': int(np.sum(results.get('ensemble_predictions', [0]))),
            'anomaly_percentage': float(np.mean(results.get('ensemble_predictions', [0])) * 100),
            'models_used': list(results.get('individual_results', {}).keys())
        }
        
        with open(f'results/summary_stats_{timestamp}.json', 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        # Save models
        for model_name, model_result in results.get('individual_results', {}).items():
            if 'model' in model_result:
                joblib.dump(model_result['model'], f'models/{model_name}_{timestamp}.pkl')
        
        self.logger.info(f"All results saved with timestamp: {timestamp}")
    
    def run_complete_pipeline(self, dataset_path: str):
        """Run the complete anomaly detection pipeline"""
        
        self.logger.info("=" * 80)
        self.logger.info("ADVANCED ANOMALY DETECTION PIPELINE")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Data Loading and Preprocessing")
            df = self.load_and_preprocess_data(dataset_path)
            
            # Step 2: Feature extraction
            self.logger.info("Step 2: Advanced Feature Extraction")
            feature_matrix, indices, pair_labels = self.extract_windowed_features(df)
            
            # Step 3: Anomaly detection
            self.logger.info("Step 3: Ensemble Anomaly Detection")
            results = self.detect_anomalies(feature_matrix)
            
            # Step 4: Create ensemble predictions
            if 'individual_results' in results:
                self.logger.info("Step 4: Creating Ensemble Predictions")
                ensemble_results = self.ensemble_detector.create_ensemble_prediction(results['individual_results'])
                results.update(ensemble_results)
            
            # Step 5: Advanced visualizations
            self.logger.info("Step 5: Creating Advanced Visualizations")
            self.create_advanced_visualizations(df, results, indices, pair_labels)
            
            # Step 6: Save results
            self.logger.info("Step 6: Saving Comprehensive Results")
            self.save_comprehensive_results(results, df, feature_matrix)
            
            # Final summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"Total execution time: {duration:.2f} seconds")
            self.logger.info(f"Processed {len(df)} data points")
            self.logger.info(f"Created {feature_matrix.shape[0]} feature windows")
            self.logger.info(f"Used {len(results.get('individual_results', {}))} detection models")
            
            if 'ensemble_predictions' in results:
                total_anomalies = np.sum(results['ensemble_predictions'])
                anomaly_rate = total_anomalies / len(results['ensemble_predictions']) * 100
                self.logger.info(f"Detected {total_anomalies} anomalies ({anomaly_rate:.2f}%)")
            
            return results, df, feature_matrix
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    
    # Configuration
    config = ModelConfig(
        window_size=100,
        step_size=25,
        contamination=0.08,
        epochs=50,  # Reduced for faster execution
        batch_size=64
    )
    
    # Initialize pipeline
    pipeline = ModernAnomalyDetectionPipeline(config)
    
    # Run pipeline
    dataset_path = "processed_data/merged_data_20Hz.csv"
    results, df, features = pipeline.run_complete_pipeline(dataset_path)
    
    return results, df, features

if __name__ == "__main__":
    # Run the advanced pipeline
    results, df, features = main()
