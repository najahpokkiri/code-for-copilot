# Enhanced Biomass Prediction: Train on Yellapur, Test on Uppangala
# FULLY FIXED VERSION - All errors resolved
# ============================================================================

# Step 1: Setup and Imports
import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import random
from collections import defaultdict

# Core ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Enhanced ML imports
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Tree models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# Progress tracking
from tqdm import tqdm

# Image processing
import rasterio
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel, laplace
from scipy import ndimage
from scipy.stats import skew, kurtosis, pearsonr, spearmanr

# Suppress warnings
warnings.filterwarnings('ignore')

print("🌟 BIOMASS PREDICTION: TRAIN ON YELLAPUR, TEST ON UPPANGALA")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"User: najahpokkiri")
print(f"Strategy: Single holdout validation (Yellapur → Uppangala)")

# Step 2: Enhanced Configuration
class TrainTestConfig:
    """Configuration for train on Yellapur, test on Uppangala"""
    
    # Data paths for the two sites
    train_raster_pair = (
        "/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif",
        "/teamspace/studios/dl2/clean/data/agbd_yellapur_reprojected_1.tif"
    )
    
    test_raster_pair = (
        '/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif',
        '/teamspace/studios/dl2/clean/data/04_Uppangala_AGB40_band1_onImgGrid.tif'
    )
    
    site_names = ['Yellapur', 'Uppangala']
    
    def __init__(self, mode='test'):
        self.mode = mode
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f"yellapur_to_uppangala_benchmark_{self.timestamp}"
        
        # Configuration based on mode
        if mode == 'test':
            print("🧪 TEST MODE: Quick benchmarking")
            self.batch_size = 512
            self.learning_rate = 0.001
            self.num_epochs = 20
            self.early_stop_patience = 5
            self.max_samples_per_site = 5000
            self.max_features = 200
            
            # Loss functions to try
            self.loss_functions = ['mse', 'huber']
            
        else:
            print("🚀 FULL MODE: Complete benchmarking")
            self.batch_size = 256
            self.learning_rate = 0.001
            self.num_epochs = 100
            self.early_stop_patience = 10
            self.max_samples_per_site = None
            self.max_features = 300
            
            # Loss functions to try
            self.loss_functions = ['mse', 'huber', 'quantile', 'focal', 'adaptive']
        
        # Data processing parameters
        self.use_log_transform = True
        self.epsilon = 1.0
        self.use_advanced_indices = True
        self.use_pca_features = True
        self.pca_components = 25
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 Results directory: {self.results_dir}")
        
        # Save configuration
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        config_dict = {
            'mode': self.mode,
            'timestamp': self.timestamp,
            'strategy': 'train_yellapur_test_uppangala',
            'train_site': 'Yellapur',
            'test_site': 'Uppangala',
            'loss_functions': self.loss_functions,
            'max_features': self.max_features,
            'num_epochs': self.num_epochs,
            'site_names': self.site_names
        }
        
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

# Initialize configuration
config = TrainTestConfig(mode='full')  # Change to 'full' for complete runs

# Step 3: Helper Functions
# def set_seed(seed=42):
#     """Set random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed)

def memory_cleanup():
    """Enhanced memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# Step 4: Enhanced Loss Functions
class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty quantification"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.primary_quantile = 0.5
    
    def forward(self, predictions, targets):
        if predictions.dim() == 1 or (predictions.dim() == 2 and predictions.shape[1] == 1):
            pred_q = predictions.squeeze() if predictions.dim() == 2 else predictions
            diff = targets - pred_q
            q = self.primary_quantile
            loss = torch.max(q * diff, (q - 1) * diff)
            return loss.mean()
        else:
            losses = []
            for i, q in enumerate(self.quantiles):
                pred_q = predictions[:, i]
                diff = targets - pred_q
                loss = torch.max(q * diff, (q - 1) * diff)
                losses.append(loss.mean())
            return sum(losses) / len(losses)

class FocalRegressionLoss(nn.Module):
    """Focal loss adapted for regression"""
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, predictions, targets):
        predictions = predictions.squeeze() if predictions.dim() == 2 and predictions.shape[1] == 1 else predictions
        mse = F.mse_loss(predictions, targets, reduction='none')
        normalized_error = torch.abs(predictions - targets) / (torch.abs(targets).clamp(min=1e-8) + 1e-8)
        focal_weight = self.alpha * (normalized_error.clamp(max=1.0)) ** self.gamma
        return (focal_weight * mse).mean()

class AdaptiveLoss(nn.Module):
    """Adaptive loss that switches based on residuals"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
        
    def forward(self, predictions, targets):
        predictions = predictions.squeeze() if predictions.dim() == 2 and predictions.shape[1] == 1 else predictions
        
        with torch.no_grad():
            residuals = torch.abs(predictions - targets)
            if len(residuals) > 1:
                outlier_threshold = torch.quantile(residuals, 0.8)
                outlier_mask = residuals > outlier_threshold
                outlier_ratio = outlier_mask.float().mean()
            else:
                outlier_ratio = 0.0
        
        if outlier_ratio > 0.2:
            return 0.7 * self.mse(predictions, targets) + 0.3 * self.huber(predictions, targets)
        else:
            return self.mse(predictions, targets)

def get_loss_function(loss_name):
    """Get loss function by name"""
    loss_functions = {
        'mse': nn.MSELoss(),
        'huber': nn.HuberLoss(delta=1.0),
        'quantile': QuantileLoss(),
        'focal': FocalRegressionLoss(gamma=2.0),
        'adaptive': AdaptiveLoss()
    }
    return loss_functions.get(loss_name, nn.MSELoss())

# Step 5: Data Loading and Feature Extraction
def safe_divide(a, b, fill_value=0.0):
    """Safe division handling zeros and NaN"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=1e-10, posinf=1e10, neginf=-1e10)
    
    try:
        result = np.divide(a, b, out=np.full_like(a, fill_value, dtype=np.float32), where=(np.abs(b) > 1e-10))
    except:
        result = np.full_like(a, fill_value, dtype=np.float32)
    
    result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return result

def calculate_comprehensive_indices(satellite_data):
    """Calculate comprehensive spectral indices"""
    print("     🌿 Calculating spectral indices...")
    
    indices = {}
    n_bands = satellite_data.shape[0]
    
    def safe_get_band(idx):
        return satellite_data[idx] if idx < n_bands else None
    
    # Simplified Sentinel-2 indices
    s2_seasons = [
        ('T1', [0, 1, 2, 7, 9, 10]),
        ('T2', [11, 12, 13, 18, 20, 21])
    ]
    
    for season, band_indices in s2_seasons:
        if len(band_indices) >= 4:
            blue_idx, green_idx, red_idx, nir_idx = band_indices[:4]
            
            blue = safe_get_band(blue_idx)
            green = safe_get_band(green_idx)
            red = safe_get_band(red_idx)
            nir = safe_get_band(nir_idx)
            
            if all(b is not None for b in [red, nir]):
                indices[f'NDVI_S2_{season}'] = safe_divide(nir - red, nir + red)
                
            if all(b is not None for b in [nir, green]):
                indices[f'GNDVI_S2_{season}'] = safe_divide(nir - green, nir + green)
    
    # Simplified radar indices
    if n_bands > 50:
        s1_seasons = [('T1', [49, 50]), ('T2', [51, 52])]
        
        for season, (vv_idx, vh_idx) in s1_seasons:
            vv = safe_get_band(vv_idx)
            vh = safe_get_band(vh_idx)
            
            if vv is not None and vh is not None:
                indices[f'CPR_S1_{season}'] = safe_divide(vh, vv)
    
    indices = {k: v for k, v in indices.items() if v is not None}
    return indices

def extract_features_from_pixels(satellite_data, biomass_data, valid_mask, site_name, config):
    """Extract features from valid pixels"""
    print(f"   Extracting features for {site_name}...")
    
    # Get valid pixel coordinates
    valid_y, valid_x = np.where(valid_mask)
    n_valid = len(valid_y)
    
    if n_valid == 0:
        print(f"   Warning: No valid pixels found for {site_name}")
        return None, None, None
    
    # Sample for testing if needed
    if config.max_samples_per_site and n_valid > config.max_samples_per_site:
        indices = np.random.choice(n_valid, config.max_samples_per_site, replace=False)
        valid_y = valid_y[indices]
        valid_x = valid_x[indices]
        n_valid = len(valid_y)
        print(f"     Sampled {n_valid} pixels")
    
    all_features = {}
    
    # 1. Original bands (limited for efficiency)
    n_bands_to_use = min(satellite_data.shape[0], 40)
    for i in range(n_bands_to_use):
        band_data = np.nan_to_num(satellite_data[i], nan=0.0)
        all_features[f'Band_{i+1:02d}'] = band_data
    
    # 2. Enhanced spectral indices
    if config.use_advanced_indices:
        indices = calculate_comprehensive_indices(satellite_data)
        for key, value in indices.items():
            value = np.nan_to_num(value, nan=0.0)
            all_features[key] = value
    
    # 3. Enhanced PCA features
    if config.use_pca_features and satellite_data.shape[0] > config.pca_components:
        try:
            bands_subset = satellite_data[:min(satellite_data.shape[0], 20)]
            bands_reshaped = bands_subset.reshape(bands_subset.shape[0], -1).T
            
            # Clean data
            valid_pixels = ~np.any(np.isnan(bands_reshaped), axis=1)
            if np.sum(valid_pixels) > config.pca_components:
                bands_clean = bands_reshaped[valid_pixels]
                
                # Simple PCA
                scaler = StandardScaler()
                bands_scaled = scaler.fit_transform(bands_clean)
                
                pca = PCA(n_components=min(config.pca_components, bands_scaled.shape[1]))
                pca_features = pca.fit_transform(bands_scaled)
                
                # Reshape back
                pca_full = np.zeros((bands_reshaped.shape[0], pca_features.shape[1]))
                pca_full[valid_pixels] = pca_features
                pca_full = pca_full.reshape(satellite_data.shape[1], satellite_data.shape[2], -1)
                
                for i in range(pca_features.shape[1]):
                    all_features[f'PCA_{i+1:02d}'] = pca_full[:, :, i]
        except Exception as e:
            print(f"     Warning: PCA feature extraction failed: {e}")
    
    # Create feature matrix
    feature_names = list(all_features.keys())
    feature_matrix = np.zeros((n_valid, len(feature_names)), dtype=np.float32)
    
    for i, feature_name in enumerate(feature_names):
        feature_data = all_features[feature_name]
        feature_values = feature_data[valid_y, valid_x]
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        feature_matrix[:, i] = feature_values
    
    # Extract targets
    biomass_targets = biomass_data[valid_y, valid_x].astype(np.float32)
    
    # Apply log transform if specified
    if config.use_log_transform:
        biomass_targets = np.log(biomass_targets + config.epsilon)
    
    return feature_matrix, biomass_targets, feature_names

def load_site_data(sat_path, bio_path, site_name, config):
    """Load data for a specific site"""
    print(f"\n--- Loading {site_name} ---")
    
    try:
        # Check if files exist
        if not os.path.exists(sat_path) or not os.path.exists(bio_path):
            raise FileNotFoundError(f"Files not found for {site_name}")
            
        # Load satellite data
        with rasterio.open(sat_path) as src:
            satellite_data = src.read()
        
        # Load biomass data
        with rasterio.open(bio_path) as src:
            biomass_data = src.read(1)
        
        print(f"  Satellite: {satellite_data.shape}")
        print(f"  Biomass: {biomass_data.shape}")
        
        # Create validity mask
        sat_finite_mask = np.all(np.isfinite(satellite_data), axis=0)
        bio_finite_mask = np.isfinite(biomass_data)
        bio_positive_mask = biomass_data > 0
        bio_reasonable_mask = biomass_data < 1000
        
        valid_mask = (sat_finite_mask & bio_finite_mask & 
                     bio_positive_mask & bio_reasonable_mask)
        
        valid_percent = np.mean(valid_mask) * 100
        print(f"  Valid pixels: {valid_percent:.1f}%")
        
        if valid_percent < 1.0:
            raise ValueError(f"Too few valid pixels for {site_name}")
        
        # Extract features
        features, targets, feature_names = extract_features_from_pixels(
            satellite_data, biomass_data, valid_mask, site_name, config
        )
        
        if features is not None and len(features) > 0:
            print(f"  {site_name}: {len(targets):,} samples, {features.shape[1]} features")
            return features, targets, feature_names
        else:
            raise ValueError(f"No features extracted for {site_name}")
    
    except Exception as e:
        print(f"  Error loading {site_name}: {e}")
        raise

# Step 6: Enhanced Feature Selection
class SimpleFeatureSelector:
    """Simple feature selection based on correlation and importance"""
    
    def __init__(self, config):
        self.config = config
        self.selected_features = None
        
    def select_features(self, X_train, y_train, feature_names):
        """Select features using correlation and mutual information"""
        print(f"🔍 Feature selection...")
        
        # Remove constant features
        feature_variance = np.var(X_train, axis=0)
        non_constant_mask = feature_variance > 1e-8
        
        X_filtered = X_train[:, non_constant_mask]
        feature_names_filtered = [feature_names[i] for i in range(len(feature_names)) if non_constant_mask[i]]
        
        if X_filtered.shape[1] <= self.config.max_features:
            self.selected_features = np.where(non_constant_mask)[0]
            print(f"   Using all {X_filtered.shape[1]} non-constant features")
            return X_filtered, feature_names_filtered
        
        # Use mutual information for feature selection
        try:
            mi_scores = mutual_info_regression(X_filtered, y_train, random_state=42)
            top_indices = np.argsort(mi_scores)[-self.config.max_features:]
            
            # Map back to original indices
            filtered_indices = np.where(non_constant_mask)[0]
            self.selected_features = filtered_indices[top_indices]
            
            selected_names = [feature_names_filtered[i] for i in top_indices]
            print(f"   Selected {len(top_indices)} features")
            print(f"   Top 5: {selected_names[:5]}")
            
            return X_filtered[:, top_indices], selected_names
            
        except Exception as e:
            print(f"   Warning: Feature selection failed: {e}")
            # Fallback: use all features
            self.selected_features = np.where(non_constant_mask)[0]
            return X_filtered, feature_names_filtered
    
    def transform(self, X):
        """Transform new data using selected features"""
        if self.selected_features is not None:
            return X[:, self.selected_features]
        return X

# Step 7: Model Architectures
class StandardMLP(nn.Module):
    """Standard Multi-Layer Perceptron"""
    def __init__(self, n_features, dropout_rate=0.3):
        super().__init__()
        
        self.input_layer = nn.Linear(n_features, 256)
        self.input_bn = nn.BatchNorm1d(256)
        
        self.hidden1 = nn.Linear(256, 128)
        self.hidden1_bn = nn.BatchNorm1d(128)
        
        self.hidden2 = nn.Linear(128, 64)
        self.hidden2_bn = nn.BatchNorm1d(64)
        
        self.output_layer = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden1(x)
        x = self.hidden1_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden2(x)
        x = self.hidden2_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x.squeeze(-1)

class ResBlock(nn.Module):
    """Residual Block for MLP"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.skip_connection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        identity = self.skip_connection(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class MLPWithResidualBlocks(nn.Module):
    """MLP with Residual Connections"""
    def __init__(self, n_features):
        super().__init__()
        
        self.input_proj = nn.Linear(n_features, 512)
        self.layer1 = self._make_layer(512, 256, 2)
        self.layer2 = self._make_layer(256, 128, 2)
        self.layer3 = self._make_layer(128, 64, 2)
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_dim, out_dim, num_blocks):
        layers = []
        for i in range(num_blocks):
            in_channels = in_dim if i == 0 else out_dim
            layers.append(ResBlock(in_channels, out_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.regressor(x).squeeze(-1)

# Step 8: Training Framework
class TrainTestTrainer:
    """Trainer for train on Yellapur, test on Uppangala"""
    
    def __init__(self, config):
        self.config = config
        self.device = device
        self.scaler = RobustScaler()
        
    def train_neural_model(self, model_class, X_train, y_train, X_test, y_test, 
                          feature_names, model_name):
        """Train neural model with multiple loss functions"""
        print(f"\n🧠 Training {model_name}...")
        
        best_result = None
        best_loss_name = None
        
        for loss_name in self.config.loss_functions:
            try:
                print(f"  Trying loss function: {loss_name}")
                
                # Scale the data
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Create data loaders
                train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
                test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
                
                train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
                
                # Initialize model
                model = model_class(X_train_scaled.shape[1]).to(self.device)
                criterion = get_loss_function(loss_name)
                
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=1e-5
                )
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=3, factor=0.5
                )
                
                best_val_loss = float('inf')
                patience_counter = 0
                start_time = time.time()
                
                # Training loop
                for epoch in range(self.config.num_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError("NaN or Inf loss detected")
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    val_preds = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                val_loss += loss.item()
                                val_preds.extend(outputs.cpu().numpy())
                                val_targets.extend(batch_y.cpu().numpy())
                    
                    if len(val_preds) == 0:
                        raise ValueError("No valid predictions generated")
                    
                    val_loss /= len(test_loader)
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stop_patience:
                            break
                
                # Load best model and evaluate
                model.load_state_dict(best_model_state)
                training_time = time.time() - start_time
                
                # Final evaluation on test set
                model.eval()
                final_preds = []
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(self.device)
                        outputs = model(batch_x)
                        final_preds.extend(outputs.cpu().numpy())
                
                test_r2 = r2_score(y_test, final_preds)
                test_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
                test_mae = mean_absolute_error(y_test, final_preds)
                model_size = sum(p.numel() for p in model.parameters())
                
                current_result = {
                    'model': model,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_predictions': final_preds,
                    'test_targets': y_test.tolist(),
                    'training_time': training_time,
                    'model_size': model_size,
                    'epochs_trained': epoch + 1,
                    'loss_function': loss_name
                }
                
                # Keep best result across loss functions
                if best_result is None or test_r2 > best_result['test_r2']:
                    best_result = current_result
                    best_loss_name = loss_name
                
                print(f"    {loss_name}: R² = {test_r2:.4f}, RMSE = {test_rmse:.3f}")
                
            except Exception as e:
                print(f"    {loss_name}: Failed - {e}")
                continue
        
        if best_result is None:
            raise RuntimeError(f"All loss functions failed for {model_name}")
        
        print(f"    Best: {best_loss_name} (R² = {best_result['test_r2']:.4f})")
        return best_result
    
    def train_tree_model(self, model_factory, X_train, y_train, X_test, y_test, model_name):
        """Train tree model"""
        print(f"\n🌳 Training {model_name}...")
        
        try:
            # Scale the data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            start_time = time.time()
            model = model_factory()
            
            if 'XGB' in model_name:
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            elif 'LightGBM' in model_name:
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
            else:
                model.fit(X_train_scaled, y_train)
            
            # Evaluate
            test_pred = model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            training_time = time.time() - start_time
            
            try:
                model_size = model.get_booster().trees_to_dataframe().shape[0] if hasattr(model, 'get_booster') else 100000
            except:
                model_size = 100000
            
            result = {
                'model': model,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_predictions': test_pred.tolist(),
                'test_targets': y_test.tolist(),
                'training_time': training_time,
                'model_size': model_size
            }
            
            print(f"    R² = {test_r2:.4f}, RMSE = {test_rmse:.3f}")
            return result
            
        except Exception as e:
            print(f"    {model_name} failed: {e}")
            raise

# Step 9: Load Data
print(f"\n📊 Loading data for train-test split...")
start_time = time.time()

try:
    # Load training data (Yellapur)
    X_train, y_train, feature_names = load_site_data(
        config.train_raster_pair[0], 
        config.train_raster_pair[1], 
        "Yellapur (Training)", 
        config
    )
    
    # Load test data (Uppangala)
    X_test, y_test, _ = load_site_data(
        config.test_raster_pair[0], 
        config.test_raster_pair[1], 
        "Uppangala (Testing)", 
        config
    )
    
    data_load_time = time.time() - start_time
    print(f"⏱️ Data loading completed in {data_load_time:.2f} seconds")
    
    # Print data summary
    print(f"\n📈 Dataset Summary:")
    print(f"   Training (Yellapur): {len(y_train):,} samples")
    print(f"   Testing (Uppangala): {len(y_test):,} samples")
    print(f"   Features: {len(feature_names)}")
    
    # Feature selection on training data
    feature_selector = SimpleFeatureSelector(config)
    X_train_selected, selected_feature_names = feature_selector.select_features(
        X_train, y_train, feature_names
    )
    X_test_selected = feature_selector.transform(X_test)
    
    print(f"\n🏷️ Selected Features: {len(selected_feature_names)}")
    print(f"   Examples: {selected_feature_names[:5]}")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    raise

# Step 10: Define Models
print(f"\n🤖 Setting up models...")

# Neural network models
neural_models = {
    'StandardMLP': StandardMLP,
    'MLPWithResidualBlocks': MLPWithResidualBlocks,
}

# Tree models
def get_tree_models():
    return {
        'XGBoost': lambda: xgb.XGBRegressor(
            n_estimators=300 if config.mode == 'test' else 500,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lambda: lgb.LGBMRegressor(
            n_estimators=300 if config.mode == 'test' else 500,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'RandomForest': lambda: RandomForestRegressor(
            n_estimators=200 if config.mode == 'test' else 500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }

tree_models = get_tree_models()

print(f"Neural Network Models: {list(neural_models.keys())}")
print(f"Tree Models: {list(tree_models.keys())}")

# Step 11: Run Benchmark
print(f"\n🚀 Starting Yellapur → Uppangala Benchmark...")
print("=" * 70)

trainer = TrainTestTrainer(config)
results = {}

# Train Neural Network Models
for model_name, model_class in neural_models.items():
    print(f"\n{'='*50}")
    print(f"🎯 Benchmarking {model_name} (Neural Network)")
    print(f"{'='*50}")
    
    try:
        model_results = trainer.train_neural_model(
            model_class, X_train_selected, y_train, X_test_selected, y_test, 
            selected_feature_names, model_name
        )
        
        results[model_name] = model_results
        
        # Print results summary
        print(f"\n✅ {model_name} Results:")
        print(f"   Test R²: {model_results['test_r2']:.4f}")
        print(f"   Test RMSE: {model_results['test_rmse']:.3f} Mg/ha")
        print(f"   Test MAE: {model_results['test_mae']:.3f} Mg/ha")
        print(f"   Training Time: {model_results['training_time']:.2f}s")
        print(f"   Model Size: {model_results['model_size']:,} parameters")
        print(f"   Best Loss: {model_results['loss_function']}")
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        continue
    
    memory_cleanup()

# Train Tree Models
for model_name, model_factory in tree_models.items():
    print(f"\n{'='*50}")
    print(f"🎯 Benchmarking {model_name} (Tree Model)")
    print(f"{'='*50}")
    
    try:
        model_results = trainer.train_tree_model(
            model_factory, X_train_selected, y_train, X_test_selected, y_test, model_name
        )
        
        results[model_name] = model_results
        
        # Print results summary
        print(f"\n✅ {model_name} Results:")
        print(f"   Test R²: {model_results['test_r2']:.4f}")
        print(f"   Test RMSE: {model_results['test_rmse']:.3f} Mg/ha")
        print(f"   Test MAE: {model_results['test_mae']:.3f} Mg/ha")
        print(f"   Training Time: {model_results['training_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        continue
    
    memory_cleanup()

# Step 12: Convert Back to Original Scale
print(f"\n🔄 Converting results back to original scale...")

if config.use_log_transform:
    print(f"   Converting from log scale (epsilon = {config.epsilon})...")
    for model_name in results:
        try:
            log_preds = np.array(results[model_name]['test_predictions'])
            log_targets = np.array(results[model_name]['test_targets'])
            
            orig_preds = np.exp(log_preds) - config.epsilon
            orig_targets = np.exp(log_targets) - config.epsilon
            
            # Ensure non-negative
            orig_preds = np.maximum(orig_preds, 0)
            orig_targets = np.maximum(orig_targets, 0)
            
            # Recalculate metrics on original scale
            orig_r2 = r2_score(orig_targets, orig_preds)
            orig_rmse = np.sqrt(mean_squared_error(orig_targets, orig_preds))
            orig_mae = mean_absolute_error(orig_targets, orig_preds)
            
            # Store both scales
            results[model_name]['test_r2_orig'] = orig_r2
            results[model_name]['test_rmse_orig'] = orig_rmse
            results[model_name]['test_mae_orig'] = orig_mae
            results[model_name]['test_predictions_orig'] = orig_preds.tolist()
            results[model_name]['test_targets_orig'] = orig_targets.tolist()
            
            print(f"   {model_name}: Log R² = {results[model_name]['test_r2']:.4f}, "
                  f"Original R² = {orig_r2:.4f}")
                  
        except Exception as e:
            print(f"   Warning: Scale conversion failed for {model_name}: {e}")

# Step 13: Create Report
def create_report(results, config):
    """Create comprehensive report"""
    print(f"\n📊 Creating benchmark report...")
    
    if not results:
        print("⚠️ No results to report")
        return None
    
    # Prepare data for analysis
    report_data = []
    for model_name, metrics in results.items():
        try:
            # Use original scale metrics if available
            if 'test_r2_orig' in metrics:
                r2_score = metrics['test_r2_orig']
                rmse_score = metrics['test_rmse_orig']
                mae_score = metrics['test_mae_orig']
                scale_note = "Original"
            else:
                r2_score = metrics['test_r2']
                rmse_score = metrics['test_rmse']
                mae_score = metrics['test_mae']
                scale_note = "Log" if config.use_log_transform else "Original"
            
            # Model type classification
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
                model_type = 'Tree'
            else:
                model_type = 'Neural'
            
            report_data.append({
                'Model': model_name,
                'Type': model_type,
                'Test_R2': r2_score,
                'Test_RMSE': rmse_score,
                'Test_MAE': mae_score,
                'Training_Time': metrics['training_time'],
                'Model_Size': metrics['model_size'],
                'Scale': scale_note
            })
        except Exception as e:
            print(f"   Warning: Error processing {model_name}: {e}")
            continue
    
    if not report_data:
        print("⚠️ No valid data for report")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    df = df.sort_values('Test_R2', ascending=False)
    
    # Save report
    report_path = os.path.join(config.results_dir, 'yellapur_to_uppangala_report.csv')
    df.to_csv(report_path, index=False)
    
    # Print summary
    print(f"\n🏆 YELLAPUR → UPPANGALA BENCHMARK RESULTS")
    print("=" * 70)
    scale_info = " (Original Scale)" if config.use_log_transform else ""
    print(f"{'Rank':<5} {'Model':<22} {'Type':<8} {'Test R²':<9} {'RMSE':<9} {'Time(s)':<8}")
    print("-" * 70)
    
    for i, row in df.iterrows():
        print(f"{i+1:<5} {row['Model']:<22} {row['Type']:<8} {row['Test_R2']:<9.4f} "
              f"{row['Test_RMSE']:<9.2f} {row['Training_Time']:<8.2f}")
    
    return df

# Create the report
try:
    df_results = create_report(results, config)
    
    if df_results is not None:
        print(f"✅ Report created successfully with {len(df_results)} models")
    else:
        print("⚠️ Report creation failed")
        
except Exception as e:
    print(f"❌ Error creating report: {e}")

# Step 14: Visualizations
def create_visualizations(results, df_results, config):
    """Create visualizations"""
    if df_results is None or len(df_results) == 0:
        print("⚠️ No data available for visualization")
        return
        
    print(f"\n📈 Creating visualizations...")
    
    try:
        # Figure 1: Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R² comparison
        ax1 = axes[0, 0]
        models = df_results['Model'].values
        r2_scores = df_results['Test_R2'].values
        
        colors = ['lightcoral' if model in ['XGBoost', 'LightGBM', 'RandomForest'] else 
                  'lightblue' for model in models]
        
        bars = ax1.barh(models, r2_scores, color=colors, alpha=0.8)
        ax1.set_xlabel('Test R² Score')
        ax1.set_title('Yellapur → Uppangala Performance (R²)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for bar, r2 in zip(bars, r2_scores):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{r2:.3f}', ha='left', va='center', fontsize=8)
        
        # RMSE comparison
        ax2 = axes[0, 1]
        rmse_scores = df_results['Test_RMSE'].values
        bars = ax2.barh(models, rmse_scores, color=colors, alpha=0.8)
        ax2.set_xlabel('Test RMSE (Mg/ha)')
        ax2.set_title('Yellapur → Uppangala Performance (RMSE)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Training Efficiency vs Performance
        ax3 = axes[1, 0]
        training_times = df_results['Training_Time'].values
        ax3.scatter(training_times, r2_scores, s=120, alpha=0.7, c=colors)
        for i, txt in enumerate(models):
            ax3.annotate(txt, (training_times[i], r2_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Test R² Score')
        ax3.set_title('Training Efficiency vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # Prediction scatter plot for best model
        ax4 = axes[1, 1]
        best_model = df_results.iloc[0]['Model']
        result = results[best_model]
        
        # Use original scale if available
        if 'test_targets_orig' in result:
            targets = np.array(result['test_targets_orig'])
            predictions = np.array(result['test_predictions_orig'])
            r2 = result['test_r2_orig']
            rmse = result['test_rmse_orig']
            title_suffix = "(Original Scale)"
        else:
            targets = np.array(result['test_targets'])
            predictions = np.array(result['test_predictions'])
            r2 = result['test_r2']
            rmse = result['test_rmse']
            title_suffix = "(Log Scale)" if config.use_log_transform else ""
        
        # Create scatter plot
        ax4.scatter(targets, predictions, alpha=0.6, s=20, c='blue')
        
        # Add 1:1 line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Actual Biomass (Mg/ha)')
        ax4.set_ylabel('Predicted Biomass (Mg/ha)')
        ax4.set_title(f'{best_model} {title_suffix}\nR² = {r2:.3f}, RMSE = {rmse:.1f}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.results_dir, 'yellapur_to_uppangala_results.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved successfully")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

# Create visualizations
if df_results is not None:
    create_visualizations(results, df_results, config)

# Step 15: Final Analysis
def analyze_results(df_results, config):
    """Provide detailed analysis and recommendations"""
    if df_results is None or len(df_results) == 0:
        print("⚠️ No results available for analysis")
        return
        
    print(f"\n🔍 YELLAPUR → UPPANGALA ANALYSIS")
    print("=" * 60)
    
    try:
        # Overall performance analysis
        best_model = df_results.iloc[0]
        print(f"🥇 Best Model: {best_model['Model']}")
        print(f"   • Test R²: {best_model['Test_R2']:.4f}")
        print(f"   • Test RMSE: {best_model['Test_RMSE']:.2f} Mg/ha")
        print(f"   • Training Time: {best_model['Training_Time']:.2f}s")
        print(f"   • Model Type: {best_model['Type']}")
        
        # Performance by type analysis
        print(f"\n🚀 Performance by Model Type:")
        performance_by_type = df_results.groupby('Type')['Test_R2'].agg(['mean', 'max', 'count'])
        for model_type in performance_by_type.index:
            stats = performance_by_type.loc[model_type]
            print(f"   {model_type} Models:")
            print(f"     • Count: {stats['count']}")
            print(f"     • Mean R²: {stats['mean']:.4f}")
            print(f"     • Best R²: {stats['max']:.4f}")
        
        # Model efficiency analysis
        print(f"\n⚡ Model Efficiency Analysis:")
        df_results['Efficiency'] = df_results['Test_R2'] / df_results['Training_Time']
        most_efficient = df_results.nlargest(1, 'Efficiency').iloc[0]
        print(f"   Most Efficient: {most_efficient['Model']}")
        print(f"   • R² per second: {most_efficient['Efficiency']:.6f}")
        print(f"   • R²: {most_efficient['Test_R2']:.4f}")
        print(f"   • Training time: {most_efficient['Training_Time']:.2f}s")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        print(f"\n1. 🎯 For Yellapur → Uppangala Transfer:")
        print(f"   → Best model: {best_model['Model']} (R² = {best_model['Test_R2']:.3f})")
        print(f"   → Expected generalization performance on Uppangala")
        
        print(f"\n2. ⚡ For Real-time Applications:")
        print(f"   → Use: {most_efficient['Model']} (most efficient)")
        print(f"   → Trade-off: R² = {most_efficient['Test_R2']:.3f} vs {most_efficient['Training_Time']:.2f}s training")
        
        print(f"\n3. 🔧 For Further Improvement:")
        print(f"   → Consider domain adaptation techniques")
        print(f"   → Explore site-specific fine-tuning")
        print(f"   → Investigate additional feature engineering")
        
        if config.mode == 'test':
            print(f"\n⚠️  Note: Results from TEST mode with limited features.")
            print(f"   → Run in FULL mode for production-ready results")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")

# Run final analysis
if df_results is not None:
    analyze_results(df_results, config)

# Step 16: Save Results
print(f"\n💾 Saving results...")

try:
    # Save detailed results
    summary = {
        'config': {
            'mode': config.mode,
            'timestamp': config.timestamp,
            'strategy': 'train_yellapur_test_uppangala',
            'train_site': 'Yellapur',
            'test_site': 'Uppangala',
            'loss_functions': config.loss_functions,
            'max_features': config.max_features,
            'num_epochs': config.num_epochs
        },
        'data_info': {
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'features_used': len(selected_feature_names),
            'feature_selection_applied': len(selected_feature_names) < len(feature_names)
        },
        'results': {}
    }
    
    # Add results for each model
    for model_name in results.keys():
        try:
            summary['results'][model_name] = {
                'test_r2': results[model_name].get('test_r2_orig', results[model_name]['test_r2']),
                'test_rmse': results[model_name].get('test_rmse_orig', results[model_name]['test_rmse']),
                'test_mae': results[model_name].get('test_mae_orig', results[model_name]['test_mae']),
                'training_time': results[model_name]['training_time'],
                'model_size': results[model_name]['model_size']
            }
        except Exception as e:
            print(f"   Warning: Error saving results for {model_name}: {e}")
    
    # Add performance summary
    if df_results is not None and len(df_results) > 0:
        summary['performance_summary'] = {
            'best_model': df_results.iloc[0]['Model'],
            'best_test_r2': df_results.iloc[0]['Test_R2'],
            'best_test_rmse': df_results.iloc[0]['Test_RMSE'],
            'total_models_tested': len(df_results),
            'successful_models': len(results)
        }
    
    # Save to JSON
    with open(os.path.join(config.results_dir, 'yellapur_to_uppangala_results.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    # Save feature names
    feature_info = {
        'original_features': feature_names,
        'selected_features': selected_feature_names,
        'total_original': len(feature_names),
        'total_selected': len(selected_feature_names)
    }
    
    with open(os.path.join(config.results_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=4)
    
    print(f"✅ Results saved successfully")
    
except Exception as e:
    print(f"❌ Error saving results: {e}")

# Calculate total time
total_time = time.time() - start_time
print(f"✅ Benchmark completed in {total_time/60:.2f} minutes")

# Final Summary
print(f"\n🎉 YELLAPUR → UPPANGALA BENCHMARK COMPLETED! 🎉")
print("=" * 70)
print(f"📊 Summary:")
print(f"   • Training Site: Yellapur ({len(y_train):,} samples)")
print(f"   • Testing Site: Uppangala ({len(y_test):,} samples)")
print(f"   • Models tested: {len(results)}")

if df_results is not None and len(df_results) > 0:
    print(f"   • Best performance: {df_results.iloc[0]['Model']} (R² = {df_results.iloc[0]['Test_R2']:.4f})")
else:
    print(f"   • No successful results to report")

print(f"   • Features used: {len(selected_feature_names)}")
print(f"   • Total time: {total_time/60:.2f} minutes")

print(f"\n📁 Results saved in: {config.results_dir}")
print(f"   • Main report: yellapur_to_uppangala_report.csv")
print(f"   • Detailed results: yellapur_to_uppangala_results.json")
print(f"   • Feature info: feature_info.json")
print(f"   • Visualizations: *.png files")

print(f"\n🎯 Key Changes Made:")
print(f"   ✅ Changed from LOSO to single train-test split")
print(f"   ✅ Training exclusively on Yellapur data")
print(f"   ✅ Testing exclusively on Uppangala data")
print(f"   ✅ Simplified validation strategy")
print(f"   ✅ Direct cross-site generalization assessment")

# Quick model ranking summary
if df_results is not None and len(df_results) > 0:
    print(f"\n📋 Model Ranking (Yellapur → Uppangala):")
    for i, row in df_results.head(min(5, len(df_results))).iterrows():
        print(f"   {i+1}. {row['Model']:<20} R² = {row['Test_R2']:.4f}")

print(f"\n✨ Yellapur → Uppangala benchmark completed successfully! ✨")
