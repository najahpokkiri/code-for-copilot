i would like to train on yellapur and test on uppangala. rewirt the script for thatt? any suggested changes?

please tell me your plan first. do no code yet
# Enhanced Biomass Prediction Architecture Benchmarking with LOSO Cross-Validation
# FULLY FIXED VERSION - All errors resolved
# ============================================================================

# # Step 1: Setup and Imports
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
from itertools import combinations

# Core ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Enhanced ML imports
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_regression
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
from rasterio.windows import Window
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel, laplace
from skimage.morphology import disk
from skimage.filters.rank import entropy
from scipy import ndimage
from scipy.stats import skew, kurtosis

# Statistical tools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

# Suppress warnings
warnings.filterwarnings('ignore')

print("🌟 ENHANCED BIOMASS PREDICTION ARCHITECTURE BENCHMARK WITH LOSO - FIXED VERSION")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"User: najahpokkiri")
print(f"Improvements: Phase 1 & Phase 2 with LOSO Cross-Validation - ALL ERRORS FIXED")

# # Step 2: Enhanced Configuration with Phase Controls
class EnhancedBenchmarkConfig:
    """Enhanced configuration with Phase 1 & 2 improvements"""
    
    # Data paths for tiles
    raster_pairs = [
        ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif",
         "/teamspace/studios/dl2/clean/data/agbd_yellapur_reprojected_1.tif"),
             ('/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif',
         '/teamspace/studios/dl2/clean/data/04_Uppangala_AGB40_band1_onImgGrid.tif'
        )
    ]
    
    site_names = ['Yellapur', 'Uppangala']
    
    def __init__(self, mode='test'):
        self.mode = mode
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f"enhanced_biomass_benchmark_{self.timestamp}"
        
        # Phase 1 & 2 Configuration
        if mode == 'test':
            print("🧪 TEST MODE: Quick benchmarking with Phase 1 & 2")
            self.batch_size = 512
            self.learning_rate = 0.001
            self.num_epochs = 15
            self.early_stop_patience = 5
            self.max_samples_per_tile = 3000
            self.max_features = 150
            
            # Phase 1 settings (test)
            self.loss_functions = ['mse', 'huber']
            self.ensemble_size = 3
            
            # Phase 2 settings (test)
            self.spatial_scales = []  # Disabled for quick test
            self.temporal_features = 'basic'
            self.loso_folds = 2  # Only 2 sites for quick test
            
        else:
            print("🚀 FULL MODE: Complete Phase 1 & 2 benchmarking")
            self.batch_size = 256
            self.learning_rate = 0.001
            self.num_epochs = 75
            self.early_stop_patience = 10
            self.max_samples_per_tile = None
            self.max_features = 300
            
            # Phase 1 settings (full)
            self.loss_functions = ['mse', 'huber', 'quantile', 'focal', 'adaptive']
            self.ensemble_size = 5
            
            # Phase 2 settings (full)
            #self.spatial_scales = ['3x3', '5x5', '7x7', '9x9']
            self.spatial_scales = []
            self.temporal_features = 'complete'
            self.loso_folds = 4  # Full LOSO
        
        # Data processing parameters
        self.use_log_transform = True
        self.epsilon = 1.0
        self.use_advanced_indices = True
        self.use_pca_features = True
        self.pca_components = 25
        
        # Phase 1 & 2 feature flags
        self.enable_phase1 = True
        self.enable_phase2 = True
        self.enable_loso = True
        
        # Site-aware processing
        self.site_aware_scaling = True
        self.site_adaptive_ensemble = True
        
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
            'phase1_enabled': self.enable_phase1,
            'phase2_enabled': self.enable_phase2,
            'loso_enabled': self.enable_loso,
            'loss_functions': self.loss_functions,
            'spatial_scales': self.spatial_scales,
            'max_features': self.max_features,
            'num_epochs': self.num_epochs,
            'site_names': self.site_names
        }
        
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

# Initialize configuration
config = EnhancedBenchmarkConfig(mode='test')  # Change to 'full' for complete runs

# # Step 3: Enhanced Helper Functions
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def memory_cleanup():
    """Enhanced memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# # Step 4: Phase 1 - Enhanced Loss Functions (FIXED)
class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty quantification - FIXED for single outputs"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.primary_quantile = 0.5  # Use median for single output models
    
    def forward(self, predictions, targets):
        # FIXED: Handle single output models
        if predictions.dim() == 1 or (predictions.dim() == 2 and predictions.shape[1] == 1):
            # Single output - use median quantile (0.5)
            pred_q = predictions.squeeze() if predictions.dim() == 2 else predictions
            diff = targets - pred_q
            q = self.primary_quantile
            loss = torch.max(q * diff, (q - 1) * diff)
            return loss.mean()
        else:
            # Multiple outputs - use all quantiles
            losses = []
            for i, q in enumerate(self.quantiles):
                pred_q = predictions[:, i]
                diff = targets - pred_q
                loss = torch.max(q * diff, (q - 1) * diff)
                losses.append(loss.mean())
            return sum(losses) / len(losses)

class FocalRegressionLoss(nn.Module):
    """Focal loss adapted for regression - FIXED"""
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, predictions, targets):
        # FIXED: Handle tensor shapes properly
        predictions = predictions.squeeze() if predictions.dim() == 2 and predictions.shape[1] == 1 else predictions
        mse = F.mse_loss(predictions, targets, reduction='none')
        # Normalize error safely
        normalized_error = torch.abs(predictions - targets) / (torch.abs(targets).clamp(min=1e-8) + 1e-8)
        focal_weight = self.alpha * (normalized_error.clamp(max=1.0)) ** self.gamma
        return (focal_weight * mse).mean()

class AdaptiveLoss(nn.Module):
    """Adaptive loss that switches based on residuals - FIXED"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
        
    def forward(self, predictions, targets):
        # FIXED: Handle tensor shapes properly
        predictions = predictions.squeeze() if predictions.dim() == 2 and predictions.shape[1] == 1 else predictions
        
        with torch.no_grad():
            residuals = torch.abs(predictions - targets)
            if len(residuals) > 1:
                outlier_threshold = torch.quantile(residuals, 0.8)
                outlier_mask = residuals > outlier_threshold
                outlier_ratio = outlier_mask.float().mean()
            else:
                outlier_ratio = 0.0
        
        # Use MSE for normal cases, Huber for outlier-heavy cases
        if outlier_ratio > 0.2:
            return 0.7 * self.mse(predictions, targets) + 0.3 * self.huber(predictions, targets)
        else:
            return self.mse(predictions, targets)

def get_loss_function(loss_name):
    """Get loss function by name - FIXED"""
    loss_functions = {
        'mse': nn.MSELoss(),
        'huber': nn.HuberLoss(delta=1.0),
        'quantile': QuantileLoss(),
        'focal': FocalRegressionLoss(gamma=2.0),
        'adaptive': AdaptiveLoss()
    }
    return loss_functions.get(loss_name, nn.MSELoss())

# # Step 5: Phase 1 - LOSO-Aware Feature Selection (FIXED)
class LOSOFeatureSelector:
    """LOSO-aware feature selection - FIXED"""
    
    def __init__(self, config):
        self.config = config
        self.selected_features = None
        self.feature_importance_scores = {}
        
    def select_features(self, X, y, site_labels, feature_names):
        """Select features using LOSO-aware methods - FIXED"""
        print(f"🔍 LOSO-aware feature selection...")
        
        n_sites = len(np.unique(site_labels))
        feature_stability_scores = np.zeros(X.shape[1])
        feature_importance_scores = np.zeros(X.shape[1])
        
        # Cross-site stability analysis
        for site_id in np.unique(site_labels):
            train_mask = site_labels != site_id
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            if len(X_train) < 100:  # Skip if too few samples
                continue
                
            try:
                # Quick Random Forest for feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                
                # Accumulate importance scores
                feature_importance_scores += rf.feature_importances_
                
                # Stability: correlation with biomass per site
                for i in range(X.shape[1]):
                    if np.std(X_train[:, i]) > 1e-8:  # Avoid constant features
                        try:
                            corr, _ = pearsonr(X_train[:, i], y_train)
                            if not np.isnan(corr):
                                feature_stability_scores[i] += abs(corr)
                        except:
                            continue
            except Exception as e:
                print(f"   Warning: Feature selection failed for site {site_id}: {e}")
                continue
        
        # Average across sites
        if n_sites > 0:
            feature_importance_scores /= n_sites
            feature_stability_scores /= n_sites
        
        # Combined score: importance + stability
        combined_scores = 0.6 * feature_importance_scores + 0.4 * feature_stability_scores
        
        # Select top features
        n_features_to_select = min(self.config.max_features, X.shape[1])
        top_indices = np.argsort(combined_scores)[-n_features_to_select:]
        
        self.selected_features = top_indices
        self.feature_importance_scores = {
            'importance': feature_importance_scores,
            'stability': feature_stability_scores,
            'combined': combined_scores
        }
        
        selected_names = [feature_names[i] for i in top_indices]
        print(f"   Selected {len(top_indices)} features")
        print(f"   Top 5: {selected_names[:5]}")
        
        return X[:, top_indices], selected_names

# # Step 6: Phase 2 - Multi-Scale Spatial Features (OPTIMIZED)
def calculate_multiscale_spatial_features(satellite_data, scales=['3x3', '5x5']):
    """Calculate multi-scale spatial features - OPTIMIZED"""
    if not scales:  # Skip if no scales specified
        return {}
        
    print(f"📐 Calculating multi-scale spatial features: {scales}")
    
    spatial_features = {}
    
    # Convert scale strings to window sizes
    scale_sizes = {
        '3x3': 3, '5x5': 5, '7x7': 7, '9x9': 9
    }
    
    # Key bands for spatial analysis (limited for efficiency)
    key_bands = [7, 18, 29, 34][:min(4, satellite_data.shape[0])]
    
    for scale_name in scales:
        if scale_name not in scale_sizes:
            continue
            
        window_size = scale_sizes[scale_name]
        print(f"   Processing {scale_name} scale...")
        
        for band_idx in key_bands:
            if band_idx >= satellite_data.shape[0]:
                continue
                
            band = satellite_data[band_idx].astype(np.float32)
            
            # Handle NaN values
            band = np.nan_to_num(band, nan=np.nanmedian(band) if not np.all(np.isnan(band)) else 0.0)
            
            # Calculate neighborhood statistics (simplified for efficiency)
            try:
                # Basic statistics only
                mean_img = ndimage.uniform_filter(band, size=window_size)
                var_img = ndimage.generic_filter(band, np.var, size=window_size)
                
                # Store features
                prefix = f'Spatial_{scale_name}_B{band_idx}'
                spatial_features[f'{prefix}_mean'] = mean_img
                spatial_features[f'{prefix}_var'] = var_img
                
            except Exception as e:
                print(f"     Warning: Error in spatial features for band {band_idx}: {e}")
                continue
    
    print(f"   ✅ Generated {len(spatial_features)} spatial features")
    return spatial_features

def calculate_enhanced_texture_features(satellite_data, scales=['3x3', '5x5']):
    """Calculate enhanced texture features - OPTIMIZED"""
    if not scales:  # Skip if no scales specified
        return {}
        
    print(f"🖼️ Calculating enhanced texture features...")
    
    texture_features = {}
    key_bands = [7, 18][:min(2, satellite_data.shape[0])]  # Reduced for efficiency
    
    for band_idx in key_bands:
        if band_idx >= satellite_data.shape[0]:
            continue
            
        band = satellite_data[band_idx]
        
        # Normalize to 0-255 for texture analysis
        try:
            band_norm = band.copy()
            band_min, band_max = np.nanpercentile(band_norm, [5, 95])  # More robust percentiles
            if band_max > band_min:
                band_norm = np.clip((band_norm - band_min) / (band_max - band_min), 0, 1)
                band_norm = (band_norm * 255).astype(np.uint8)
            else:
                band_norm = np.zeros_like(band, dtype=np.uint8)
            band_norm = np.nan_to_num(band_norm, nan=128)  # Use neutral value for NaN
            
            # Simple texture features only
            # Edge-based texture
            sobel_response = sobel(band_norm.astype(float))
            texture_features[f'Edge_B{band_idx}_mean'] = float(np.mean(sobel_response))
            texture_features[f'Edge_B{band_idx}_std'] = float(np.std(sobel_response))
            
        except Exception as e:
            print(f"     Warning: Texture calculation failed for band {band_idx}: {e}")
            continue
    
    print(f"   ✅ Generated {len(texture_features)} texture features")
    return texture_features

# # Step 7: Phase 2 - Enhanced Temporal Features (OPTIMIZED)
def calculate_enhanced_temporal_features(satellite_data, temporal_mode='basic'):
    """Calculate enhanced temporal features - OPTIMIZED"""
    print(f"📅 Calculating temporal features (mode: {temporal_mode})...")
    
    temporal_features = {}
    
    # Sentinel-2 temporal analysis (3 seasons) - simplified
    s2_seasons = {
        'T1': [0, 1, 2, 7],      # Season 1 bands (reduced)
        'T2': [11, 12, 13, 18],  # Season 2 bands  
        'T3': [22, 23, 24, 29]   # Season 3 bands
    }
    
    # Basic temporal features only
    try:
        for band_type, (t1_idx, t2_idx, t3_idx, nir_idx) in enumerate(zip(*s2_seasons.values())):
            if all(idx < satellite_data.shape[0] for idx in [t1_idx, t2_idx, t3_idx]):
                t1_band = satellite_data[t1_idx]
                t2_band = satellite_data[t2_idx] 
                t3_band = satellite_data[t3_idx]
                
                # Basic differences only
                temporal_features[f'S2_B{band_type}_T2T1_diff'] = t2_band - t1_band
                temporal_features[f'S2_B{band_type}_T3T2_diff'] = t3_band - t2_band
                
                # Safe ratios
                t1_safe = np.where(np.abs(t1_band) < 1e-8, 1e-8, t1_band)
                temporal_features[f'S2_B{band_type}_T2T1_ratio'] = t2_band / t1_safe
    except Exception as e:
        print(f"   Warning: Temporal feature calculation failed: {e}")
    
    print(f"   ✅ Generated {len(temporal_features)} temporal features")
    return temporal_features

# # Step 8: Enhanced Data Loading with Site Information (FIXED)
def extract_enhanced_features_from_pixels(satellite_data, biomass_data, valid_mask, site_id, config):
    """Extract enhanced features with site information - FIXED"""
    print(f"   Extracting enhanced features for site {config.site_names[site_id]}...")
    
    # Get valid pixel coordinates
    valid_y, valid_x = np.where(valid_mask)
    n_valid = len(valid_y)
    
    if n_valid == 0:
        print(f"   Warning: No valid pixels found for site {config.site_names[site_id]}")
        return None, None, None, None
    
    # Sample for testing if needed
    if config.max_samples_per_tile and n_valid > config.max_samples_per_tile:
        indices = np.random.choice(n_valid, config.max_samples_per_tile, replace=False)
        valid_y = valid_y[indices]
        valid_x = valid_x[indices]
        n_valid = len(valid_y)
        print(f"     Sampled {n_valid} pixels")
    
    all_features = {}
    
    # 1. Original bands (limited for efficiency)
    n_bands_to_use = min(satellite_data.shape[0], 30)  # Reduced from 50
    for i in range(n_bands_to_use):
        band_data = np.nan_to_num(satellite_data[i], nan=0.0)
        all_features[f'Band_{i+1:02d}'] = band_data
    
    # 2. Enhanced spectral indices
    if config.use_advanced_indices:
        indices = calculate_comprehensive_indices(satellite_data)
        for key, value in indices.items():
            value = np.nan_to_num(value, nan=0.0)
            all_features[key] = value
    
    # 3. Phase 2: Multi-scale spatial features
    if config.enable_phase2:
        spatial_features = calculate_multiscale_spatial_features(
            satellite_data, config.spatial_scales
        )
        all_features.update(spatial_features)
        
        # Enhanced texture features
        texture_features = calculate_enhanced_texture_features(
            satellite_data, config.spatial_scales
        )
        
        # Convert scalar texture features to images
        for key, value in texture_features.items():
            if np.isscalar(value):
                # Broadcast scalar to image
                texture_img = np.full(satellite_data.shape[1:], value, dtype=np.float32)
                all_features[key] = texture_img
    
    # 4. Phase 2: Enhanced temporal features  
    if config.enable_phase2:
        temporal_features = calculate_enhanced_temporal_features(
            satellite_data, config.temporal_features
        )
        all_features.update(temporal_features)
    
    # 5. Enhanced PCA features (simplified)
    if config.use_pca_features and satellite_data.shape[0] > config.pca_components:
        try:
            bands_subset = satellite_data[:min(satellite_data.shape[0], 15)]  # Reduced from 20
            bands_reshaped = bands_subset.reshape(bands_subset.shape[0], -1).T
            
            # Clean data
            valid_pixels = ~np.any(np.isnan(bands_reshaped), axis=1)
            if np.sum(valid_pixels) > config.pca_components:
                bands_clean = bands_reshaped[valid_pixels]
                
                # Simple PCA without outlier removal for efficiency
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
    
    # Create site labels
    site_labels = np.full(n_valid, site_id, dtype=int)
    
    return feature_matrix, biomass_targets, site_labels, feature_names

def safe_divide(a, b, fill_value=0.0):
    """Safe division handling zeros and NaN - FIXED"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=1e-10, posinf=1e10, neginf=-1e10)
    
    # Handle broadcasting
    try:
        result = np.divide(a, b, out=np.full_like(a, fill_value, dtype=np.float32), where=(np.abs(b) > 1e-10))
    except:
        result = np.full_like(a, fill_value, dtype=np.float32)
    
    result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return result

def calculate_comprehensive_indices(satellite_data):
    """Calculate comprehensive spectral indices - OPTIMIZED"""
    print("     🌿 Calculating spectral indices...")
    
    indices = {}
    n_bands = satellite_data.shape[0]
    
    def safe_get_band(idx):
        return satellite_data[idx] if idx < n_bands else None
    
    # Simplified Sentinel-2 indices (fewer seasons for efficiency)
    s2_seasons = [
        ('T1', [0, 1, 2, 7, 9, 10]),
        ('T2', [11, 12, 13, 18, 20, 21])
    ]
    
    for season, band_indices in s2_seasons:
        if len(band_indices) >= 4:  # Need at least 4 bands
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
    if n_bands > 50:  # Check if we have radar data
        s1_seasons = [('T1', [49, 50]), ('T2', [51, 52])]
        
        for season, (vv_idx, vh_idx) in s1_seasons:
            vv = safe_get_band(vv_idx)
            vh = safe_get_band(vh_idx)
            
            if vv is not None and vh is not None:
                indices[f'CPR_S1_{season}'] = safe_divide(vh, vv)
    
    # Clean up None values
    indices = {k: v for k, v in indices.items() if v is not None}
    return indices

def load_enhanced_biomass_data(config):
    """Load biomass data with enhanced features and site information - FIXED"""
    print(f"\n🗺️ Loading enhanced biomass data from {len(config.raster_pairs)} sites...")
    
    all_features = []
    all_targets = []
    all_site_labels = []
    feature_names = None
    
    for site_id, (sat_path, bio_path) in enumerate(config.raster_pairs):
        site_name = config.site_names[site_id]
        print(f"\n--- Processing {site_name} (Site {site_id}) ---")
        
        try:
            # Check if files exist
            if not os.path.exists(sat_path) or not os.path.exists(bio_path):
                print(f"  Warning: Files not found for {site_name}, skipping...")
                continue
                
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
            
            if valid_percent < 1.0:  # Skip if too few valid pixels
                print(f"  Warning: Too few valid pixels for {site_name}, skipping...")
                continue
            
            # Extract enhanced features
            features, targets, site_labels, names = extract_enhanced_features_from_pixels(
                satellite_data, biomass_data, valid_mask, site_id, config
            )
            
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_targets.append(targets)
                all_site_labels.append(site_labels)
                
                if feature_names is None:
                    feature_names = names
                
                print(f"  {site_name}: {len(targets):,} samples, {features.shape[1]} features")
            else:
                print(f"  Warning: No features extracted for {site_name}")
        
        except Exception as e:
            print(f"  Error loading {site_name}: {e}")
            continue
        
        memory_cleanup()
    
    # Combine all data
    if all_features:
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        site_labels = np.hstack(all_site_labels)
        
        print(f"\n✅ Combined enhanced dataset:")
        print(f"   Shape: {X.shape}")
        print(f"   Total samples: {len(y):,}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Sites: {len(np.unique(site_labels))}")
        
        # Phase 1: LOSO-aware feature selection
        if config.enable_phase1 and len(feature_names) > config.max_features:
            print(f"\n🔍 Phase 1: LOSO-aware feature selection...")
            selector = LOSOFeatureSelector(config)
            X, feature_names = selector.select_features(X, y, site_labels, feature_names)
            print(f"   Reduced to: {X.shape[1]} features")
        
        return X, y, site_labels, feature_names
    else:
        raise ValueError("No valid data could be loaded from any site")

# # Step 9: Enhanced Model Architectures (FIXED)
class StandardMLP(nn.Module):
    """Standard Multi-Layer Perceptron - FIXED"""
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

class MLPWithResidualBlocks(nn.Module):
    """MLP with Residual Connections - FIXED"""
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

class ResBlock(nn.Module):
    """Residual Block for MLP - FIXED"""
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

class DenselyConnectedMLP(nn.Module):
    """MLP with Dense Connections - FIXED"""
    def __init__(self, n_features):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.dense1 = DenseBlock(256, growth_rate=32, num_layers=4)
        self.trans1 = Transition(256 + 4*32, 256)
        
        self.dense2 = DenseBlock(256, growth_rate=32, num_layers=4)
        self.trans2 = Transition(256 + 4*32, 128)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        return self.classifier(x).squeeze(-1)

class DenseBlock(nn.Module):
    """Dense Block for MLP - FIXED"""
    def __init__(self, in_dim, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(in_dim + i * growth_rate),
                nn.ReLU(),
                nn.Linear(in_dim + i * growth_rate, 4 * growth_rate),
                nn.ReLU(),
                nn.Linear(4 * growth_rate, growth_rate)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class Transition(nn.Module):
    """Transition Layer for Dense MLP - FIXED"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, x):
        return self.transition(x)

class EncoderDecoderMLP(nn.Module):
    """Encoder-Decoder MLP with Skip Connections - FIXED"""
    def __init__(self, n_features):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(n_features, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder with skip connections
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(1024, 256)
        self.dec2 = self._upconv_block(512, 128)
        self.dec1 = self._upconv_block(256, 64)
        
        self.final = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        self._init_weights()
    
    def _conv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def _upconv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1).squeeze(-1)

class TransformerEncoder(nn.Module):
    """Transformer Encoder for Biomass Regression - FIXED"""
    def __init__(self, n_features, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(n_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # FIXED: Ensure proper tensor dimensions
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        return self.output_proj(x).squeeze(-1)

# # Step 10: Phase 2 - Site-Aware Normalization (FIXED)
class SiteAwareScaler:
    """Site-aware normalization for LOSO cross-validation - FIXED"""
    
    def __init__(self):
        self.site_scalers = {}
        self.global_scaler = RobustScaler()
        
    def fit(self, X, site_labels):
        """Fit scalers for each site and globally"""
        # Fit global scaler
        self.global_scaler.fit(X)
        
        # Fit per-site scalers
        for site_id in np.unique(site_labels):
            site_mask = site_labels == site_id
            if np.sum(site_mask) > 1:  # Need at least 2 samples
                site_scaler = RobustScaler()
                site_scaler.fit(X[site_mask])
                self.site_scalers[site_id] = site_scaler
    
    def transform(self, X, site_labels, mode='site_aware'):
        """Transform data using site-aware or global scaling"""
        if mode == 'global':
            return self.global_scaler.transform(X)
        elif mode == 'site_aware':
            X_scaled = np.zeros_like(X)
            for site_id in np.unique(site_labels):
                site_mask = site_labels == site_id
                if site_id in self.site_scalers:
                    X_scaled[site_mask] = self.site_scalers[site_id].transform(X[site_mask])
                else:
                    # Fallback to global scaler for unseen sites
                    X_scaled[site_mask] = self.global_scaler.transform(X[site_mask])
            return X_scaled
        else:
            raise ValueError(f"Unknown scaling mode: {mode}")

# # Step 11: Phase 1 - Enhanced Ensemble Methods (FIXED)
class EnhancedEnsemble:
    """Enhanced ensemble methods for biomass prediction - FIXED"""
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.weights = None
        self.site_weights = {}
        
    def add_model(self, model, validation_score, site_scores=None):
        """Add model to ensemble with performance info"""
        self.models.append({
            'model': model,
            'score': validation_score,
            'site_scores': site_scores or {}
        })
    
    def fit_ensemble_weights(self, val_predictions, val_targets, val_site_labels=None):
        """Fit ensemble weights using validation data - FIXED"""
        n_models = len(val_predictions)
        
        # FIXED: Ensure all predictions are numpy arrays
        val_predictions = [np.array(pred) for pred in val_predictions]
        val_targets = np.array(val_targets)
        
        if val_site_labels is not None:
            val_site_labels = np.array(val_site_labels)
        
        if self.config.site_adaptive_ensemble and val_site_labels is not None:
            # Site-adaptive weighting
            for site_id in np.unique(val_site_labels):
                site_mask = val_site_labels == site_id
                # FIXED: Ensure proper array indexing
                site_preds = [pred[site_mask] for pred in val_predictions]
                site_targets = val_targets[site_mask]
                
                if len(site_targets) > 0:  # Check if site has any samples
                    # Simple performance-based weighting for each site
                    site_weights = []
                    for pred in site_preds:
                        try:
                            r2 = r2_score(site_targets, pred)
                            site_weights.append(max(r2, 0.01))  # Avoid negative weights
                        except:
                            site_weights.append(0.01)  # Fallback weight
                    
                    # Normalize weights
                    site_weights = np.array(site_weights)
                    if site_weights.sum() > 0:
                        site_weights = site_weights / site_weights.sum()
                    else:
                        site_weights = np.ones(len(site_weights)) / len(site_weights)
                    self.site_weights[site_id] = site_weights
        else:
            # Global performance-based weighting
            weights = []
            for pred in val_predictions:
                try:
                    r2 = r2_score(val_targets, pred)
                    weights.append(max(r2, 0.01))
                except:
                    weights.append(0.01)  # Fallback weight
            
            self.weights = np.array(weights)
            if self.weights.sum() > 0:
                self.weights = self.weights / self.weights.sum()
            else:
                self.weights = np.ones(len(weights)) / len(weights)
    
    def predict(self, predictions, site_labels=None):
        """Generate ensemble predictions - FIXED"""
        # FIXED: Ensure all predictions are numpy arrays
        predictions = [np.array(pred) for pred in predictions]
        
        if self.config.site_adaptive_ensemble and site_labels is not None:
            site_labels = np.array(site_labels)
            # Site-adaptive ensemble
            ensemble_pred = np.zeros(len(predictions[0]))
            for site_id in np.unique(site_labels):
                site_mask = site_labels == site_id
                if site_id in self.site_weights:
                    weights = self.site_weights[site_id]
                else:
                    weights = self.weights if self.weights is not None else np.ones(len(predictions)) / len(predictions)
                
                site_ensemble = np.zeros(np.sum(site_mask))
                for i, pred in enumerate(predictions):
                    site_ensemble += weights[i] * pred[site_mask]
                ensemble_pred[site_mask] = site_ensemble
            
            return ensemble_pred
        else:
            # Global ensemble
            weights = self.weights if self.weights is not None else np.ones(len(predictions)) / len(predictions)
            ensemble_pred = np.zeros(len(predictions[0]))
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
            return ensemble_pred

# # Step 12: Enhanced Training Framework with LOSO (FIXED)
class EnhancedModelTrainer:
    """Enhanced trainer with LOSO cross-validation and Phase 1 & 2 features - FIXED"""
    
    def __init__(self, config):
        self.config = config
        self.device = device
        self.site_scaler = SiteAwareScaler()
        
    def train_neural_model_loso(self, model_class, X, y, site_labels, feature_names, model_name):
        """Train neural model with LOSO cross-validation - FIXED"""
        print(f"\n🧠 Training {model_name} with LOSO CV...")
        
        unique_sites = np.unique(site_labels)
        if self.config.loso_folds < len(unique_sites):
            # Use only subset of sites for testing
            unique_sites = unique_sites[:self.config.loso_folds]
        
        fold_results = []
        fold_predictions = []
        fold_targets = []
        fold_site_labels = []
        
        for fold, val_site in enumerate(unique_sites):
            print(f"  Fold {fold+1}/{len(unique_sites)}: Validating on {self.config.site_names[val_site]}")
            
            # Create train/val split
            train_mask = site_labels != val_site
            val_mask = site_labels == val_site
            
            if np.sum(val_mask) == 0:  # Skip if no validation samples
                print(f"    No validation samples for site {val_site}, skipping...")
                continue
            
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            site_train, site_val = site_labels[train_mask], site_labels[val_mask]
            
            # Site-aware scaling
            fold_scaler = SiteAwareScaler()
            fold_scaler.fit(X_train, site_train)
            X_train_scaled = fold_scaler.transform(X_train, site_train, mode='site_aware')
            X_val_scaled = fold_scaler.transform(X_val, site_val, mode='global')  # Use global for unseen site
            
            # Convert to tensors
            train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Train model for this fold
            try:
                fold_result = self._train_single_fold(model_class, train_loader, val_loader, 
                                                    X_train_scaled.shape[1], f"{model_name}_fold_{fold}")
                
                fold_results.append(fold_result)
                fold_predictions.extend(fold_result['val_predictions'])
                fold_targets.extend(fold_result['val_targets'])
                fold_site_labels.extend(site_val)
            except Exception as e:
                print(f"    Fold {fold+1} failed: {e}")
                continue
        
        if not fold_results:
            raise RuntimeError(f"All folds failed for {model_name}")
        
        # Aggregate results
        overall_r2 = r2_score(fold_targets, fold_predictions)
        overall_rmse = np.sqrt(mean_squared_error(fold_targets, fold_predictions))
        overall_mae = mean_absolute_error(fold_targets, fold_predictions)
        
        # Per-site analysis
        site_results = {}
        for site_id in unique_sites:
            site_mask = np.array(fold_site_labels) == site_id
            if np.sum(site_mask) > 0:
                site_preds = np.array(fold_predictions)[site_mask]
                site_targs = np.array(fold_targets)[site_mask]
                site_r2 = r2_score(site_targs, site_preds)
                site_rmse = np.sqrt(mean_squared_error(site_targs, site_preds))
                site_results[self.config.site_names[site_id]] = {'r2': site_r2, 'rmse': site_rmse}
        
        avg_training_time = np.mean([r['training_time'] for r in fold_results])
        avg_model_size = np.mean([r['model_size'] for r in fold_results])
        
        return {
            'loso_r2': overall_r2,
            'loso_rmse': overall_rmse,
            'loso_mae': overall_mae,
            'loso_r2_std': np.std([r['val_r2'] for r in fold_results]),
            'site_results': site_results,
            'fold_results': fold_results,
            'predictions': fold_predictions,
            'targets': fold_targets,
            'site_labels': fold_site_labels,
            'training_time': avg_training_time,
            'model_size': avg_model_size,
            'n_folds': len(fold_results)
        }
    
    def _train_single_fold(self, model_class, train_loader, val_loader, n_features, fold_name):
        """Train a single fold - FIXED"""
        # Phase 1: Try multiple loss functions
        best_result = None
        best_loss_name = None
        
        for loss_name in self.config.loss_functions:
            try:
                model = model_class(n_features).to(self.device)
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
                
                for epoch in range(self.config.num_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        # Check for NaN loss
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
                        for batch_x, batch_y in val_loader:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                val_loss += loss.item()
                                val_preds.extend(outputs.cpu().numpy())
                                val_targets.extend(batch_y.cpu().numpy())
                    
                    if len(val_preds) == 0:  # No valid predictions
                        raise ValueError("No valid predictions generated")
                    
                    val_loss /= len(val_loader)
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
                val_r2 = r2_score(val_targets, val_preds)
                val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
                model_size = sum(p.numel() for p in model.parameters())
                
                current_result = {
                    'model': model,
                    'val_r2': val_r2,
                    'val_rmse': val_rmse,
                    'val_loss': best_val_loss,
                    'val_predictions': val_preds,
                    'val_targets': val_targets,
                    'training_time': training_time,
                    'model_size': model_size,
                    'epochs_trained': epoch + 1,
                    'loss_function': loss_name
                }
                
                # Keep best result across loss functions
                if best_result is None or val_r2 > best_result['val_r2']:
                    best_result = current_result
                    best_loss_name = loss_name
                
                print(f"    {loss_name}: R² = {val_r2:.4f}, RMSE = {val_rmse:.3f}")
                
            except Exception as e:
                print(f"    {loss_name}: Failed - {e}")
                continue
        
        if best_result is None:
            raise RuntimeError(f"All loss functions failed for {fold_name}")
        
        print(f"    Best: {best_loss_name} (R² = {best_result['val_r2']:.4f})")
        return best_result
    
    def train_tree_model_loso(self, model_class, X, y, site_labels, model_name):
        """Train tree model with LOSO cross-validation - FIXED"""
        print(f"\n🌳 Training {model_name} with LOSO CV...")
        
        unique_sites = np.unique(site_labels)
        if self.config.loso_folds < len(unique_sites):
            unique_sites = unique_sites[:self.config.loso_folds]
        
        fold_results = []
        fold_predictions = []
        fold_targets = []
        fold_site_labels = []
        
        for fold, val_site in enumerate(unique_sites):
            print(f"  Fold {fold+1}/{len(unique_sites)}: Validating on {self.config.site_names[val_site]}")
            
            # Create train/val split
            train_mask = site_labels != val_site
            val_mask = site_labels == val_site
            
            if np.sum(val_mask) == 0:  # Skip if no validation samples
                print(f"    No validation samples for site {val_site}, skipping...")
                continue
            
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            site_train, site_val = site_labels[train_mask], site_labels[val_mask]
            
            # Site-aware scaling
            fold_scaler = SiteAwareScaler()
            fold_scaler.fit(X_train, site_train)
            X_train_scaled = fold_scaler.transform(X_train, site_train, mode='site_aware')
            X_val_scaled = fold_scaler.transform(X_val, site_val, mode='global')
            
            # Train model
            start_time = time.time()
            try:
                model = model_class()
                
                if 'XGB' in model_name:
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False
                    )
                elif 'LightGBM' in model_name:
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate
                val_pred = model.predict(X_val_scaled)
                val_r2 = r2_score(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                training_time = time.time() - start_time
                
                try:
                    model_size = model.get_booster().trees_to_dataframe().shape[0] if hasattr(model, 'get_booster') else 100000
                except:
                    model_size = 100000
                
                fold_result = {
                    'model': model,
                    'val_r2': val_r2,
                    'val_rmse': val_rmse,
                    'training_time': training_time,
                    'model_size': model_size
                }
                
                fold_results.append(fold_result)
                fold_predictions.extend(val_pred)
                fold_targets.extend(y_val)
                fold_site_labels.extend(site_val)
                
                print(f"    R² = {val_r2:.4f}, RMSE = {val_rmse:.3f}")
                
            except Exception as e:
                print(f"    Fold {fold+1} failed: {e}")
                continue
        
        if not fold_results:
            raise RuntimeError(f"All folds failed for {model_name}")
        
        # Aggregate results
        overall_r2 = r2_score(fold_targets, fold_predictions)
        overall_rmse = np.sqrt(mean_squared_error(fold_targets, fold_predictions))
        overall_mae = mean_absolute_error(fold_targets, fold_predictions)
        
        # Per-site analysis
        site_results = {}
        for site_id in unique_sites:
            site_mask = np.array(fold_site_labels) == site_id
            if np.sum(site_mask) > 0:
                site_preds = np.array(fold_predictions)[site_mask]
                site_targs = np.array(fold_targets)[site_mask]
                site_r2 = r2_score(site_targs, site_preds)
                site_rmse = np.sqrt(mean_squared_error(site_targs, site_preds))
                site_results[self.config.site_names[site_id]] = {'r2': site_r2, 'rmse': site_rmse}
        
        avg_training_time = np.mean([r['training_time'] for r in fold_results])
        avg_model_size = np.mean([r['model_size'] for r in fold_results])
        
        return {
            'loso_r2': overall_r2,
            'loso_rmse': overall_rmse,
            'loso_mae': overall_mae,
            'loso_r2_std': np.std([r['val_r2'] for r in fold_results]),
            'site_results': site_results,
            'fold_results': fold_results,
            'predictions': fold_predictions,
            'targets': fold_targets,
            'site_labels': fold_site_labels,
            'training_time': avg_training_time,
            'model_size': avg_model_size,
            'n_folds': len(fold_results)
        }

# # Step 13: Enhanced Model Registry (FIXED)
class ModelRegistry:
    """Model registry to avoid lambda function issues - FIXED"""
    
    @staticmethod
    def get_neural_models():
        """Get neural network models"""
        return {
            'StandardMLP': StandardMLP,
            'MLPWithResidualBlocks': MLPWithResidualBlocks,
            'DenselyConnectedMLP': DenselyConnectedMLP,
            'EncoderDecoderMLP': EncoderDecoderMLP,
            'TransformerEncoder': TransformerEncoder
        }
    
    @staticmethod
    def get_tree_models(config):
        """Get tree models with configurations"""
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
    
    @staticmethod
    def is_neural_model(model_name):
        """Check if model is a neural network"""
        neural_models = ModelRegistry.get_neural_models()
        return model_name in neural_models

# # Step 14: Load Enhanced Data (WITH ERROR HANDLING)
print(f"\n📊 Loading enhanced biomass data...")
start_time = time.time()

try:
    # Load enhanced data with site information
    X, y, site_labels, feature_names = load_enhanced_biomass_data(config)
    
    data_load_time = time.time() - start_time
    print(f"⏱️ Enhanced data loading completed in {data_load_time:.2f} seconds")
    
    # Print data summary
    print(f"\n📈 Enhanced Dataset Summary:")
    print(f"   Total samples: {len(y):,}")
    print(f"   Total features: {len(feature_names)}")
    print(f"   Sites: {len(np.unique(site_labels))}")
    for site_id in np.unique(site_labels):
        site_count = np.sum(site_labels == site_id)
        site_name = config.site_names[site_id]
        print(f"   {site_name}: {site_count:,} samples")
    
    # Sample feature names by category
    print(f"\n🏷️ Feature Categories (sample):")
    feature_categories = {
        'Original Bands': [f for f in feature_names if f.startswith('Band_')],
        'Spectral Indices': [f for f in feature_names if any(x in f for x in ['NDVI', 'EVI', 'SAVI'])],
        'Spatial Features': [f for f in feature_names if 'Spatial_' in f],
        'Temporal Features': [f for f in feature_names if any(x in f for x in ['T1T2', 'T2T3', 'temporal'])],
        'Texture Features': [f for f in feature_names if any(x in f for x in ['GLCM', 'LBP', 'Edge'])],
        'PCA Features': [f for f in feature_names if f.startswith('PCA_')]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"   {category}: {len(features)} features")
            print(f"     Examples: {features[:3]}")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    raise

# # Step 15: Define Enhanced Models with Registry (FIXED)
print(f"\n🤖 Setting up Enhanced Model Registry...")

# Get models from registry
neural_models = ModelRegistry.get_neural_models()
tree_models = ModelRegistry.get_tree_models(config)

print(f"Neural Network Models: {list(neural_models.keys())}")
print(f"Tree Models: {list(tree_models.keys())}")

# # Step 16: Run Enhanced LOSO Benchmark (FIXED)
print(f"\n🚀 Starting Enhanced LOSO Biomass Prediction Benchmark...")
print("=" * 70)
print(f"🔧 Configuration:")
print(f"   Mode: {config.mode}")
print(f"   Phase 1 Enabled: {config.enable_phase1}")
print(f"   Phase 2 Enabled: {config.enable_phase2}")
print(f"   LOSO Folds: {config.loso_folds}")
print(f"   Loss Functions: {config.loss_functions}")
print(f"   Spatial Scales: {config.spatial_scales}")
print(f"   Site-Aware Scaling: {config.site_aware_scaling}")

trainer = EnhancedModelTrainer(config)
results = {}

# Train Neural Network Models
for model_name, model_class in neural_models.items():
    print(f"\n{'='*50}")
    print(f"🎯 Benchmarking {model_name} (Neural Network)")
    print(f"{'='*50}")
    
    try:
        model_results = trainer.train_neural_model_loso(
            model_class, X, y, site_labels, feature_names, model_name
        )
        
        results[model_name] = model_results
        
        # Print results summary
        print(f"\n✅ {model_name} Results:")
        print(f"   LOSO R²: {model_results['loso_r2']:.4f} ± {model_results['loso_r2_std']:.4f}")
        print(f"   LOSO RMSE: {model_results['loso_rmse']:.3f} Mg/ha")
        print(f"   LOSO MAE: {model_results['loso_mae']:.3f} Mg/ha")
        print(f"   Training Time: {model_results['training_time']:.2f}s")
        print(f"   Model Size: {model_results['model_size']:,} parameters")
        print(f"   Folds Completed: {model_results['n_folds']}")
        
        # Per-site performance
        print(f"\n   📍 Per-Site Performance:")
        for site_name, site_result in model_results['site_results'].items():
            print(f"     {site_name}: R² = {site_result['r2']:.4f}, RMSE = {site_result['rmse']:.3f}")
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # Memory cleanup
    memory_cleanup()

# Train Tree Models
for model_name, model_factory in tree_models.items():
    print(f"\n{'='*50}")
    print(f"🎯 Benchmarking {model_name} (Tree Model)")
    print(f"{'='*50}")
    
    try:
        model_results = trainer.train_tree_model_loso(
            model_factory, X, y, site_labels, model_name
        )
        
        results[model_name] = model_results
        
        # Print results summary
        print(f"\n✅ {model_name} Results:")
        print(f"   LOSO R²: {model_results['loso_r2']:.4f} ± {model_results['loso_r2_std']:.4f}")
        print(f"   LOSO RMSE: {model_results['loso_rmse']:.3f} Mg/ha")
        print(f"   LOSO MAE: {model_results['loso_mae']:.3f} Mg/ha")
        print(f"   Training Time: {model_results['training_time']:.2f}s")
        print(f"   Model Size: {model_results['model_size']:,} parameters")
        print(f"   Folds Completed: {model_results['n_folds']}")
        
        # Per-site performance
        print(f"\n   📍 Per-Site Performance:")
        for site_name, site_result in model_results['site_results'].items():
            print(f"     {site_name}: R² = {site_result['r2']:.4f}, RMSE = {site_result['rmse']:.3f}")
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # Memory cleanup
    memory_cleanup()

# # Step 17: Phase 1 - Enhanced Ensemble Creation (FIXED)
print(f"\n🎭 Phase 1: Creating Enhanced Ensembles...")

if len(results) >= 2:
    # Get top performing models for ensemble
    model_scores = [(name, res['loso_r2']) for name, res in results.items()]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    top_models = model_scores[:min(config.ensemble_size, len(model_scores))]
    
    print(f"Top {len(top_models)} models for ensemble:")
    for i, (name, score) in enumerate(top_models, 1):
        print(f"  {i}. {name}: R² = {score:.4f}")
    
    # Create ensemble predictions
    try:
        ensemble = EnhancedEnsemble(config)
        
        # Collect predictions and targets from all models
        all_predictions = []
        common_targets = None
        common_site_labels = None
        
        for model_name, _ in top_models:
            if model_name in results:
                model_result = results[model_name]
                all_predictions.append(model_result['predictions'])
                
                if common_targets is None:
                    common_targets = np.array(model_result['targets'])
                    common_site_labels = np.array(model_result['site_labels'])
        
        if len(all_predictions) >= 2:
            # Fit ensemble weights
            ensemble.fit_ensemble_weights(all_predictions, common_targets, common_site_labels)
            
            # Generate ensemble predictions
            ensemble_pred = ensemble.predict(all_predictions, common_site_labels)
            
            # Evaluate ensemble
            ensemble_r2 = r2_score(common_targets, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(common_targets, ensemble_pred))
            ensemble_mae = mean_absolute_error(common_targets, ensemble_pred)
            
            # Per-site ensemble performance
            ensemble_site_results = {}
            for site_id in np.unique(common_site_labels):
                site_mask = common_site_labels == site_id
                if np.sum(site_mask) > 0:
                    site_preds = ensemble_pred[site_mask]
                    site_targs = common_targets[site_mask]
                    site_r2 = r2_score(site_targs, site_preds)
                    site_rmse = np.sqrt(mean_squared_error(site_targs, site_preds))
                    ensemble_site_results[config.site_names[site_id]] = {'r2': site_r2, 'rmse': site_rmse}
            
            # Add ensemble results
            results['EnhancedEnsemble'] = {
                'loso_r2': ensemble_r2,
                'loso_rmse': ensemble_rmse,
                'loso_mae': ensemble_mae,
                'loso_r2_std': 0.0,  # Single ensemble, no std
                'site_results': ensemble_site_results,
                'predictions': ensemble_pred.tolist(),
                'targets': common_targets.tolist(),
                'site_labels': common_site_labels.tolist(),
                'training_time': np.mean([results[name]['training_time'] for name, _ in top_models]),
                'model_size': np.sum([results[name]['model_size'] for name, _ in top_models]),
                'n_folds': config.loso_folds,
                'ensemble_models': [name for name, _ in top_models]
            }
            
            print(f"\n🎭 Enhanced Ensemble Results:")
            print(f"   LOSO R²: {ensemble_r2:.4f}")
            print(f"   LOSO RMSE: {ensemble_rmse:.3f} Mg/ha")
            print(f"   LOSO MAE: {ensemble_mae:.3f} Mg/ha")
            print(f"   Improvement over best single model: {ensemble_r2 - top_models[0][1]:.4f}")
            
            print(f"\n   📍 Ensemble Per-Site Performance:")
            for site_name, site_result in ensemble_site_results.items():
                print(f"     {site_name}: R² = {site_result['r2']:.4f}, RMSE = {site_result['rmse']:.3f}")
    
    except Exception as e:
        print(f"❌ Ensemble creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠️ Not enough successful models for ensemble creation (need ≥2, got {len(results)})")

# # Step 18: Convert Back to Original Scale (FIXED)
print(f"\n🔄 Converting results back to original scale...")

if config.use_log_transform:
    print(f"   Converting from log scale (epsilon = {config.epsilon})...")
    for model_name in results:
        if 'predictions' in results[model_name]:
            try:
                log_preds = np.array(results[model_name]['predictions'])
                log_targets = np.array(results[model_name]['targets'])
                
                orig_preds = np.exp(log_preds) - config.epsilon
                orig_targets = np.exp(log_targets) - config.epsilon
                
                # Ensure non-negative
                orig_preds = np.maximum(orig_preds, 0)
                orig_targets = np.maximum(orig_targets, 0)
                
                # Recalculate metrics on original scale
                orig_r2 = r2_score(orig_targets, orig_preds)
                orig_rmse = np.sqrt(mean_squared_error(orig_targets, orig_preds))
                orig_mae = mean_absolute_error(orig_targets, orig_preds)
                
                # Per-site performance on original scale
                orig_site_results = {}
                site_labels_array = np.array(results[model_name]['site_labels'])
                for site_id in np.unique(site_labels_array):
                    site_mask = site_labels_array == site_id
                    if np.sum(site_mask) > 0:
                        site_preds = orig_preds[site_mask]
                        site_targs = orig_targets[site_mask]
                        site_r2 = r2_score(site_targs, site_preds)
                        site_rmse = np.sqrt(mean_squared_error(site_targs, site_preds))
                        orig_site_results[config.site_names[site_id]] = {'r2': site_r2, 'rmse': site_rmse}
                
                # Store both scales
                results[model_name]['loso_r2_orig'] = orig_r2
                results[model_name]['loso_rmse_orig'] = orig_rmse
                results[model_name]['loso_mae_orig'] = orig_mae
                results[model_name]['site_results_orig'] = orig_site_results
                results[model_name]['predictions_orig'] = orig_preds.tolist()
                results[model_name]['targets_orig'] = orig_targets.tolist()
                
                print(f"   {model_name}: Log R² = {results[model_name]['loso_r2']:.4f}, "
                      f"Original R² = {orig_r2:.4f}")
                      
            except Exception as e:
                print(f"   Warning: Scale conversion failed for {model_name}: {e}")

# # Step 19: Create Comprehensive LOSO Report (FIXED)
def create_enhanced_loso_report(results, config):
    """Create comprehensive LOSO benchmark report - FIXED"""
    print(f"\n📊 Creating Enhanced LOSO Benchmark Report...")
    
    if not results:
        print("⚠️ No results to report")
        return None, None
    
    # Prepare data for analysis
    report_data = []
    for model_name, metrics in results.items():
        try:
            # Use original scale metrics if available
            if 'loso_r2_orig' in metrics:
                r2_score = metrics['loso_r2_orig']
                rmse_score = metrics['loso_rmse_orig']
                mae_score = metrics['loso_mae_orig']
                scale_note = "Original"
            else:
                r2_score = metrics['loso_r2']
                rmse_score = metrics['loso_rmse']
                mae_score = metrics['loso_mae']
                scale_note = "Log" if config.use_log_transform else "Original"
            
            # Model type classification
            if model_name == 'EnhancedEnsemble':
                model_type = 'Ensemble'
            elif model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
                model_type = 'Tree'
            else:
                model_type = 'Neural'
            
            report_data.append({
                'Model': model_name,
                'Type': model_type,
                'LOSO_R2': r2_score,
                'LOSO_RMSE': rmse_score,
                'LOSO_MAE': mae_score,
                'LOSO_R2_Std': metrics.get('loso_r2_std', 0.0),
                'Training_Time': metrics['training_time'],
                'Model_Size': metrics['model_size'],
                'N_Folds': metrics['n_folds'],
                'Scale': scale_note
            })
        except Exception as e:
            print(f"   Warning: Error processing {model_name}: {e}")
            continue
    
    if not report_data:
        print("⚠️ No valid data for report")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    df = df.sort_values('LOSO_R2', ascending=False)
    
    # Save comprehensive report
    report_path = os.path.join(config.results_dir, 'enhanced_loso_benchmark_report.csv')
    df.to_csv(report_path, index=False)
    
    # Save per-site results
    site_report_data = []
    for model_name, metrics in results.items():
        try:
            site_results_key = 'site_results_orig' if 'site_results_orig' in metrics else 'site_results'
            site_results = metrics[site_results_key]
            
            for site_name, site_metrics in site_results.items():
                site_report_data.append({
                    'Model': model_name,
                    'Site': site_name,
                    'R2': site_metrics['r2'],
                    'RMSE': site_metrics['rmse']
                })
        except Exception as e:
            print(f"   Warning: Error processing site results for {model_name}: {e}")
            continue
    
    site_df = None
    if site_report_data:
        site_df = pd.DataFrame(site_report_data)
        site_report_path = os.path.join(config.results_dir, 'loso_per_site_results.csv')
        site_df.to_csv(site_report_path, index=False)
    
    # Print summary
    print(f"\n🏆 ENHANCED LOSO BIOMASS PREDICTION BENCHMARK RESULTS")
    print("=" * 70)
    scale_info = " (Original Scale)" if config.use_log_transform else ""
    print(f"{'Rank':<5} {'Model':<22} {'Type':<8} {'LOSO R²':<9} {'±Std':<7} {'RMSE':<9} {'Time(s)':<8}")
    print("-" * 70)
    
    for i, row in df.iterrows():
        print(f"{i+1:<5} {row['Model']:<22} {row['Type']:<8} {row['LOSO_R2']:<9.4f} "
              f"{row['LOSO_R2_Std']:<7.4f} {row['LOSO_RMSE']:<9.2f} {row['Training_Time']:<8.2f}")
    
    return df, site_df

# Create the enhanced report
try:
    df_results, site_df = create_enhanced_loso_report(results, config)
    
    if df_results is not None:
        print(f"✅ Report created successfully with {len(df_results)} models")
    else:
        print("⚠️ Report creation failed")
        
except Exception as e:
    print(f"❌ Error creating report: {e}")
    import traceback
    traceback.print_exc()

# # Step 20: Enhanced Visualizations (SIMPLIFIED AND FIXED)
def create_enhanced_loso_visualizations(results, df_results, site_df, config):
    """Create comprehensive LOSO visualizations - FIXED"""
    if df_results is None or len(df_results) == 0:
        print("⚠️ No data available for visualization")
        return
        
    print(f"\n📈 Creating Enhanced LOSO Visualizations...")
    
    try:
        # Figure 1: LOSO Performance Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # LOSO R² comparison
        ax1 = axes[0, 0]
        models = df_results['Model'].values
        r2_scores = df_results['LOSO_R2'].values
        r2_stds = df_results['LOSO_R2_Std'].values
        
        colors = ['gold' if model == 'EnhancedEnsemble' else 
                  'lightcoral' if model in ['XGBoost', 'LightGBM', 'RandomForest'] else 
                  'lightblue' for model in models]
        
        bars = ax1.barh(models, r2_scores, xerr=r2_stds, color=colors, alpha=0.8, capsize=3)
        ax1.set_xlabel('LOSO R² Score ± Std')
        ax1.set_title('LOSO Cross-Validation Performance (R²)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for bar, r2, std in zip(bars, r2_scores, r2_stds):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{r2:.3f}±{std:.3f}', ha='left', va='center', fontsize=8)
        
        # LOSO RMSE comparison
        ax2 = axes[0, 1]
        rmse_scores = df_results['LOSO_RMSE'].values
        bars = ax2.barh(models, rmse_scores, color=colors, alpha=0.8)
        ax2.set_xlabel('LOSO RMSE (Mg/ha)')
        ax2.set_title('LOSO Cross-Validation Performance (RMSE)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Training Efficiency vs Performance
        ax3 = axes[1, 0]
        training_times = df_results['Training_Time'].values
        ax3.scatter(training_times, r2_scores, s=120, alpha=0.7, c=colors)
        for i, txt in enumerate(models):
            ax3.annotate(txt, (training_times[i], r2_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Average Training Time (seconds)')
        ax3.set_ylabel('LOSO R² Score')
        ax3.set_title('Training Efficiency vs LOSO Performance')
        ax3.grid(True, alpha=0.3)
        
        # Model Size vs Performance
        ax4 = axes[1, 1]
        model_sizes = df_results['Model_Size'].values / 1e6  # Convert to millions
        ax4.scatter(model_sizes, r2_scores, s=120, alpha=0.7, c=colors)
        for i, txt in enumerate(models):
            ax4.annotate(txt, (model_sizes[i], r2_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Model Size (Million Parameters)')
        ax4.set_ylabel('LOSO R² Score')
        ax4.set_title('Model Complexity vs LOSO Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.results_dir, 'enhanced_loso_benchmark_overview.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Prediction scatter plots for top models
        if len(df_results) >= 2:
            top_models = df_results.head(min(4, len(df_results)))['Model'].values
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, model_name in enumerate(top_models):
                if i >= 4:
                    break
                    
                ax = axes[i]
                result = results[model_name]
                
                # Use original scale if available
                if 'targets_orig' in result:
                    targets = np.array(result['targets_orig'])
                    predictions = np.array(result['predictions_orig'])
                    r2 = result['loso_r2_orig']
                    rmse = result['loso_rmse_orig']
                    title_suffix = "(Original Scale)"
                else:
                    targets = np.array(result['targets'])
                    predictions = np.array(result['predictions'])
                    r2 = result['loso_r2']
                    rmse = result['loso_rmse']
                    title_suffix = "(Log Scale)" if config.use_log_transform else ""
                
                # Create scatter plot
                ax.scatter(targets, predictions, alpha=0.6, s=20, c='blue')
                
                # Add 1:1 line
                min_val = min(np.min(targets), np.min(predictions))
                max_val = max(np.max(targets), np.max(predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Actual Biomass (Mg/ha)')
                ax.set_ylabel('Predicted Biomass (Mg/ha)')
                ax.set_title(f'{model_name} LOSO {title_suffix}\nR² = {r2:.3f}, RMSE = {rmse:.1f}')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for j in range(len(top_models), 4):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.results_dir, 'enhanced_loso_predictions_comparison.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"✅ Visualizations saved successfully")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

# Create enhanced visualizations
if df_results is not None:
    create_enhanced_loso_visualizations(results, df_results, site_df, config)

# # Step 21: Final Enhanced Analysis and Recommendations (FIXED)
def analyze_enhanced_results(df_results, site_df, config):
    """Provide detailed analysis and recommendations for enhanced LOSO results - FIXED"""
    if df_results is None or len(df_results) == 0:
        print("⚠️ No results available for analysis")
        return
        
    print(f"\n🔍 ENHANCED LOSO ANALYSIS AND RECOMMENDATIONS")
    print("=" * 60)
    
    try:
        # Overall performance analysis
        best_model = df_results.iloc[0]
        print(f"🥇 Best LOSO Model: {best_model['Model']}")
        print(f"   • LOSO R²: {best_model['LOSO_R2']:.4f} ± {best_model['LOSO_R2_Std']:.4f}")
        print(f"   • LOSO RMSE: {best_model['LOSO_RMSE']:.2f} Mg/ha")
        print(f"   • Training Time: {best_model['Training_Time']:.2f}s")
        print(f"   • Model Type: {best_model['Type']}")
        
        # Cross-site generalization analysis
        if site_df is not None and len(site_df) > 0:
            print(f"\n🌍 Cross-Site Generalization Analysis:")
            site_performance = site_df.groupby('Model').agg({
                'R2': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            if len(site_performance) > 0:
                best_generalization = site_performance.loc[:, ('R2', 'std')].idxmin()
                most_consistent = site_performance.loc[best_generalization, ('R2', 'mean')]
                consistency_std = site_performance.loc[best_generalization, ('R2', 'std')]
                
                print(f"   Most Consistent Across Sites: {best_generalization}")
                print(f"   • Mean R²: {most_consistent:.4f}")
                print(f"   • Std across sites: {consistency_std:.4f}")
        
        # Performance by type analysis
        print(f"\n🚀 Performance by Model Type:")
        performance_by_type = df_results.groupby('Type')['LOSO_R2'].agg(['mean', 'max', 'count'])
        for model_type in performance_by_type.index:
            stats = performance_by_type.loc[model_type]
            print(f"   {model_type} Models:")
            print(f"     • Count: {stats['count']}")
            print(f"     • Mean R²: {stats['mean']:.4f}")
            print(f"     • Best R²: {stats['max']:.4f}")
        
        # Ensemble improvement
        if 'EnhancedEnsemble' in df_results['Model'].values:
            ensemble_row = df_results[df_results['Model'] == 'EnhancedEnsemble'].iloc[0]
            best_single = df_results[df_results['Model'] != 'EnhancedEnsemble'].iloc[0]
            improvement = ensemble_row['LOSO_R2'] - best_single['LOSO_R2']
            print(f"\n🎭 Ensemble Improvement:")
            print(f"   • Ensemble R²: {ensemble_row['LOSO_R2']:.4f}")
            print(f"   • Best Single Model R²: {best_single['LOSO_R2']:.4f}")
            print(f"   • Improvement: +{improvement:.4f} ({improvement/best_single['LOSO_R2']*100:.1f}%)")
        
        # Model efficiency analysis
        print(f"\n⚡ Model Efficiency Analysis:")
        df_results['Efficiency'] = df_results['LOSO_R2'] / df_results['Training_Time']
        most_efficient = df_results.nlargest(1, 'Efficiency').iloc[0]
        print(f"   Most Efficient: {most_efficient['Model']}")
        print(f"   • R² per second: {most_efficient['Efficiency']:.6f}")
        print(f"   • R²: {most_efficient['LOSO_R2']:.4f}")
        print(f"   • Training time: {most_efficient['Training_Time']:.2f}s")
        
        # Recommendations
        print(f"\n💡 ENHANCED RECOMMENDATIONS:")
        
        print(f"\n1. 🎯 For Production Deployment:")
        print(f"   → Primary: {best_model['Model']} (best LOSO performance)")
        print(f"   → Expected cross-site R²: {best_model['LOSO_R2']:.3f} ± {best_model['LOSO_R2_Std']:.3f}")
        
        print(f"\n2. ⚡ For Real-time Applications:")
        print(f"   → Use: {most_efficient['Model']} (most efficient)")
        print(f"   → Trade-off: R² = {most_efficient['LOSO_R2']:.3f} vs {most_efficient['Training_Time']:.2f}s training")
        
        print(f"\n3. 🔧 For Further Improvement:")
        print(f"   → Consider hyperparameter tuning for top models")
        print(f"   → Investigate site-specific adaptations")
        print(f"   → Explore additional ensemble techniques")
        
        if config.mode == 'test':
            print(f"\n⚠️  Note: Results from TEST mode with limited features.")
            print(f"   → Run in FULL mode for production-ready results")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

# Run enhanced final analysis
if df_results is not None:
    analyze_enhanced_results(df_results, site_df, config)

# # Step 22: Save Enhanced Results (FIXED)
print(f"\n💾 Saving enhanced benchmark results...")

try:
    # Save detailed results
    enhanced_results_summary = {
        'benchmark_config': {
            'mode': config.mode,
            'timestamp': config.timestamp,
            'phase1_enabled': config.enable_phase1,
            'phase2_enabled': config.enable_phase2,
            'loso_enabled': config.enable_loso,
            'loso_folds': config.loso_folds,
            'loss_functions': config.loss_functions,
            'spatial_scales': config.spatial_scales,
            'temporal_features': config.temporal_features,
            'site_aware_scaling': config.site_aware_scaling,
            'max_features': config.max_features,
            'num_epochs': config.num_epochs,
            'site_names': config.site_names
        },
        'data_info': {
            'total_samples': len(y),
            'features_used': len(feature_names),
            'sites_used': len(np.unique(site_labels)),
            'tiles_processed': len(config.raster_pairs),
            'feature_categories': {
                'original_bands': len([f for f in feature_names if f.startswith('Band_')]),
                'spectral_indices': len([f for f in feature_names if any(x in f for x in ['NDVI', 'EVI', 'SAVI'])]),
                'spatial_features': len([f for f in feature_names if 'Spatial_' in f]),
                'temporal_features': len([f for f in feature_names if any(x in f for x in ['T1T2', 'T2T3', 'temporal'])]),
                'texture_features': len([f for f in feature_names if any(x in f for x in ['GLCM', 'LBP', 'Edge'])]),
                'pca_features': len([f for f in feature_names if f.startswith('PCA_')])
            }
        },
        'loso_results': {}
    }
    
    # Add results for each model
    for model_name in results.keys():
        try:
            enhanced_results_summary['loso_results'][model_name] = {
                'loso_r2': results[model_name].get('loso_r2_orig', results[model_name]['loso_r2']),
                'loso_rmse': results[model_name].get('loso_rmse_orig', results[model_name]['loso_rmse']),
                'loso_mae': results[model_name].get('loso_mae_orig', results[model_name]['loso_mae']),
                'loso_r2_std': results[model_name]['loso_r2_std'],
                'training_time': results[model_name]['training_time'],
                'model_size': results[model_name]['model_size'],
                'n_folds': results[model_name]['n_folds'],
                'site_results': results[model_name].get('site_results_orig', results[model_name]['site_results'])
            }
        except Exception as e:
            print(f"   Warning: Error saving results for {model_name}: {e}")
    
    # Add performance summary
    if df_results is not None and len(df_results) > 0:
        enhanced_results_summary['performance_summary'] = {
            'best_model': df_results.iloc[0]['Model'],
            'best_loso_r2': df_results.iloc[0]['LOSO_R2'],
            'best_loso_rmse': df_results.iloc[0]['LOSO_RMSE'],
            'total_models_tested': len(df_results),
            'successful_models': len(results)
        }
        
        if 'EnhancedEnsemble' in df_results['Model'].values:
            ensemble_improvement = (
                df_results[df_results['Model'] == 'EnhancedEnsemble']['LOSO_R2'].iloc[0] - 
                df_results[df_results['Model'] != 'EnhancedEnsemble']['LOSO_R2'].iloc[0]
            )
            enhanced_results_summary['performance_summary']['ensemble_improvement'] = ensemble_improvement
    
    # Save to JSON
    with open(os.path.join(config.results_dir, 'enhanced_loso_results.json'), 'w') as f:
        json.dump(enhanced_results_summary, f, indent=4, default=str)
    
    # Save feature names and importance
    feature_info = {
        'feature_names': feature_names,
        'total_features': len(feature_names),
        'feature_selection_applied': len(feature_names) < X.shape[1] if hasattr(config, 'max_features') else False
    }
    
    with open(os.path.join(config.results_dir, 'enhanced_feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=4)
    
    print(f"✅ Results saved successfully")
    
except Exception as e:
    print(f"❌ Error saving results: {e}")
    import traceback
    traceback.print_exc()

# Calculate total time
total_time = time.time() - start_time
print(f"✅ Enhanced benchmark completed in {total_time/60:.2f} minutes")

# # Step 23: Final Enhanced Summary (FIXED)
print(f"\n🎉 ENHANCED LOSO BIOMASS BENCHMARK COMPLETED! 🎉")
print("=" * 70)
print(f"📊 Enhanced Summary:")
print(f"   • Phase 1 & 2 Features: ✅ Enabled")
print(f"   • LOSO Cross-Validation: ✅ {config.loso_folds}-fold")
print(f"   • Models tested: {len(results)}")

if df_results is not None and len(df_results) > 0:
    print(f"   • Best performance: {df_results.iloc[0]['Model']} (R² = {df_results.iloc[0]['LOSO_R2']:.4f})")
    print(f"   • Cross-site consistency: ±{df_results.iloc[0]['LOSO_R2_Std']:.4f}")
else:
    print(f"   • No successful results to report")

print(f"   • Data samples: {len(y):,}")
print(f"   • Enhanced features: {len(feature_names)}")
print(f"   • Total time: {total_time/60:.2f} minutes")

print(f"\n📁 Enhanced Results saved in: {config.results_dir}")
print(f"   • Main report: enhanced_loso_benchmark_report.csv")
if site_df is not None:
    print(f"   • Per-site results: loso_per_site_results.csv")
print(f"   • Detailed results: enhanced_loso_results.json")
print(f"   • Feature info: enhanced_feature_info.json")
print(f"   • Visualizations: *.png files")

print(f"\n🚀 Key Fixes Implemented:")
print(f"   ✅ FIXED: Lambda function calling pattern")
print(f"   ✅ FIXED: Loss function compatibility with single outputs")
print(f"   ✅ FIXED: Ensemble array indexing and type consistency")
print(f"   ✅ FIXED: Proper error handling throughout pipeline")
print(f"   ✅ FIXED: Data loading robustness and validation")
print(f"   ✅ FIXED: Model registry to avoid instantiation issues")

print(f"\n🎯 Next Steps:")
print(f"   1. Review LOSO performance for deployment readiness")
print(f"   2. Consider running in FULL mode for production results")
print(f"   3. Implement best model in production pipeline")
print(f"   4. Monitor performance on new forest sites")

# Quick model ranking summary
if df_results is not None and len(df_results) > 0:
    print(f"\n📋 Enhanced LOSO Model Ranking:")
    for i, row in df_results.head(min(5, len(df_results))).iterrows():
        consistency = f"±{row['LOSO_R2_Std']:.3f}" if row['LOSO_R2_Std'] > 0 else ""
        print(f"   {i+1}. {row['Model']:<20} R² = {row['LOSO_R2']:.4f} {consistency}")

print(f"\n✨ Enhanced LOSO benchmark completed successfully! ✨")
print(f"🌟 All major errors have been fixed and pipeline is robust! 🌟")
