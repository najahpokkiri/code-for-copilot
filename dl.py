#!/usr/bin/env python
"""
Revised Spatial-Aware Biomass Preprocessing Script

This script preprocesses biomass and remote sensing data for machine learning while 
ensuring proper spatial handling and balanced data splits.

Author: najahpokkiri
Last Update: 2025-05-28
"""

import os
import sys
import json
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

class SpatialAwarePreprocessingConfig:
    """Configuration for spatial-aware biomass preprocessing."""
    
    # Input paths using the tuple format
    raster_pairs = [
        # Each tuple is (satellite_image_path, biomass_raster_path, site_name)
        ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif", "/teamspace/studios/dl2/clean/data/agbd_yellapur_reprojected_1.tif", "Site_1"),
        ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_betul_2020_clipped.tif","/teamspace/studios/dl2/clean/data/01_Betul_AGB40_band1_onImgGrid.tif", "Site_2"), 
        ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_achankumar_2020_clipped.tif", "/teamspace/studios/dl2/clean/data/02_Achanakmar_AGB40_band1_onImgGrid.tif", "Site_3"),
        ("/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_khaoyai_2020_clipped.tif","/teamspace/studios/dl2/clean/data/05_Khaoyai_AGB40_band1_onImgGrid.tif", "Site_4"),
        ('/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif', '/teamspace/studios/dl2/clean/data/04_Uppangala_AGB40_band1_onImgGrid.tif', "Site_5"),
    ]
    
    # Output paths
    output_dir = "spatial_biomass_results"
    processed_dir = "spatial_biomass_results/processed_data"
    
    # Preprocessing parameters
    chip_size = 24  # Size of extracted chips (pixels)
    overlap = 0.1   # Overlap between chips (as fraction of chip_size)
    
    # Data transformation
    use_log_transform = True  # Log transform biomass data
    
    # Quality filtering
    min_valid_pixels = 0.7  # Minimum fraction of valid pixels in a chip
    
    # NaN handling
    max_sat_nan_fraction = 0.3  # Maximum fraction of NaN allowed in satellite data
    
    # Split ratios
    test_ratio = 0.2   # Fraction of data for testing
    val_ratio = 0.15   # Fraction of training data for validation
    min_val_samples = 60  # Minimum number of validation samples
    
    def __init__(self):
        """Initialize and create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)


def find_input_files(config):
    """Load raster pairs from the tuple format configuration."""
    print("Loading specified raster pairs...")
    
    paired_files = []
    for pair in config.raster_pairs:
        sat_path, bio_path, site_id = pair
        
        # Check if files exist
        if not os.path.exists(bio_path):
            print(f"Warning: Biomass file not found: {bio_path}")
            continue
            
        if not os.path.exists(sat_path):
            print(f"Warning: Satellite file not found: {sat_path}")
            continue
            
        # Add to paired files
        paired_files.append({
            'site_id': site_id,
            'biomass_file': bio_path,
            'satellite_file': sat_path
        })
    
    if not paired_files:
        print("Error: No valid raster pairs found")
        return None
        
    print(f"Found {len(paired_files)} valid raster pairs:")
    for pair in paired_files:
        print(f"  Site {pair['site_id']}: {os.path.basename(pair['satellite_file'])} + {os.path.basename(pair['biomass_file'])}")
    
    return paired_files


def extract_chips(file_pairs, config):
    """Extract chips from biomass and satellite data rasters with robust NaN handling."""
    print("\n==== Extracting Chips ====")
    
    all_sat_chips = []
    all_bio_chips = []
    all_sources = []  # Track which site each chip comes from
    all_coords = []   # Store coordinates for each chip
    
    site_counts = []
    site_names = []
    site_coords = []  # Central coordinates of sites
    
    # Statistics for NaN handling
    nan_statistics = {
        'total_potential_chips': 0,
        'discarded_bio_nan': 0,
        'discarded_sat_excessive_nan': 0,
        'imputed_sat_minor_nan': 0,
        'clean_chips': 0
    }
    
    for i, file_pair in enumerate(file_pairs):
        site_id = file_pair['site_id']
        bio_file = file_pair['biomass_file']
        sat_file = file_pair['satellite_file']
        
        print(f"\nProcessing site {site_id}:")
        print(f"  Biomass: {bio_file}")
        print(f"  Satellite: {sat_file}")
        
        # Open raster files
        with rasterio.open(bio_file) as bio_src, rasterio.open(sat_file) as sat_src:
            # Read data
            bio_data = bio_src.read(1)  # Biomass data (single band)
            sat_data = sat_src.read()   # Satellite data (multiple bands)
            
            # Get site central coordinates
            site_center_x = bio_src.bounds.left + (bio_src.bounds.right - bio_src.bounds.left) / 2
            site_center_y = bio_src.bounds.bottom + (bio_src.bounds.top - bio_src.bounds.bottom) / 2
            site_coords.append([site_center_x, site_center_y])
            
            # Print shapes
            print(f"  Biomass shape: {bio_data.shape}")
            print(f"  Satellite shape: {sat_data.shape}")
            
            # Calculate chip extraction parameters
            height, width = bio_data.shape
            chip_size = config.chip_size
            stride = int(chip_size * (1 - config.overlap))
            
            # Calculate number of potential chips
            n_y = (height - chip_size) // stride + 1
            n_x = (width - chip_size) // stride + 1
            total_potential = n_y * n_x
            nan_statistics['total_potential_chips'] += total_potential
            print(f"  Potential chips: {total_potential} ({n_y}×{n_x})")
            
            # Extract chips
            site_chips = []
            site_bio = []
            site_coords_list = []
            
            site_stats = {
                'discarded_bio_nan': 0,
                'discarded_sat_excessive_nan': 0,
                'imputed_sat_minor_nan': 0,
                'clean_chips': 0
            }
            
            for y in range(0, height - chip_size + 1, stride):
                for x in range(0, width - chip_size + 1, stride):
                    # Extract biomass chip
                    bio_chip = bio_data[y:y+chip_size, x:x+chip_size].copy()
                    
                    # Check if biomass chip has enough valid data
                    valid_mask_bio = ~np.isnan(bio_chip)
                    if bio_src.nodata is not None:
                        valid_mask_bio = valid_mask_bio & (bio_chip != bio_src.nodata)
                    
                    valid_fraction_bio = np.sum(valid_mask_bio) / (chip_size * chip_size)
                    
                    # If biomass chip doesn't have enough valid data, skip it
                    if valid_fraction_bio < config.min_valid_pixels:
                        site_stats['discarded_bio_nan'] += 1
                        continue
                    
                    # Calculate mean biomass from valid pixels
                    mean_biomass = np.mean(bio_chip[valid_mask_bio])
                    
                    # Skip if biomass mean is invalid (e.g., all zeros leading to NaN after log transform)
                    if np.isnan(mean_biomass) or mean_biomass <= 0:
                        site_stats['discarded_bio_nan'] += 1
                        continue
                    
                    # Transform biomass if configured
                    if config.use_log_transform:
                        mean_biomass = np.log(mean_biomass + 1)
                    
                    # Extract satellite chip
                    sat_chip = sat_data[:, y:y+chip_size, x:x+chip_size].copy()
                    
                    # Check for NaN values in satellite data
                    nan_mask_sat = np.isnan(sat_chip)
                    nan_fraction_sat = np.sum(nan_mask_sat) / sat_chip.size
                    
                    # Handle satellite NaN values based on fraction
                    if nan_fraction_sat > 0:
                        if nan_fraction_sat > config.max_sat_nan_fraction:  # If too many NaNs, discard the chip
                            site_stats['discarded_sat_excessive_nan'] += 1
                            continue
                        else:
                            # Impute NaNs in satellite data - band by band, pixel by pixel median
                            for band_idx in range(sat_chip.shape[0]):
                                band = sat_chip[band_idx]
                                if np.any(np.isnan(band)):
                                    # Get median of valid values in this band for this chip
                                    band_valid_values = band[~np.isnan(band)]
                                    if len(band_valid_values) > 0:
                                        band_median = np.median(band_valid_values)
                                        band[np.isnan(band)] = band_median
                                    else:
                                        # If entire band is NaN, use 0 (last resort)
                                        band[np.isnan(band)] = 0
                            
                            site_stats['imputed_sat_minor_nan'] += 1
                    else:
                        site_stats['clean_chips'] += 1
                    
                    # Get pixel coordinates in the original raster
                    center_y, center_x = y + chip_size // 2, x + chip_size // 2
                    
                    # Convert to geo-coordinates if available
                    if bio_src.transform:
                        geo_x, geo_y = bio_src.xy(center_y, center_x)
                        chip_coord = (geo_x, geo_y)
                    else:
                        chip_coord = (center_x, center_y)
                    
                    # Final check for any remaining NaNs
                    if np.any(np.isnan(sat_chip)):
                        print(f"WARNING: NaNs still present after imputation in chip at ({x},{y})")
                        # Replace any remaining NaNs with zeros as last resort
                        sat_chip = np.nan_to_num(sat_chip, nan=0.0)
                    
                    # Add to list
                    site_chips.append(sat_chip)
                    site_bio.append(mean_biomass)
                    site_coords_list.append(chip_coord)
            
            # Add site data to global lists
            n_chips = len(site_chips)
            if n_chips > 0:
                all_sat_chips.extend(site_chips)
                all_bio_chips.extend(site_bio)
                all_sources.extend([i] * n_chips)
                all_coords.extend(site_coords_list)
                
                site_counts.append(n_chips)
                site_names.append(site_id)
                
                # Update global statistics
                for key in site_stats:
                    nan_statistics[key] += site_stats[key]
                
                print(f"  Extracted {n_chips} valid chips:")
                print(f"    - Clean chips: {site_stats['clean_chips']}")
                print(f"    - Chips with minor NaN imputation: {site_stats['imputed_sat_minor_nan']}")
                print(f"    - Discarded (excessive satellite NaN): {site_stats['discarded_sat_excessive_nan']}")
                print(f"    - Discarded (invalid biomass): {site_stats['discarded_bio_nan']}")
            else:
                print(f"  No valid chips extracted")
                site_counts.append(0)
                site_names.append(site_id)
    
    # Convert to numpy arrays
    X = np.array(all_sat_chips)
    y = np.array(all_bio_chips)
    sources = np.array(all_sources)
    
    # Print summary
    print(f"\nTotal extracted: {len(all_sat_chips)} chips from {len(site_counts)} sites")
    if nan_statistics['total_potential_chips'] > 0:
        print(f"NaN handling summary:")
        print(f"  - Total potential chips: {nan_statistics['total_potential_chips']}")
        print(f"  - Clean chips (no NaN): {nan_statistics['clean_chips']} ({100*nan_statistics['clean_chips']/nan_statistics['total_potential_chips']:.1f}%)")
        print(f"  - Chips with minor NaN imputation: {nan_statistics['imputed_sat_minor_nan']} ({100*nan_statistics['imputed_sat_minor_nan']/nan_statistics['total_potential_chips']:.1f}%)")
        print(f"  - Discarded (excessive satellite NaN): {nan_statistics['discarded_sat_excessive_nan']} ({100*nan_statistics['discarded_sat_excessive_nan']/nan_statistics['total_potential_chips']:.1f}%)")
        print(f"  - Discarded (invalid biomass): {nan_statistics['discarded_bio_nan']} ({100*nan_statistics['discarded_bio_nan']/nan_statistics['total_potential_chips']:.1f}%)")
        print(f"  - Total kept: {nan_statistics['clean_chips'] + nan_statistics['imputed_sat_minor_nan']} ({100*(nan_statistics['clean_chips'] + nan_statistics['imputed_sat_minor_nan'])/nan_statistics['total_potential_chips']:.1f}%)")
    
    for i, (count, name) in enumerate(zip(site_counts, site_names)):
        print(f"  Site {name}: {count} chips")
    
    site_info = {
        'counts': site_counts,
        'names': site_names,
        'coords': site_coords,
    }
    
    return X, y, sources, all_coords, site_info


def analyze_spatial_autocorrelation(X, y, coordinates, sources, site_info):
    """Analyze spatial autocorrelation in the data."""
    print("\n==== Analyzing Spatial Autocorrelation ====")
    
    site_autocorr = {}
    site_ranges = {}
    
    # Group by site
    unique_sites = np.unique(sources)
    
    for site in unique_sites:
        site_name = site_info['names'][site]
        mask = (sources == site)
        
        # Skip sites with too few samples
        if np.sum(mask) < 10:
            print(f"Skipping site {site_name} (too few samples: {np.sum(mask)})")
            continue
        
        site_y = y[mask]
        site_coords = [coordinates[i] for i in range(len(coordinates)) if mask[i]]
        
        # Convert coordinates to numpy array
        coord_array = np.array(site_coords)
        
        # Calculate distances between all pairs of points
        distances = squareform(pdist(coord_array))
        
        # Create array of absolute biomass differences
        biomass_diffs = np.abs(site_y.reshape(-1, 1) - site_y.reshape(1, -1))
        
        # Flatten the arrays (excluding self-comparisons)
        mask = ~np.eye(distances.shape[0], dtype=bool)
        distances_flat = distances[mask]
        biomass_diffs_flat = biomass_diffs[mask]
        
        # Calculate correlation
        correlation, pvalue = spearmanr(distances_flat, biomass_diffs_flat)
        
        # Store results
        site_autocorr[site] = {
            'correlation': correlation,
            'pvalue': pvalue,
            'n_samples': np.sum(mask)
        }
        
        # Estimate range of spatial autocorrelation
        # (simplified approach using distance bins)
        max_dist = np.max(distances_flat)
        bins = np.linspace(0, max_dist, 10)
        bin_indices = np.digitize(distances_flat, bins) - 1
        
        bin_corrs = []
        for i in range(len(bins) - 1):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 10:
                bin_corr, _ = spearmanr(
                    distances_flat[bin_mask], 
                    biomass_diffs_flat[bin_mask]
                )
                bin_corrs.append((bins[i] + bins[i+1]) / 2)
        
        # Find distance where correlation is below threshold
        # (simplified estimate of spatial range)
        autocorr_range = max_dist / 2  # Default to half the maximum distance
        
        # Store estimated range
        site_ranges[site] = autocorr_range
        
        print(f"Site {site_name}:")
        print(f"  Spatial autocorrelation: {correlation:.4f} (p={pvalue:.4f})")
        print(f"  Estimated autocorrelation range: {autocorr_range:.2f} units")
    
    return site_autocorr, site_ranges


def create_site_based_split(X, y, coordinates, sources, site_info, config):
    """Create a site-based train/val/test split that respects spatial boundaries."""
    print("\n==== Creating Spatially-Aware Data Split ====")
    
    # Get unique sites and their counts
    unique_sites = np.unique(sources)
    sites_with_counts = [(site, np.sum(sources == site), site_info['names'][i]) 
                        for i, site in enumerate(unique_sites)]
    
    # Sort sites by size for strategic selection
    sites_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Print site information
    print("Site distribution:")
    for site_id, count, name in sites_with_counts:
        print(f"  {name} (ID: {site_id}): {count} samples ({count/len(y)*100:.1f}%)")
    
    # Strategic site selection:
    # - Select medium-sized site for testing (Site_2 or Site_3)
    # - Use all other sites for train/val split
    
    # For testing, select either the second or third largest site
    # depending on which has a more suitable size
    if len(sites_with_counts) >= 3:
        if sites_with_counts[1][1] > 100:  # If second largest site has >100 samples
            test_sites = [sites_with_counts[1][0]]  # Use second largest site for testing
        else:
            test_sites = [sites_with_counts[2][0]]  # Use third largest site for testing
    else:
        # Fallback if fewer than 3 sites
        test_sites = [sites_with_counts[-1][0]]  # Use smallest site
    
    # Use all other sites for train/validation
    train_val_sites = [s[0] for s in sites_with_counts if s[0] not in test_sites]
    
    # Create masks for testing
    test_mask = np.zeros_like(sources, dtype=bool)
    for site in test_sites:
        test_mask |= (sources == site)
    
    # Create training/validation mask
    train_val_mask = np.zeros_like(sources, dtype=bool)
    for site in train_val_sites:
        train_val_mask |= (sources == site)
    
    # Randomly split training data into train and validation
    # (this ensures validation is representative)
    train_val_indices = np.where(train_val_mask)[0]
    np.random.shuffle(train_val_indices)
    
    # Ensure validation set has at least min_val_samples (default: 60)
    val_size = max(config.min_val_samples, int(config.val_ratio * len(train_val_indices)))
    val_indices = train_val_indices[:val_size]
    train_indices = train_val_indices[val_size:]
    
    # Create final masks
    train_mask = np.zeros_like(sources, dtype=bool)
    train_mask[train_indices] = True
    
    val_mask = np.zeros_like(sources, dtype=bool)
    val_mask[val_indices] = True
    
    # Get final split counts
    train_count = np.sum(train_mask)
    val_count = np.sum(val_mask)
    test_count = np.sum(test_mask)
    
    # Print split information
    print("\nFinal data split:")
    print(f"  Training: {train_count} samples ({train_count/len(y)*100:.1f}%)")
    print(f"  Validation: {val_count} samples ({val_count/len(y)*100:.1f}%)")
    print(f"  Testing: {test_count} samples ({test_count/len(y)*100:.1f}%)")
    
    # Print site distribution in each split
    train_site_counts = [(np.sum((sources == site) & train_mask), site_info['names'][i])
                         for i, site in enumerate(unique_sites)]
    val_site_counts = [(np.sum((sources == site) & val_mask), site_info['names'][i])
                       for i, site in enumerate(unique_sites)]
    test_site_counts = [(np.sum((sources == site) & test_mask), site_info['names'][i])
                        for i, site in enumerate(unique_sites)]
    
    print("\nTraining set site distribution:")
    for count, name in sorted(train_site_counts, reverse=True):
        if count > 0:
            print(f"  {name}: {count} samples ({count/train_count*100:.1f}%)")
    
    print("\nValidation set site distribution:")
    for count, name in sorted(val_site_counts, reverse=True):
        if count > 0:
            print(f"  {name}: {count} samples ({count/val_count*100:.1f}%)")
    
    print("\nTest set site distribution:")
    for count, name in sorted(test_site_counts, reverse=True):
        if count > 0:
            print(f"  {name}: {count} samples ({count/test_count*100:.1f}%)")
    
    # Calculate distance between training and testing samples to verify spatial separation
    train_coords = np.array([coordinates[i] for i, m in enumerate(train_mask) if m])
    test_coords = np.array([coordinates[i] for i, m in enumerate(test_mask) if m])
    
    if len(train_coords) > 0 and len(test_coords) > 0:
        # Calculate minimum distance between any training and test point
        min_distances = []
        
        # Use a subset for efficiency if datasets are very large
        max_points = 1000
        train_subset = train_coords[:min(max_points, len(train_coords))]
        test_subset = test_coords[:min(max_points, len(test_coords))]
        
        for test_point in test_subset:
            distances = np.sqrt(np.sum((train_subset - test_point)**2, axis=1))
            min_distances.append(np.min(distances))
        
        avg_min_distance = np.mean(min_distances)
        print(f"\nSpatial separation: Average minimum distance between train and test samples: {avg_min_distance:.2f} units")
    
    # Create dictionary with split information
    split_info = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_sites': [site_info['names'][i] for i, site in enumerate(unique_sites) if np.sum((sources == site) & train_mask) > 0],
        'val_sites': [site_info['names'][i] for i, site in enumerate(unique_sites) if np.sum((sources == site) & val_mask) > 0],
        'test_sites': [site_info['names'][i] for i, site in enumerate(unique_sites) if np.sum((sources == site) & test_mask) > 0]
    }
    
    return split_info


def save_processed_data(X, y, sources, coordinates, split_info, site_info, site_autocorr, config):
    """Save processed data for training."""
    print("\n==== Saving Processed Data ====")
    
    # Create timestamp for this preprocessing run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create paths
    X_path = os.path.join(config.processed_dir, f"X_{timestamp}.npy")
    y_path = os.path.join(config.processed_dir, f"y_{timestamp}.npy")
    sources_path = os.path.join(config.processed_dir, f"sources_{timestamp}.npy")
    coord_path = os.path.join(config.processed_dir, f"coordinates_{timestamp}.pkl")
    split_path = os.path.join(config.processed_dir, f"split_{timestamp}.npz")
    config_path = os.path.join(config.processed_dir, f"preprocessing_config_{timestamp}.json")
    latest_path = os.path.join(config.processed_dir, "latest.txt")
    
    # Save numpy arrays
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(sources_path, sources)
    
    # Save coordinates
    with open(coord_path, 'wb') as f:
        pickle.dump(coordinates, f)
    
    # Save split masks
    np.savez(split_path, 
             train_mask=split_info['train_mask'],
             val_mask=split_info['val_mask'], 
             test_mask=split_info['test_mask'])
    
    # Save preprocessing config and site information
    config_dict = {
        'timestamp': timestamp,
        'chip_size': config.chip_size,
        'overlap': config.overlap,
        'use_log_transform': config.use_log_transform,
        'site_counts': site_info['counts'],
        'site_names': site_info['names'],
        'train_sites': split_info['train_sites'],
        'val_sites': split_info['val_sites'],
        'test_sites': split_info['test_sites'],
        'spatial_autocorr': {str(site): {'correlation': float(info['correlation']), 
                                         'pvalue': float(info['pvalue'])} 
                            for site, info in site_autocorr.items()}
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Update latest timestamp reference
    with open(latest_path, 'w') as f:
        f.write(timestamp)
    
    print(f"Data saved with timestamp: {timestamp}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"sources shape: {sources.shape}")
    print(f"coordinates: {len(coordinates)} points")
    print(f"Split masks saved with train: {np.sum(split_info['train_mask'])}, " 
          f"val: {np.sum(split_info['val_mask'])}, test: {np.sum(split_info['test_mask'])} samples")


def main():
    """Main function for preprocessing biomass data."""
    print("=" * 80)
    print("Revised Spatial-Aware Biomass Preprocessing")
    print("=" * 80)
    
    # Initialize config
    config = SpatialAwarePreprocessingConfig()
    
    # Find input files
    file_pairs = find_input_files(config)
    if not file_pairs:
        sys.exit(1)
    
    # Extract chips from rasters
    X, y, sources, coordinates, site_info = extract_chips(file_pairs, config)
    
    # Analyze spatial autocorrelation
    site_autocorr, site_ranges = analyze_spatial_autocorrelation(
        X, y, coordinates, sources, site_info
    )
    
    # Create data split
    split_info = create_site_based_split(
        X, y, coordinates, sources, site_info, config
    )
    
    # Save processed data
    save_processed_data(
        X, y, sources, coordinates, split_info, 
        site_info, site_autocorr, config
    )
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
================================================================================
Revised Spatial-Aware Biomass Preprocessing
================================================================================
Loading specified raster pairs...
Found 5 valid raster pairs:
  Site Site_1: s1_s2_l8_palsar_ch_dem_yellapur_2020.tif + agbd_yellapur_reprojected_1.tif
  Site Site_2: s1_s2_l8_palsar_ch_betul_2020_clipped.tif + 01_Betul_AGB40_band1_onImgGrid.tif
  Site Site_3: s1_s2_l8_palsar_ch_goa_achankumar_2020_clipped.tif + 02_Achanakmar_AGB40_band1_onImgGrid.tif
  Site Site_4: s1_s2_l8_palsar_ch_goa_khaoyai_2020_clipped.tif + 05_Khaoyai_AGB40_band1_onImgGrid.tif
  Site Site_5: s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif + 04_Uppangala_AGB40_band1_onImgGrid.tif

==== Extracting Chips ====

Processing site Site_1:
  Biomass: /teamspace/studios/dl2/clean/data/agbd_yellapur_reprojected_1.tif
  Satellite: /teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif
  Biomass shape: (665, 658)
  Satellite shape: (59, 665, 658)
  Potential chips: 961 (31×31)
  Extracted 421 valid chips:
    - Clean chips: 372
    - Chips with minor NaN imputation: 49
    - Discarded (excessive satellite NaN): 0
    - Discarded (invalid biomass): 540

Processing site Site_2:
  Biomass: /teamspace/studios/dl2/clean/data/01_Betul_AGB40_band1_onImgGrid.tif
  Satellite: /teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_betul_2020_clipped.tif
  Biomass shape: (252, 266)
  Satellite shape: (59, 252, 266)
  Potential chips: 132 (11×12)
  Extracted 132 valid chips:
    - Clean chips: 132
    - Chips with minor NaN imputation: 0
    - Discarded (excessive satellite NaN): 0
    - Discarded (invalid biomass): 0

Processing site Site_3:
  Biomass: /teamspace/studios/dl2/clean/data/02_Achanakmar_AGB40_band1_onImgGrid.tif
  Satellite: /teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_achankumar_2020_clipped.tif
  Biomass shape: (275, 284)
  Satellite shape: (59, 275, 284)
  Potential chips: 156 (12×13)
  Extracted 156 valid chips:
    - Clean chips: 156
    - Chips with minor NaN imputation: 0
    - Discarded (excessive satellite NaN): 0
    - Discarded (invalid biomass): 0

Processing site Site_4:
  Biomass: /teamspace/studios/dl2/clean/data/05_Khaoyai_AGB40_band1_onImgGrid.tif
  Satellite: /teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_khaoyai_2020_clipped.tif
  Biomass shape: (185, 181)
  Satellite shape: (59, 185, 181)
  Potential chips: 64 (8×8)
  Extracted 63 valid chips:
    - Clean chips: 41
    - Chips with minor NaN imputation: 22
    - Discarded (excessive satellite NaN): 0
    - Discarded (invalid biomass): 1

Processing site Site_5:
  Biomass: /teamspace/studios/dl2/clean/data/04_Uppangala_AGB40_band1_onImgGrid.tif
  Satellite: /teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_uppangala_2020_clipped.tif
  Biomass shape: (103, 66)
  Satellite shape: (59, 103, 66)
  Potential chips: 12 (4×3)
  Extracted 12 valid chips:
    - Clean chips: 10
    - Chips with minor NaN imputation: 2
    - Discarded (excessive satellite NaN): 0
    - Discarded (invalid biomass): 0

Total extracted: 784 chips from 5 sites
NaN handling summary:
  - Total potential chips: 1325
  - Clean chips (no NaN): 711 (53.7%)
  - Chips with minor NaN imputation: 73 (5.5%)
  - Discarded (excessive satellite NaN): 0 (0.0%)
  - Discarded (invalid biomass): 541 (40.8%)
  - Total kept: 784 (59.2%)
  Site Site_1: 421 chips
  Site Site_2: 132 chips
  Site Site_3: 156 chips
  Site Site_4: 63 chips
  Site Site_5: 12 chips

==== Analyzing Spatial Autocorrelation ====
Site Site_1:
  Spatial autocorrelation: 0.0617 (p=0.0000)
  Estimated autocorrelation range: 0.12 units
Site Site_2:
  Spatial autocorrelation: 0.1527 (p=0.0000)
  Estimated autocorrelation range: 0.06 units
Site Site_3:
  Spatial autocorrelation: 0.1670 (p=0.0000)
  Estimated autocorrelation range: 0.06 units
Site Site_4:
  Spatial autocorrelation: -0.1058 (p=0.0000)
  Estimated autocorrelation range: 0.04 units
Site Site_5:
  Spatial autocorrelation: 0.2680 (p=0.0019)
  Estimated autocorrelation range: 0.01 units

==== Creating Spatially-Aware Data Split ====
Site distribution:
  Site_1 (ID: 0): 421 samples (53.7%)
  Site_3 (ID: 2): 156 samples (19.9%)
  Site_2 (ID: 1): 132 samples (16.8%)
  Site_4 (ID: 3): 63 samples (8.0%)
  Site_5 (ID: 4): 12 samples (1.5%)

Final data split:
  Training: 534 samples (68.1%)
  Validation: 94 samples (12.0%)
  Testing: 156 samples (19.9%)

Training set site distribution:
  Site_1: 362 samples (67.8%)
  Site_2: 111 samples (20.8%)
  Site_4: 51 samples (9.6%)
  Site_5: 10 samples (1.9%)

Validation set site distribution:
  Site_1: 59 samples (62.8%)
  Site_2: 21 samples (22.3%)
  Site_4: 12 samples (12.8%)
  Site_5: 2 samples (2.1%)

Test set site distribution:
  Site_3: 156 samples (100.0%)

Spatial separation: Average minimum distance between train and test samples: 4.29 units

==== Saving Processed Data ====
Data saved with timestamp: 20250528_014647
X shape: (784, 59, 24, 24)
y shape: (784,)
sources shape: (784,)
coordinates: 784 points
Split masks saved with train: 534, val: 94, test: 156 samples

================================================================================
✅ PREPROCESSING COMPLETED SUCCESSFULLY!
================================================================================

"""

#!/usr/bin/env python
"""
Hybrid Site-Spatial Cross-Validation for Biomass Prediction

This script implements a combined site and spatial approach for biomass prediction,
ensuring all sites are represented in training while respecting spatial autocorrelation.

Author: najahpokkiri
Date: 2025-05-28
"""

import os
import sys
import json
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

# Suppress warnings
warnings.filterwarnings('ignore')


# ======================================================================
# CONFIGURATION SETTINGS
# ======================================================================

class HybridCVConfig:
    """Configuration for hybrid site-spatial CV biomass model training."""
    
    # Paths
    preprocessed_dir = "spatial_biomass_results/processed_data"
    results_dir = "spatial_biomass_results/hybrid_results"
    visualization_dir = "spatial_biomass_results/hybrid_visualizations"
    cv_dir = "spatial_biomass_results/hybrid_cv"
    
    # Cross-validation settings
    n_folds = 5                      # Number of CV folds
    spatial_buffer = 20              # Buffer distance between train and test
    min_site_samples = 10            # Minimum samples required for site splitting
    
    # Model configuration
    model_type = "cnn_coordinate"    # Spatial-aware CNN model
    
    # Feature engineering
    add_derived_features = True      # Add vegetation indices
    standardize_features = True      # Standardize features
    
    # Loss function
    loss_function = "huber"          # Options: "mse", "huber", "spatial"
    huber_delta = 1.0                # Delta parameter for Huber loss
    spatial_loss_weight = 0.2        # Weight for spatial loss component
    
    # Training parameters
    batch_size = 16
    num_epochs = 100
    base_learning_rate = 0.001
    weight_decay = 5e-3              # Strong regularization
    early_stopping_patience = 20
    
    # Data augmentation
    use_geometric_aug = True
    use_spectral_aug = True
    aug_probability = 0.7
    
    # Advanced sampling
    use_hard_negative_mining = True   # Focus on difficult samples
    hard_negative_start_epoch = 20    # Start hard negative after this epoch
    oversampling_factor = 2.0         # Oversample by this factor
    
    # Test-time augmentation
    use_test_time_augmentation = True # Apply augmentation at test time
    tta_samples = 3                   # Number of augmentations per test sample
    
    def __init__(self):
        """Initialize and create output directories."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.cv_dir, exist_ok=True)


# ======================================================================
# MODEL DEFINITIONS
# ======================================================================

class HuberLoss(nn.Module):
    """Huber loss function that is less sensitive to outliers."""
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + self.delta * linear)


class SpatialLoss(nn.Module):
    """Loss function that penalizes spatial autocorrelation in residuals."""
    
    def __init__(self, mse_weight=0.8, spatial_weight=0.2, device=None):
        super(SpatialLoss, self).__init__()
        self.mse_weight = mse_weight
        self.spatial_weight = spatial_weight
        self.device = device
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true, coordinates=None):
        # Standard MSE loss
        mse_loss = self.mse(y_pred, y_true)
        
        # If no coordinates provided, just return MSE
        if coordinates is None:
            return mse_loss
        
        # Calculate residuals
        residuals = y_pred - y_true
        
        # Convert coordinates to tensor if needed
        if not torch.is_tensor(coordinates):
            coordinates = torch.tensor(coordinates).float().to(self.device)
        
        # Calculate spatial weights (simplified)
        # For efficiency on larger batches, use a subsample if n is large
        n = len(residuals)
        if n > 100:
            indices = torch.randperm(n)[:100]
            sub_coords = coordinates[indices]
            sub_residuals = residuals[indices]
            n = 100
        else:
            sub_coords = coordinates
            sub_residuals = residuals
        
        # Calculate spatial penalty
        spatial_penalty = 0.0
        
        # For each point, find the nearest points and check residual similarity
        for i in range(n):
            # Calculate distances from point i to all other points
            dists = torch.sqrt(torch.sum((sub_coords - sub_coords[i]).pow(2), dim=1))
            
            # Get indices of nearest points (excluding self)
            _, indices = torch.topk(dists, min(11, n), largest=False)
            indices = indices[1:]  # Remove self
            
            # Calculate residual differences
            res_diffs = torch.abs(sub_residuals[indices] - sub_residuals[i])
            
            if len(indices) > 1:
                # Normalize distances to [0,1]
                norm_dists = dists[indices] / (torch.max(dists[indices]) + 1e-8)
                
                # Calculate penalty (closer points should have similar residuals)
                penalty = torch.mean(torch.abs(res_diffs - norm_dists))
                spatial_penalty += penalty
        
        spatial_penalty = spatial_penalty / n
        
        # Combine MSE and spatial penalty
        total_loss = self.mse_weight * mse_loss + self.spatial_weight * spatial_penalty
        
        return total_loss


class CNNCoordinateModel(nn.Module):
    """CNN with coordinate channels for spatial awareness using InstanceNorm."""
    
    def __init__(self, input_channels, height, width):
        super(CNNCoordinateModel, self).__init__()
        
        # Account for 2 additional coordinate channels
        self.input_channels = input_channels + 2
        
        # Enhanced CNN architecture using InstanceNorm instead of BatchNorm
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        # Calculate size after convolutions and pooling
        conv_height = height // 4
        conv_width = width // 4
        
        # Adjust if dimensions get too small
        if conv_height < 1: conv_height = 1
        if conv_width < 1: conv_width = 1
            
        # Fully connected layers with Layer Normalization
        self.fc1 = nn.Linear(128 * conv_height * conv_width, 128)
        self.norm5 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.norm6 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Create coordinate channels
        x_coords = torch.linspace(-1, 1, width).view(1, 1, 1, width).repeat(batch_size, 1, height, 1).to(x.device)
        y_coords = torch.linspace(-1, 1, height).view(1, 1, height, 1).repeat(batch_size, 1, 1, width).to(x.device)
        
        # Concatenate coordinate channels
        x = torch.cat([x, x_coords, y_coords], dim=1)
        
        # Convolutional layers
        x1 = self.relu(self.norm1(self.conv1(x)))
        x1 = self.pool(x1)
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.norm2(self.conv2(x1)))
        x2 = self.pool(x2)
        x2 = self.dropout(x2)
        
        x3 = self.relu(self.norm3(self.conv3(x2)))
        x3 = self.dropout(x3)
        
        # Flatten
        x = x3.view(x3.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.norm5(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.squeeze(1)

def create_model(model_type, input_channels, height, width, device):
    """Create a model based on the specified type."""
    print(f"\nCreating {model_type.upper()} model...")
    
    if model_type.lower() == "cnn_coordinate":
        model = CNNCoordinateModel(input_channels, height, width).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def create_loss_function(config, device=None):
    """Create a loss function based on the configuration."""
    loss_type = config.loss_function.lower()
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return HuberLoss(delta=config.huber_delta)
    elif loss_type == "spatial":
        return SpatialLoss(mse_weight=1.0-config.spatial_loss_weight,
                          spatial_weight=config.spatial_loss_weight,
                          device=device)
    else:
        print(f"Unknown loss function: {loss_type}, defaulting to MSE")
        return nn.MSELoss()


# ======================================================================
# DATA LOADING AND PROCESSING
# ======================================================================

def load_preprocessed_data(config):
    """Load the latest preprocessed data."""
    print("\n==== Loading Preprocessed Data ====")
    
    # Find latest timestamp
    latest_path = os.path.join(config.preprocessed_dir, "latest.txt")
    
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            timestamp = f.read().strip()
    else:
        # Find most recent files if latest.txt doesn't exist
        pattern = "X_*.npy"
        files = list(Path(config.preprocessed_dir).glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No preprocessed data found in {config.preprocessed_dir}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_file.name.split('_')[1].split('.')[0]
    
    # Load data
    X_path = os.path.join(config.preprocessed_dir, f"X_{timestamp}.npy")
    y_path = os.path.join(config.preprocessed_dir, f"y_{timestamp}.npy")
    sources_path = os.path.join(config.preprocessed_dir, f"sources_{timestamp}.npy")
    coord_path = os.path.join(config.preprocessed_dir, f"coordinates_{timestamp}.pkl")
    
    # Check core files exist
    for path in [X_path, y_path, sources_path, coord_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    
    # Load arrays
    X = np.load(X_path)
    y = np.load(y_path)
    sources = np.load(sources_path)
    
    with open(coord_path, 'rb') as f:
        coordinates = pickle.load(f)
    
    # Get config file if it exists
    config_path = os.path.join(config.preprocessed_dir, f"preprocessing_config_{timestamp}.json")
    preprocess_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            preprocess_config = json.load(f)
    
    print(f"Loaded data with timestamp: {timestamp}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"sources shape: {sources.shape}")
    print(f"coordinates: {len(coordinates)} points")
    
    # Summarize site information
    unique_sites = np.unique(sources)
    site_counts = [np.sum(sources == s) for s in unique_sites]
    
    print("\nSite breakdown:")
    for site_id, count in zip(unique_sites, site_counts):
        site_name = f"Site_{site_id+1}"
        if 'site_names' in preprocess_config and site_id < len(preprocess_config['site_names']):
            site_name = preprocess_config['site_names'][site_id]
        print(f"  {site_name} (ID: {site_id}): {count} samples")
    
    # Create data dictionary
    data = {
        'X': X,
        'y': y,
        'coordinates': coordinates,
        'sources': sources,
        'timestamp': timestamp,
        'preprocess_config': preprocess_config
    }
    
    return data

def create_hybrid_site_spatial_split(coordinates, sources, config):
    """Create CV splits that ensure site representation while respecting spatial autocorrelation."""
    print("\n==== Creating Hybrid Site-Spatial CV Splits ====")
    
    unique_sites = np.unique(sources)
    n_sites = len(unique_sites)
    print(f"Found {n_sites} unique sites")
    
    # Create folds list
    folds = []
    
    # Set minimum required samples for each split
    min_train = 10  # Absolute minimum for training
    min_val = 5     # Absolute minimum for validation
    min_test = 5    # Absolute minimum for testing
    
    # Create a fixed number of folds
    for fold_idx in range(config.n_folds):
        # Initialize masks for this fold
        train_mask = np.zeros(len(coordinates), dtype=bool)
        val_mask = np.zeros(len(coordinates), dtype=bool)
        test_mask = np.zeros(len(coordinates), dtype=bool)
        
        # Process each site separately
        for site in unique_sites:
            # Get indices for this site
            site_indices = np.where(sources == site)[0]
            n_site_samples = len(site_indices)
            
            # For small sites, just add to training (skip splitting)
            if n_site_samples < (min_train + min_val + min_test):
                print(f"  Site {site} has only {n_site_samples} samples - adding all to training")
                train_mask[site_indices] = True
                continue
                
            # Get coordinates for this site
            site_coords = np.array([coordinates[i] for i in site_indices])
            
            # Use spatial clustering to ensure test samples are spatially coherent
            # Create more clusters than we need
            n_clusters = min(10, max(3, n_site_samples // 20))
            
            # Make sure we don't create too many clusters for small sites
            n_clusters = min(n_clusters, n_site_samples // (min_test * 2))
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=fold_idx + 42)
            clusters = kmeans.fit_predict(site_coords)
            
            # For this fold, select clusters that will go to test
            # Rotate which clusters based on fold index
            cluster_indices = np.arange(n_clusters)
            np.random.RandomState(fold_idx + 42).shuffle(cluster_indices)
            
            # Try different test cluster configurations until we find one that works
            valid_split = False
            for test_size_fraction in [0.2, 0.3, 0.4, 0.15, 0.25]:
                # Calculate how many clusters to assign to test
                n_test_clusters = max(1, int(round(n_clusters * test_size_fraction)))
                test_clusters = cluster_indices[:n_test_clusters]
                
                # Assign samples to appropriate splits
                site_test_indices = []
                site_train_val_indices = []
                
                for i, cluster in enumerate(clusters):
                    if cluster in test_clusters:
                        site_test_indices.append(site_indices[i])
                    else:
                        site_train_val_indices.append(site_indices[i])
                
                # Check if we have enough test samples
                if len(site_test_indices) < min_test:
                    continue
                    
                # Apply spatial buffer between train and test (with safety check)
                if config.spatial_buffer > 0 and len(site_test_indices) > 0:
                    # Get test coordinates
                    test_coords = np.array([coordinates[i] for i in site_test_indices])
                    
                    # Build KD tree for test coordinates
                    test_tree = cKDTree(test_coords)
                    
                    # Check each train/val point
                    filtered_train_val = []
                    for idx in site_train_val_indices:
                        point = coordinates[idx]
                        # Find distance to nearest test point
                        dist, _ = test_tree.query(point, k=1)
                        if dist >= config.spatial_buffer:
                            filtered_train_val.append(idx)
                    
                    # Only apply buffer if we'll have enough samples left
                    if len(filtered_train_val) >= (min_train + min_val):
                        site_train_val_indices = filtered_train_val
                
                # Check if we have enough train+val samples
                if len(site_train_val_indices) < (min_train + min_val):
                    continue
                
                # Split remaining samples into train/val
                np.random.RandomState(fold_idx + 42).shuffle(site_train_val_indices)
                n_val = max(min_val, int(len(site_train_val_indices) * 0.2))
                n_val = min(n_val, len(site_train_val_indices) - min_train)  # Ensure enough for training
                
                site_val_indices = site_train_val_indices[:n_val]
                site_train_indices = site_train_val_indices[n_val:]
                
                # Double-check we have enough samples in each split
                if len(site_train_indices) >= min_train and len(site_val_indices) >= min_val:
                    valid_split = True
                    break
            
            # If we couldn't find a valid split, use a simple ratio split with no spatial clustering
            if not valid_split:
                print(f"  Warning: Couldn't create spatial clusters for Site {site}, using simple split")
                np.random.RandomState(fold_idx + 42).shuffle(site_indices)
                n_test = max(min_test, int(len(site_indices) * 0.2))
                n_val = max(min_val, int(len(site_indices) * 0.2))
                n_train = len(site_indices) - n_test - n_val
                
                if n_train < min_train:  # Still not enough for a valid split
                    print(f"  Site {site} split problem - adding all to training")
                    train_mask[site_indices] = True
                    continue
                
                site_test_indices = site_indices[:n_test]
                site_val_indices = site_indices[n_test:n_test+n_val]
                site_train_indices = site_indices[n_test+n_val:]
            
            # Update masks
            test_mask[site_test_indices] = True
            val_mask[site_val_indices] = True
            train_mask[site_train_indices] = True
            
            print(f"  Site {site}: Train={len(site_train_indices)}, Val={len(site_val_indices)}, Test={len(site_test_indices)}")
        
        # Verify we have data in all splits before adding this fold
        if np.sum(train_mask) == 0 or np.sum(val_mask) == 0 or np.sum(test_mask) == 0:
            print(f"Warning: Fold {fold_idx+1} has an empty split, skipping")
            continue
            
        # Add fold to list
        folds.append({
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'fold_idx': fold_idx
        })
        
        print(f"Fold {fold_idx+1}: Train={np.sum(train_mask)}, Val={np.sum(val_mask)}, Test={np.sum(test_mask)}")
    
    # Make sure we have at least one fold
    if len(folds) == 0:
        raise ValueError("Could not create any valid folds with current settings. Try reducing min_site_samples or spatial_buffer.")
    
    return folds

# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def add_derived_features(X):
    """Add derived features like NDVI, EVI, etc."""
    print("\nAdding derived features...")
    
    # Check if we have enough bands
    if X.shape[1] < 5:
        print("Warning: Not enough bands for derived features, skipping")
        return X
    
    # Assuming standardized band positions:
    # Band 1: Blue
    # Band 2: Green
    # Band 3: Red
    # Band 4: NIR
    blue_idx, green_idx, red_idx, nir_idx = 1, 2, 3, 4
    
    # Make a copy to avoid modifying original
    X_new = X.copy()
    
    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    epsilon = 1e-8  # To avoid division by zero
    
    nir = X[:, nir_idx, :, :]
    red = X[:, red_idx, :, :]
    denominator = nir + red + epsilon
    ndvi[:, 0, :, :] = (nir - red) / denominator
    
    # Calculate EVI: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    evi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    blue = X[:, blue_idx, :, :]
    denominator = nir + 6*red - 7.5*blue + 1 + epsilon
    evi[:, 0, :, :] = 2.5 * (nir - red) / denominator
    
    # Calculate SAVI: ((NIR - Red) / (NIR + Red + 0.5)) * (1.5)
    savi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    denominator = nir + red + 0.5 + epsilon
    savi[:, 0, :, :] = ((nir - red) / denominator) * 1.5
    
    # Calculate GNDVI: (NIR - Green) / (NIR + Green)
    gndvi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    green = X[:, green_idx, :, :]
    denominator = nir + green + epsilon
    gndvi[:, 0, :, :] = (nir - green) / denominator
    
    # Calculate NDWI: (Green - NIR) / (Green + NIR)
    ndwi = np.zeros((X.shape[0], 1, X.shape[2], X.shape[3]))
    denominator = green + nir + epsilon
    ndwi[:, 0, :, :] = (green - nir) / denominator
    
    # Add new features to X
    X_new = np.concatenate([X_new, ndvi, evi, savi, gndvi, ndwi], axis=1)
    
    # Replace any NaN values with 0
    X_new = np.nan_to_num(X_new, nan=0.0)
    
    print(f"Added 5 derived features, new shape: {X_new.shape}")
    return X_new


def standardize_features(X_train, X_val=None, X_test=None):
    """Standardize features by band, using training set statistics."""
    print("\nStandardizing features...")
    
    # Initialize output arrays
    X_train_std = np.zeros_like(X_train)
    X_val_std = None if X_val is None else np.zeros_like(X_val)
    X_test_std = None if X_test is None else np.zeros_like(X_test)
    
    # Standardize each band separately
    for b in range(X_train.shape[1]):
        # Get band data and reshape to 2D
        band_train = X_train[:, b, :, :].reshape(X_train.shape[0], -1)
        
        # Calculate mean and std
        band_mean = np.mean(band_train)
        band_std = np.std(band_train)
        
        # Handle constant bands
        if band_std == 0:
            band_std = 1.0
        
        # Standardize training data
        X_train_std[:, b, :, :] = ((X_train[:, b, :, :] - band_mean) / band_std)
        
        # Standardize validation data if provided
        if X_val is not None:
            X_val_std[:, b, :, :] = ((X_val[:, b, :, :] - band_mean) / band_std)
        
        # Standardize test data if provided
        if X_test is not None:
            X_test_std[:, b, :, :] = ((X_test[:, b, :, :] - band_mean) / band_std)
    
    # Replace any NaN values with 0
    X_train_std = np.nan_to_num(X_train_std, nan=0.0)
    
    if X_val is not None:
        X_val_std = np.nan_to_num(X_val_std, nan=0.0)
    
    if X_test is not None:
        X_test_std = np.nan_to_num(X_test_std, nan=0.0)
    
    return X_train_std, X_val_std, X_test_std


# ======================================================================
# DATA AUGMENTATION
# ======================================================================

def create_data_augmentation(config):
    """Create data augmentation transformations."""
    if not (config.use_geometric_aug or config.use_spectral_aug) or config.aug_probability <= 0:
        return None
    
    print("\nSetting up data augmentation:")
    transforms_list = []
    
    # Geometric augmentations
    if config.use_geometric_aug:
        print("  - Using geometric augmentation (flips, rotations)")
        
        # Custom transforms that work with 3D tensors (C, H, W)
        class RandomHorizontalFlip:
            def __init__(self, p=0.5):
                self.p = p
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    return torch.flip(x, [1])  # Flip height dimension (dim 1 for 3D tensor)
                return x
        
        class RandomVerticalFlip:
            def __init__(self, p=0.5):
                self.p = p
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    return torch.flip(x, [2])  # Flip width dimension (dim 2 for 3D tensor)
                return x
        
        class RandomRotation90:
            def __init__(self, p=0.5):
                self.p = p
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 for 90, 180, 270 degrees
                    return torch.rot90(x, k, [1, 2])  # For 3D tensor, rotate dims 1 & 2
                return x
        
        transforms_list.extend([
            RandomHorizontalFlip(p=config.aug_probability),
            RandomVerticalFlip(p=config.aug_probability),
            RandomRotation90(p=config.aug_probability)
        ])
    
    # Spectral augmentations
    if config.use_spectral_aug:
        print("  - Using spectral augmentation (band jittering)")
        
        # Custom transform for multi-band spectral augmentation
        class SpectralJitter:
            def __init__(self, p=0.5, brightness_factor=0.1, contrast_factor=0.1):
                self.p = p
                self.brightness_factor = brightness_factor
                self.contrast_factor = contrast_factor
                
            def __call__(self, x):
                if torch.rand(1) < self.p:
                    # Select random bands to modify (1-5 bands)
                    num_bands = x.shape[0]  # First dimension is channels for 3D tensor
                    num_to_modify = torch.randint(1, min(6, max(2, num_bands // 3)), (1,)).item()
                    bands_to_modify = torch.randperm(num_bands)[:num_to_modify]
                    
                    # Make a copy
                    x_aug = x.clone()
                    
                    for band_idx in bands_to_modify:
                        # Random brightness adjustment
                        if torch.rand(1) < 0.5:
                            brightness_change = 1.0 + (torch.rand(1) * 2 - 1) * self.brightness_factor
                            x_aug[band_idx] = x_aug[band_idx] * brightness_change
                        
                        # Random contrast adjustment
                        if torch.rand(1) < 0.5:
                            contrast_change = 1.0 + (torch.rand(1) * 2 - 1) * self.contrast_factor
                            mean = torch.mean(x_aug[band_idx])
                            x_aug[band_idx] = (x_aug[band_idx] - mean) * contrast_change + mean
                    
                    return x_aug
                return x
        
        transforms_list.append(SpectralJitter(p=config.aug_probability))
    
    # Print summary
    print(f"  - Augmentation probability: {config.aug_probability}")
    print(f"  - Total transformations: {len(transforms_list)}")
    
    # Compose the transforms
    class SequentialTransforms:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x
    
    return SequentialTransforms(transforms_list)


# ======================================================================
# ADVANCED SAMPLING
# ======================================================================

class HardNegativeMiningDataset(Dataset):
    """Dataset with hard negative mining capabilities."""
    
    def __init__(self, X, y, coordinates, model=None, device=None, transforms=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.coordinates = coordinates
        self.transforms = transforms
        self.model = model
        self.device = device
        
        # Error tracking for hard negative mining
        self.errors = torch.zeros_like(self.y)
        self.has_errors = False
    
    def update_errors(self, model, device):
        """Update error metrics for each sample."""
        self.model = model
        self.device = device
        
        # Skip if model is not provided
        if model is None:
            self.has_errors = False
            return
        
        # Set model to evaluation mode
        model.eval()
        
        # Calculate errors for each sample
        errors = []
        with torch.no_grad():
            for i in range(len(self.X)):
                x = self.X[i:i+1].to(device)
                y_true = self.y[i].item()
                
                # Forward pass
                y_pred = model(x).item()
                
                # Calculate error
                error = abs(y_pred - y_true)
                errors.append(error)
        
        # Update errors tensor
        self.errors = torch.tensor(errors)
        self.has_errors = True
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        coords = self.coordinates[idx]
        
        # Apply augmentations if available
        if self.transforms:
            x = self.transforms(x)
        
        # Include error if available
        if self.has_errors:
            return x, y, coords, self.errors[idx]
        
        return x, y, coords


def create_hard_negative_sampler(dataset, config):
    """Create a sampler that prioritizes hard examples."""
    if not config.use_hard_negative_mining or not dataset.has_errors:
        return None
    
    print("\nCreating hard negative mining sampler:")
    
    # Calculate weights based on errors
    errors = dataset.errors.numpy()
    
    # Scale errors to [0,1] for numerical stability
    if np.max(errors) > np.min(errors):
        weights = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
    else:
        weights = np.ones_like(errors)
    
    # Add small constant to ensure all samples get some weight
    weights = weights + 0.1
    
    # Square weights to emphasize harder examples
    weights = weights ** 2
    
    # Display error distribution
    print("  - Error distribution:")
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        print(f"    {p}th percentile: {np.percentile(errors, p):.4f}")
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Create sampler
    num_samples = int(len(weights) * config.oversampling_factor)
    print(f"  - Using {num_samples} samples with hard negative mining")
    
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=num_samples)


# ======================================================================
# TRAINING AND EVALUATION
# ======================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                config, device, coordinates_val=None):
    """Train the model with early stopping and learning rate scheduling."""
    print("\nTraining model...")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config.num_epochs,
                                 eta_min=config.base_learning_rate / 10)
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Reset hard negative mining dataset if configured
        if isinstance(train_loader.dataset, HardNegativeMiningDataset) and epoch >= config.hard_negative_start_epoch:
            train_loader.dataset.update_errors(model, device)
            # Re-create sampler based on updated errors
            if config.use_hard_negative_mining:
                hard_negative_sampler = create_hard_negative_sampler(train_loader.dataset, config)
                if hard_negative_sampler is not None:
                    train_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=config.batch_size,
                        sampler=hard_negative_sampler
                    )
        
        # Training batches
        batch_losses = []
        for batch in train_loader:
            # Unpack batch
            if len(batch) == 4:
                inputs, targets, coords, _ = batch  # With error info
            else:
                inputs, targets, coords = batch     # Without error info
            
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if isinstance(criterion, SpatialLoss):
                # For spatial loss, include coordinates
                batch_coords = torch.tensor(coords, dtype=torch.float32).to(device)
                loss = criterion(outputs, targets, batch_coords)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            batch_losses.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, coords in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss - use standard MSE for validation
                loss = F.mse_loss(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Step learning rate scheduler
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config.num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}", end="")
        
        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"\n  → New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print("")  # Newline
                
        # Early stopping check
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rates": learning_rates,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses)
    }


def test_time_augmentation(model, X_test, config, device):
    """Apply test-time augmentation to improve prediction accuracy."""
    if not config.use_test_time_augmentation:
        return None
    
    print("\nApplying test-time augmentation...")
    
    # Create augmentation transforms
    transforms = create_data_augmentation(config)
    if transforms is None:
        return None
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_test)
    
    # Set model to evaluation mode
    model.eval()
    
    # Store predictions
    all_predictions = []
    
    # Make predictions with multiple augmentations
    with torch.no_grad():
        # Original prediction
        batch_size = 32
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            all_predictions.append(outputs.cpu())
        
        # Augmented predictions
        for aug_idx in range(config.tta_samples):
            print(f"  - TTA iteration {aug_idx+1}/{config.tta_samples}")
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].clone()
                
                # Apply augmentation to each sample
                for j in range(len(batch)):
                    batch[j] = transforms(batch[j])
                
                # Move to device and predict
                batch = batch.to(device)
                outputs = model(batch)
                all_predictions.append(outputs.cpu())
    
    # Combine predictions
    all_predictions = torch.cat(all_predictions).reshape(config.tta_samples + 1, len(X_test))
    y_pred = torch.mean(all_predictions, dim=0).numpy()
    
    print(f"  - Final predictions created from {config.tta_samples + 1} versions")
    
    return y_pred


def evaluate_model(model, X_test, y_test, coordinates_test, sources_test, config, device):
    """Evaluate the model on test data."""
    print("\nEvaluating model...")
    
    # Convert test data to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions - use test-time augmentation if configured
    if config and config.use_test_time_augmentation:
        y_pred = test_time_augmentation(model, X_test, config, device)
    else:
        # Standard prediction
        with torch.no_grad():
            predictions = []
            batch_size = 32
            
            for i in range(0, len(X_test), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                outputs = model(batch)
                predictions.append(outputs.cpu().numpy())
            
            y_pred = np.concatenate(predictions)
    
    # Handle NaN values
    valid_mask = ~(np.isnan(y_test) | np.isnan(y_pred))
    if not np.all(valid_mask):
        print(f"WARNING: Found {np.sum(~valid_mask)} NaN values. These will be excluded from metrics.")
        
        # Use only valid values for metric calculation
        y_test_valid = y_test[valid_mask]
        y_pred_valid = y_pred[valid_mask]
    else:
        y_test_valid = y_test
        y_pred_valid = y_pred
    
    # Calculate metrics
    metrics = {}
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_valid))
    metrics['rmse'] = rmse
    
    # R²
    r2 = r2_score(y_test_valid, y_pred_valid)
    metrics['r2'] = r2
    
    # MAE
    mae = mean_absolute_error(y_test_valid, y_pred_valid)
    metrics['mae'] = mae
    
    # Spearman correlation
    spearman, _ = spearmanr(y_test_valid, y_pred_valid)
    metrics['spearman'] = spearman
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'residual': np.where(valid_mask, y_pred - y_test, np.nan),
        'source': sources_test,
        'valid': valid_mask
    })
    
    # Add coordinates
    results_df['x_coord'] = [coord[0] for coord in coordinates_test]
    results_df['y_coord'] = [coord[1] for coord in coordinates_test]
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Spearman correlation: {spearman:.4f}")
    
    # Analyze by site if multiple sites present
    if len(np.unique(sources_test)) > 1:
        print("\nMetrics by site:")
        for site in np.unique(sources_test):
            site_mask = (sources_test == site) & valid_mask
            if np.sum(site_mask) > 0:
                site_rmse = np.sqrt(mean_squared_error(y_test[site_mask], y_pred[site_mask]))
                site_r2 = r2_score(y_test[site_mask], y_pred[site_mask])
                site_mae = mean_absolute_error(y_test[site_mask], y_pred[site_mask])
                print(f"  Site {site}: RMSE={site_rmse:.4f}, R²={site_r2:.4f}, MAE={site_mae:.4f}, n={np.sum(site_mask)}")
                
                # Add site-specific metrics
                metrics[f'site_{site}_rmse'] = site_rmse
                metrics[f'site_{site}_r2'] = site_r2
                metrics[f'site_{site}_mae'] = site_mae
    
    return results_df, metrics


# ======================================================================
# SPATIAL CROSS-VALIDATION
# ======================================================================

def run_spatial_cv_fold(fold, data, config):
    """Run a single fold of spatial cross-validation."""
    print(f"\n{'='*40}")
    print(f"Running Hybrid CV Fold {fold['fold_idx']+1}")
    print(f"{'='*40}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get train/val/test data
    X_train = data['X'][fold['train_mask']]
    y_train = data['y'][fold['train_mask']]
    coords_train = [data['coordinates'][i] for i, mask in enumerate(fold['train_mask']) if mask]
    sources_train = data['sources'][fold['train_mask']]
    
    X_val = data['X'][fold['val_mask']]
    y_val = data['y'][fold['val_mask']]
    coords_val = [data['coordinates'][i] for i, mask in enumerate(fold['val_mask']) if mask]
    sources_val = data['sources'][fold['val_mask']]
    
    X_test = data['X'][fold['test_mask']]
    y_test = data['y'][fold['test_mask']]
    coords_test = [data['coordinates'][i] for i, mask in enumerate(fold['test_mask']) if mask]
    sources_test = data['sources'][fold['test_mask']]
    
    # Feature engineering
    if config.add_derived_features:
        X_train = add_derived_features(X_train)
        X_val = add_derived_features(X_val)
        X_test = add_derived_features(X_test)
    
    # Feature standardization
    if config.standardize_features:
        X_train, X_val, X_test = standardize_features(X_train, X_val, X_test)
    
    # Create data augmentation
    transforms = create_data_augmentation(config)
    
    # Create dataset
    if config.use_hard_negative_mining:
        train_dataset = HardNegativeMiningDataset(X_train, y_train, coords_train, transforms=transforms)
    else:
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(coords_train))
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.FloatTensor(coords_val))
    
    # # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    # Create data loaders with sensible batch sizes
    train_batch_size = min(config.batch_size, max(1, len(train_dataset) // 2))
    val_batch_size = min(config.batch_size, max(1, len(val_dataset) // 2))
        
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
        
    # Create model
    input_channels = X_train.shape[1]  # Number of channels
    height, width = X_train.shape[2], X_train.shape[3]  # Spatial dimensions
    
    model = create_model(config.model_type, input_channels, height, width, device)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config.base_learning_rate, 
                          weight_decay=config.weight_decay)
    
    # Define loss function
    criterion = create_loss_function(config, device)
    
    # Train model
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config, device
    )
    
    # Evaluate model
    results_df, metrics = evaluate_model(
        trained_model, X_test, y_test, coords_test, sources_test,
        config, device
    )
    
    return trained_model, results_df, metrics, history


def run_hybrid_cross_validation(data, config):
    """Run complete hybrid site-spatial cross-validation."""
    print("\n" + "=" * 80)
    print("HYBRID SITE-SPATIAL CROSS-VALIDATION")
    print("=" * 80)
    
    # Create CV folds
    folds = create_hybrid_site_spatial_split(data['coordinates'], data['sources'], config)
    
    # Store results
    fold_models = []
    fold_results = []
    fold_metrics = []
    fold_histories = []
    
    # Run each fold
    for fold_idx, fold in enumerate(folds):
        # Set random seeds for reproducibility
        torch.manual_seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + fold_idx)
        
        # Run fold
        model, results_df, metrics, history = run_spatial_cv_fold(fold, data, config)
        
        # Store results
        fold_models.append(model)
        fold_results.append(results_df)
        fold_metrics.append(metrics)
        fold_histories.append(history)
    
    # Calculate overall metrics
    print("\nHybrid CV Summary:")
    rmse_values = [m['rmse'] for m in fold_metrics]
    r2_values = [m['r2'] for m in fold_metrics]
    mae_values = [m['mae'] for m in fold_metrics]
    spearman_values = [m['spearman'] for m in fold_metrics]
    
    print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
    print(f"R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    print(f"MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
    print(f"Spearman: {np.mean(spearman_values):.4f} ± {np.std(spearman_values):.4f}")
    
    # Create timestamp for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create result directory
    result_dir = os.path.join(config.cv_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save CV results
    cv_summary = {
        'fold_metrics': fold_metrics,
        'mean_rmse': float(np.mean(rmse_values)),
        'std_rmse': float(np.std(rmse_values)),
        'mean_r2': float(np.mean(r2_values)),
        'std_r2': float(np.std(r2_values)),
        'mean_mae': float(np.mean(mae_values)),
        'std_mae': float(np.std(mae_values)),
        'mean_spearman': float(np.mean(spearman_values)),
        'std_spearman': float(np.std(spearman_values)),
    }
    
    # Save CV summary
    summary_path = os.path.join(result_dir, "cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    # Save fold results
    for i, results in enumerate(fold_results):
        results_path = os.path.join(result_dir, f"fold_{i+1}_results.csv")
        results.to_csv(results_path, index=False)
    
    # Save fold models
    for i, model in enumerate(fold_models):
        model_path = os.path.join(result_dir, f"fold_{i+1}_model.pt")
        torch.save(model.state_dict(), model_path)
    
    # Save config
    config_dict = {attr: getattr(config, attr) for attr in dir(config) 
                 if not attr.startswith('__') and not callable(getattr(config, attr))}
    
    config_path = os.path.join(result_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Create visualizations
    visualize_cv_results(fold_results, fold_metrics, fold_histories, result_dir)
    
    print(f"\nHybrid CV complete. Results saved to {result_dir}")
    
    return fold_models, fold_results, fold_metrics, fold_histories


# ======================================================================
# VISUALIZATION FUNCTIONS
# ======================================================================

def visualize_cv_results(fold_results, fold_metrics, fold_histories, output_dir):
    """Create visualizations for CV results."""
    print("\nCreating visualizations...")
    
    # Create combined predictions vs ground truth plot
    plt.figure(figsize=(10, 8))
    
    # Collect all predictions
    all_true = []
    all_pred = []
    all_sources = []
    all_valid = []
    
    for fold_idx, results_df in enumerate(fold_results):
        # Get values from dataframe
        y_true = results_df['y_true'].values
        y_pred = results_df['y_pred'].values
        sources = results_df['source'].values
        valid = results_df['valid'].values if 'valid' in results_df else np.ones_like(y_true, dtype=bool)
        
        all_true.extend(y_true[valid])
        all_pred.extend(y_pred[valid])
        all_sources.extend(sources[valid])
        all_valid.extend(valid)
    
    # Convert to arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_sources = np.array(all_sources)
    
    # Plot by source
    unique_sources = np.unique(all_sources)
    cmap = plt.cm.get_cmap('tab10', len(unique_sources))
    
    for i, source in enumerate(unique_sources):
        mask = (all_sources == source)
        plt.scatter(
            all_true[mask], all_pred[mask], 
            alpha=0.7, s=30, color=cmap(i),
            label=f"Site {source}"
        )
    
    # Add 1:1 line
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    spearman, _ = spearmanr(all_true, all_pred)
    
    # Add metrics text
    plt.text(0.05, 0.95, 
             f"RMSE = {rmse:.4f}\n$R^2$ = {r2:.4f}\nMAE = {mae:.4f}\nSpearman = {spearman:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and legend
    plt.xlabel('Observed Biomass')
    plt.ylabel('Predicted Biomass')
    plt.title('Hybrid CV: Predicted vs Observed Biomass')
    plt.legend(title="Site")
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_predictions.png"), dpi=300)
    plt.close()
    
    # Create metrics by fold plot
    plt.figure(figsize=(10, 6))
    
    fold_indices = np.arange(len(fold_metrics))
    rmse_values = [m['rmse'] for m in fold_metrics]
    r2_values = [m['r2'] for m in fold_metrics]
    
    # Create bar chart
    plt.subplot(1, 2, 1)
    plt.bar(fold_indices, rmse_values, alpha=0.7)
    plt.axhline(y=np.mean(rmse_values), color='red', linestyle='--', 
               label=f"Mean: {np.mean(rmse_values):.4f}")
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE by Fold')
    plt.xticks(fold_indices, [f"{i+1}" for i in fold_indices])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(fold_indices, r2_values, alpha=0.7)
    plt.axhline(y=np.mean(r2_values), color='red', linestyle='--',
               label=f"Mean: {np.mean(r2_values):.4f}")
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.title('R² by Fold')
    plt.xticks(fold_indices, [f"{i+1}" for i in fold_indices])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_metrics_by_fold.png"), dpi=300)
    plt.close()
    
    # Create training curves plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history['train_losses'], label=f"Fold {i+1} Train")
        plt.plot(history['val_losses'], label=f"Fold {i+1} Val", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss by Fold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for i, history in enumerate(fold_histories):
        if 'learning_rates' in history:
            plt.plot(history['learning_rates'], label=f"Fold {i+1}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_training_curves.png"), dpi=300)
    plt.close()
    
    # Create spatial error distribution plot
    plt.figure(figsize=(10, 8))
    
    # Collect all coordinates and residuals
    all_x = []
    all_y = []
    all_residuals = []
    
    for results_df in fold_results:
        valid = results_df['valid'].values if 'valid' in results_df else np.ones(len(results_df), dtype=bool)
        
        all_x.extend(results_df['x_coord'].values[valid])
        all_y.extend(results_df['y_coord'].values[valid])
        all_residuals.extend(results_df['residual'].values[valid])
    
    # Convert to arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_residuals = np.array(all_residuals)
    
    # Create scatter plot
    sc = plt.scatter(all_x, all_y, c=all_residuals, cmap='RdBu_r', 
                    s=30, alpha=0.7, edgecolors='k', linewidths=0.5,
                    vmin=-np.max(np.abs(all_residuals)), vmax=np.max(np.abs(all_residuals)))
    
    plt.colorbar(sc, label='Residual (Predicted - Observed)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Spatial Distribution of Prediction Errors')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_spatial_errors.png"), dpi=300)
    plt.close()


# ======================================================================
# MAIN FUNCTION
# ======================================================================

def main():
    """Main function to run hybrid site-spatial cross-validation."""
    print("=" * 80)
    print("Hybrid Site-Spatial Cross-Validation for Biomass Prediction")
    print("=" * 80)
    
    # Initialize config
    config = HybridCVConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    try:
        data = load_preprocessed_data(config)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run hybrid cross-validation
    fold_models, fold_results, fold_metrics, fold_histories = run_hybrid_cross_validation(data, config)
    
    print("\nHybrid cross-validation complete!")


if __name__ == "__main__":
    main()

"""
================================================================================
Hybrid Site-Spatial Cross-Validation for Biomass Prediction
================================================================================
Using device: cuda

==== Loading Preprocessed Data ====
Loaded data with timestamp: 20250528_014647
X shape: (784, 59, 24, 24)
y shape: (784,)
sources shape: (784,)
coordinates: 784 points

Site breakdown:
  Site_1 (ID: 0): 421 samples
  Site_2 (ID: 1): 132 samples
  Site_3 (ID: 2): 156 samples
  Site_4 (ID: 3): 63 samples
  Site_5 (ID: 4): 12 samples

================================================================================
HYBRID SITE-SPATIAL CROSS-VALIDATION
================================================================================

==== Creating Hybrid Site-Spatial CV Splits ====
Found 5 unique sites
  Site 0: Train=269, Val=67, Test=85
  Site 1: Train=88, Val=21, Test=23
  Site 2: Train=108, Val=27, Test=21
  Site 3: Train=32, Val=8, Test=23
  Site 4 has only 12 samples - adding all to training
Fold 1: Train=509, Val=123, Test=152
  Site 0: Train=272, Val=68, Test=81
  Site 1: Train=91, Val=22, Test=19
  Site 2: Train=107, Val=26, Test=23
  Site 3: Train=35, Val=8, Test=20
  Site 4 has only 12 samples - adding all to training
Fold 2: Train=517, Val=124, Test=143
  Site 0: Train=271, Val=67, Test=83
  Site 1: Train=88, Val=22, Test=22
  Site 2: Train=107, Val=26, Test=23
  Site 3: Train=33, Val=8, Test=22
  Site 4 has only 12 samples - adding all to training
Fold 3: Train=511, Val=123, Test=150
  Site 0: Train=271, Val=67, Test=83
  Site 1: Train=87, Val=21, Test=24
  Site 2: Train=108, Val=27, Test=21
  Site 3: Train=35, Val=8, Test=20
  Site 4 has only 12 samples - adding all to training
Fold 4: Train=513, Val=123, Test=148
  Site 0: Train=260, Val=65, Test=96
  Site 1: Train=89, Val=22, Test=21
  Site 2: Train=105, Val=26, Test=25
  Site 3: Train=33, Val=8, Test=22
  Site 4 has only 12 samples - adding all to training
Fold 5: Train=499, Val=121, Test=164

========================================
Running Hybrid CV Fold 1
========================================

Adding derived features...
Added 5 derived features, new shape: (509, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (123, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (152, 64, 24, 24)

Standardizing features...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4

Creating CNN_COORDINATE model...

Training model...
Epoch 1/100, Train Loss: 2.3801, Val Loss: 3.5416, LR: 0.001000
  → New best validation loss: 3.5416
Epoch 2/100, Train Loss: 1.1824, Val Loss: 0.7291, LR: 0.000999
  → New best validation loss: 0.7291
Epoch 3/100, Train Loss: 0.6662, Val Loss: 0.2132, LR: 0.000998
  → New best validation loss: 0.2132
Epoch 4/100, Train Loss: 0.6247, Val Loss: 0.2241, LR: 0.000996
Epoch 5/100, Train Loss: 0.6229, Val Loss: 0.2382, LR: 0.000994
Epoch 6/100, Train Loss: 0.6164, Val Loss: 0.2262, LR: 0.000992
Epoch 7/100, Train Loss: 0.6087, Val Loss: 0.2183, LR: 0.000989
Epoch 8/100, Train Loss: 0.5466, Val Loss: 0.2243, LR: 0.000986
Epoch 9/100, Train Loss: 0.6255, Val Loss: 0.2004, LR: 0.000982
  → New best validation loss: 0.2004
Epoch 10/100, Train Loss: 0.5616, Val Loss: 0.2399, LR: 0.000978
Epoch 11/100, Train Loss: 0.5385, Val Loss: 0.2021, LR: 0.000973
Epoch 12/100, Train Loss: 0.6135, Val Loss: 0.2311, LR: 0.000968
Epoch 13/100, Train Loss: 0.6051, Val Loss: 0.1925, LR: 0.000963
  → New best validation loss: 0.1925
Epoch 14/100, Train Loss: 0.5453, Val Loss: 0.1828, LR: 0.000957
  → New best validation loss: 0.1828
Epoch 15/100, Train Loss: 0.5644, Val Loss: 0.2064, LR: 0.000951
Epoch 16/100, Train Loss: 0.5552, Val Loss: 0.1517, LR: 0.000944
  → New best validation loss: 0.1517
Epoch 17/100, Train Loss: 0.5354, Val Loss: 0.1010, LR: 0.000937
  → New best validation loss: 0.1010
Epoch 18/100, Train Loss: 0.5481, Val Loss: 0.1539, LR: 0.000930
Epoch 19/100, Train Loss: 0.5222, Val Loss: 0.1079, LR: 0.000922
Epoch 20/100, Train Loss: 0.4881, Val Loss: 0.0950, LR: 0.000914
  → New best validation loss: 0.0950

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0840
    50th percentile: 0.1847
    75th percentile: 0.3106
    100th percentile: 2.7325
  - Using 1018 samples with hard negative mining
Epoch 21/100, Train Loss: 0.3653, Val Loss: 0.1258, LR: 0.000906

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0912
    50th percentile: 0.1910
    75th percentile: 0.3277
    100th percentile: 1.9382
  - Using 1018 samples with hard negative mining
Epoch 22/100, Train Loss: 0.1315, Val Loss: 0.0718, LR: 0.000897
  → New best validation loss: 0.0718

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0730
    50th percentile: 0.1583
    75th percentile: 0.2532
    100th percentile: 2.6662
  - Using 1018 samples with hard negative mining
Epoch 23/100, Train Loss: 0.1861, Val Loss: 0.0575, LR: 0.000888
  → New best validation loss: 0.0575

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0734
    50th percentile: 0.1589
    75th percentile: 0.2635
    100th percentile: 0.7403
  - Using 1018 samples with hard negative mining
Epoch 24/100, Train Loss: 0.0675, Val Loss: 0.0394, LR: 0.000878
  → New best validation loss: 0.0394

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0016
    25th percentile: 0.0594
    50th percentile: 0.1371
    75th percentile: 0.2119
    100th percentile: 0.6392
  - Using 1018 samples with hard negative mining
Epoch 25/100, Train Loss: 0.0431, Val Loss: 0.0381, LR: 0.000868
  → New best validation loss: 0.0381

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0505
    50th percentile: 0.1176
    75th percentile: 0.2041
    100th percentile: 0.6002
  - Using 1018 samples with hard negative mining
Epoch 26/100, Train Loss: 0.0339, Val Loss: 0.0291, LR: 0.000858
  → New best validation loss: 0.0291

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0533
    50th percentile: 0.1154
    75th percentile: 0.1894
    100th percentile: 0.4907
  - Using 1018 samples with hard negative mining
Epoch 27/100, Train Loss: 0.0275, Val Loss: 0.0328, LR: 0.000848

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0438
    50th percentile: 0.0980
    75th percentile: 0.1771
    100th percentile: 0.5431
  - Using 1018 samples with hard negative mining
Epoch 28/100, Train Loss: 0.0250, Val Loss: 0.0233, LR: 0.000837
  → New best validation loss: 0.0233

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0468
    50th percentile: 0.0933
    75th percentile: 0.1634
    100th percentile: 0.3784
  - Using 1018 samples with hard negative mining
Epoch 29/100, Train Loss: 0.0202, Val Loss: 0.0391, LR: 0.000826

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0484
    50th percentile: 0.1081
    75th percentile: 0.1942
    100th percentile: 0.5168
  - Using 1018 samples with hard negative mining
Epoch 30/100, Train Loss: 0.0179, Val Loss: 0.0243, LR: 0.000815

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0500
    50th percentile: 0.1107
    75th percentile: 0.1852
    100th percentile: 0.3738
  - Using 1018 samples with hard negative mining
Epoch 31/100, Train Loss: 0.0149, Val Loss: 0.0295, LR: 0.000803

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0006
    25th percentile: 0.0331
    50th percentile: 0.0808
    75th percentile: 0.1517
    100th percentile: 0.4204
  - Using 1018 samples with hard negative mining
Epoch 32/100, Train Loss: 0.0165, Val Loss: 0.0192, LR: 0.000791
  → New best validation loss: 0.0192

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0438
    50th percentile: 0.0826
    75th percentile: 0.1370
    100th percentile: 0.3270
  - Using 1018 samples with hard negative mining
Epoch 33/100, Train Loss: 0.0137, Val Loss: 0.0233, LR: 0.000779

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0352
    50th percentile: 0.0786
    75th percentile: 0.1375
    100th percentile: 0.3880
  - Using 1018 samples with hard negative mining
Epoch 34/100, Train Loss: 0.0124, Val Loss: 0.0172, LR: 0.000767
  → New best validation loss: 0.0172

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0333
    50th percentile: 0.0687
    75th percentile: 0.1099
    100th percentile: 0.2640
  - Using 1018 samples with hard negative mining
Epoch 35/100, Train Loss: 0.0110, Val Loss: 0.0203, LR: 0.000754

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0310
    50th percentile: 0.0685
    75th percentile: 0.1071
    100th percentile: 0.2539
  - Using 1018 samples with hard negative mining
Epoch 36/100, Train Loss: 0.0089, Val Loss: 0.0156, LR: 0.000742
  → New best validation loss: 0.0156

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0314
    50th percentile: 0.0610
    75th percentile: 0.1053
    100th percentile: 0.2488
  - Using 1018 samples with hard negative mining
Epoch 37/100, Train Loss: 0.0086, Val Loss: 0.0187, LR: 0.000729

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0005
    25th percentile: 0.0258
    50th percentile: 0.0606
    75th percentile: 0.1048
    100th percentile: 0.2617
  - Using 1018 samples with hard negative mining
Epoch 38/100, Train Loss: 0.0074, Val Loss: 0.0147, LR: 0.000716
  → New best validation loss: 0.0147

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0282
    50th percentile: 0.0554
    75th percentile: 0.0952
    100th percentile: 0.2313
  - Using 1018 samples with hard negative mining
Epoch 39/100, Train Loss: 0.0071, Val Loss: 0.0160, LR: 0.000702

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0265
    50th percentile: 0.0533
    75th percentile: 0.0880
    100th percentile: 0.2237
  - Using 1018 samples with hard negative mining
Epoch 40/100, Train Loss: 0.0081, Val Loss: 0.0175, LR: 0.000689

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0291
    50th percentile: 0.0572
    75th percentile: 0.0878
    100th percentile: 0.2046
  - Using 1018 samples with hard negative mining
Epoch 41/100, Train Loss: 0.0071, Val Loss: 0.0143, LR: 0.000676
  → New best validation loss: 0.0143

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0230
    50th percentile: 0.0498
    75th percentile: 0.0767
    100th percentile: 0.1924
  - Using 1018 samples with hard negative mining
Epoch 42/100, Train Loss: 0.0063, Val Loss: 0.0168, LR: 0.000662

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0225
    50th percentile: 0.0485
    75th percentile: 0.0830
    100th percentile: 0.3194
  - Using 1018 samples with hard negative mining
Epoch 43/100, Train Loss: 0.0063, Val Loss: 0.0135, LR: 0.000648
  → New best validation loss: 0.0135

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0239
    50th percentile: 0.0479
    75th percentile: 0.0709
    100th percentile: 0.1770
  - Using 1018 samples with hard negative mining
Epoch 44/100, Train Loss: 0.0051, Val Loss: 0.0161, LR: 0.000634

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0182
    50th percentile: 0.0409
    75th percentile: 0.0698
    100th percentile: 0.1788
  - Using 1018 samples with hard negative mining
Epoch 45/100, Train Loss: 0.0042, Val Loss: 0.0146, LR: 0.000620

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0179
    50th percentile: 0.0420
    75th percentile: 0.0666
    100th percentile: 0.1745
  - Using 1018 samples with hard negative mining
Epoch 46/100, Train Loss: 0.0042, Val Loss: 0.0160, LR: 0.000606

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0208
    50th percentile: 0.0399
    75th percentile: 0.0673
    100th percentile: 0.1586
  - Using 1018 samples with hard negative mining
Epoch 47/100, Train Loss: 0.0049, Val Loss: 0.0125, LR: 0.000592
  → New best validation loss: 0.0125

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0189
    50th percentile: 0.0393
    75th percentile: 0.0635
    100th percentile: 0.1874
  - Using 1018 samples with hard negative mining
Epoch 48/100, Train Loss: 0.0044, Val Loss: 0.0172, LR: 0.000578

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0172
    50th percentile: 0.0405
    75th percentile: 0.0742
    100th percentile: 0.1628
  - Using 1018 samples with hard negative mining
Epoch 49/100, Train Loss: 0.0034, Val Loss: 0.0127, LR: 0.000564

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0173
    50th percentile: 0.0347
    75th percentile: 0.0613
    100th percentile: 0.1740
  - Using 1018 samples with hard negative mining
Epoch 50/100, Train Loss: 0.0029, Val Loss: 0.0184, LR: 0.000550

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0170
    50th percentile: 0.0387
    75th percentile: 0.0687
    100th percentile: 0.1636
  - Using 1018 samples with hard negative mining
Epoch 51/100, Train Loss: 0.0030, Val Loss: 0.0135, LR: 0.000536

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0161
    50th percentile: 0.0323
    75th percentile: 0.0502
    100th percentile: 0.1193
  - Using 1018 samples with hard negative mining
Epoch 52/100, Train Loss: 0.0026, Val Loss: 0.0158, LR: 0.000522

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0143
    50th percentile: 0.0302
    75th percentile: 0.0502
    100th percentile: 0.1331
  - Using 1018 samples with hard negative mining
Epoch 53/100, Train Loss: 0.0023, Val Loss: 0.0133, LR: 0.000508

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0157
    50th percentile: 0.0321
    75th percentile: 0.0586
    100th percentile: 0.1273
  - Using 1018 samples with hard negative mining
Epoch 54/100, Train Loss: 0.0028, Val Loss: 0.0147, LR: 0.000494

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0130
    50th percentile: 0.0286
    75th percentile: 0.0501
    100th percentile: 0.1227
  - Using 1018 samples with hard negative mining
Epoch 55/100, Train Loss: 0.0023, Val Loss: 0.0141, LR: 0.000480

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0164
    50th percentile: 0.0340
    75th percentile: 0.0570
    100th percentile: 0.1182
  - Using 1018 samples with hard negative mining
Epoch 56/100, Train Loss: 0.0022, Val Loss: 0.0152, LR: 0.000466

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0126
    50th percentile: 0.0301
    75th percentile: 0.0520
    100th percentile: 0.1314
  - Using 1018 samples with hard negative mining
Epoch 57/100, Train Loss: 0.0021, Val Loss: 0.0131, LR: 0.000452

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0109
    50th percentile: 0.0244
    75th percentile: 0.0401
    100th percentile: 0.1089
  - Using 1018 samples with hard negative mining
Epoch 58/100, Train Loss: 0.0018, Val Loss: 0.0132, LR: 0.000438

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0113
    50th percentile: 0.0259
    75th percentile: 0.0442
    100th percentile: 0.0996
  - Using 1018 samples with hard negative mining
Epoch 59/100, Train Loss: 0.0016, Val Loss: 0.0147, LR: 0.000424

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0111
    50th percentile: 0.0241
    75th percentile: 0.0434
    100th percentile: 0.1095
  - Using 1018 samples with hard negative mining
Epoch 60/100, Train Loss: 0.0015, Val Loss: 0.0142, LR: 0.000411

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0125
    50th percentile: 0.0223
    75th percentile: 0.0397
    100th percentile: 0.1147
  - Using 1018 samples with hard negative mining
Epoch 61/100, Train Loss: 0.0014, Val Loss: 0.0141, LR: 0.000398

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0121
    50th percentile: 0.0229
    75th percentile: 0.0365
    100th percentile: 0.0895
  - Using 1018 samples with hard negative mining
Epoch 62/100, Train Loss: 0.0013, Val Loss: 0.0132, LR: 0.000384

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0100
    50th percentile: 0.0207
    75th percentile: 0.0359
    100th percentile: 0.0884
  - Using 1018 samples with hard negative mining
Epoch 63/100, Train Loss: 0.0015, Val Loss: 0.0138, LR: 0.000371

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0090
    50th percentile: 0.0183
    75th percentile: 0.0302
    100th percentile: 0.0797
  - Using 1018 samples with hard negative mining
Epoch 64/100, Train Loss: 0.0015, Val Loss: 0.0136, LR: 0.000358

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0079
    50th percentile: 0.0190
    75th percentile: 0.0336
    100th percentile: 0.0735
  - Using 1018 samples with hard negative mining
Epoch 65/100, Train Loss: 0.0012, Val Loss: 0.0142, LR: 0.000346

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0080
    50th percentile: 0.0167
    75th percentile: 0.0297
    100th percentile: 0.0727
  - Using 1018 samples with hard negative mining
Epoch 66/100, Train Loss: 0.0010, Val Loss: 0.0135, LR: 0.000333

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0083
    50th percentile: 0.0161
    75th percentile: 0.0267
    100th percentile: 0.0695
  - Using 1018 samples with hard negative mining
Epoch 67/100, Train Loss: 0.0010, Val Loss: 0.0139, LR: 0.000321
Early stopping at epoch 67

Evaluating model...

Applying test-time augmentation...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4
  - TTA iteration 1/3
  - TTA iteration 2/3
  - TTA iteration 3/3
  - Final predictions created from 4 versions

Test Metrics:
RMSE: 0.1030
R²: 0.9293
MAE: 0.0815
Spearman correlation: 0.9591

Metrics by site:
  Site 0: RMSE=0.1124, R²=0.8411, MAE=0.0892, n=85
  Site 1: RMSE=0.1007, R²=0.7780, MAE=0.0787, n=23
  Site 2: RMSE=0.0828, R²=0.2164, MAE=0.0677, n=21
  Site 3: RMSE=0.0837, R²=0.0319, MAE=0.0687, n=23

========================================
Running Hybrid CV Fold 2
========================================

Adding derived features...
Added 5 derived features, new shape: (517, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (124, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (143, 64, 24, 24)

Standardizing features...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4

Creating CNN_COORDINATE model...

Training model...
Epoch 1/100, Train Loss: 1.9545, Val Loss: 1.5660, LR: 0.001000
  → New best validation loss: 1.5660
Epoch 2/100, Train Loss: 0.7667, Val Loss: 0.2458, LR: 0.000999
  → New best validation loss: 0.2458
Epoch 3/100, Train Loss: 0.5548, Val Loss: 0.2026, LR: 0.000998
  → New best validation loss: 0.2026
Epoch 4/100, Train Loss: 0.4990, Val Loss: 0.2240, LR: 0.000996
Epoch 5/100, Train Loss: 0.5459, Val Loss: 0.2098, LR: 0.000994
Epoch 6/100, Train Loss: 0.5216, Val Loss: 0.2020, LR: 0.000992
  → New best validation loss: 0.2020
Epoch 7/100, Train Loss: 0.5406, Val Loss: 0.2049, LR: 0.000989
Epoch 8/100, Train Loss: 0.5181, Val Loss: 0.2100, LR: 0.000986
Epoch 9/100, Train Loss: 0.5188, Val Loss: 0.1998, LR: 0.000982
  → New best validation loss: 0.1998
Epoch 10/100, Train Loss: 0.4694, Val Loss: 0.2011, LR: 0.000978
Epoch 11/100, Train Loss: 0.5019, Val Loss: 0.2069, LR: 0.000973
Epoch 12/100, Train Loss: 0.5036, Val Loss: 0.1981, LR: 0.000968
  → New best validation loss: 0.1981
Epoch 13/100, Train Loss: 0.4946, Val Loss: 0.2027, LR: 0.000963
Epoch 14/100, Train Loss: 0.4926, Val Loss: 0.1958, LR: 0.000957
  → New best validation loss: 0.1958
Epoch 15/100, Train Loss: 0.4856, Val Loss: 0.1888, LR: 0.000951
  → New best validation loss: 0.1888
Epoch 16/100, Train Loss: 0.4958, Val Loss: 0.2135, LR: 0.000944
Epoch 17/100, Train Loss: 0.5039, Val Loss: 0.1632, LR: 0.000937
  → New best validation loss: 0.1632
Epoch 18/100, Train Loss: 0.4531, Val Loss: 0.1487, LR: 0.000930
  → New best validation loss: 0.1487
Epoch 19/100, Train Loss: 0.4975, Val Loss: 0.1145, LR: 0.000922
  → New best validation loss: 0.1145
Epoch 20/100, Train Loss: 0.4348, Val Loss: 0.0912, LR: 0.000914
  → New best validation loss: 0.0912

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0020
    25th percentile: 0.0950
    50th percentile: 0.1952
    75th percentile: 0.3094
    100th percentile: 2.7137
  - Using 1034 samples with hard negative mining
Epoch 21/100, Train Loss: 0.2833, Val Loss: 0.1320, LR: 0.000906

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0009
    25th percentile: 0.1283
    50th percentile: 0.2482
    75th percentile: 0.3772
    100th percentile: 1.1440
  - Using 1034 samples with hard negative mining
Epoch 22/100, Train Loss: 0.0700, Val Loss: 0.0539, LR: 0.000897
  → New best validation loss: 0.0539

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0541
    50th percentile: 0.1137
    75th percentile: 0.2137
    100th percentile: 1.6726
  - Using 1034 samples with hard negative mining
Epoch 23/100, Train Loss: 0.0942, Val Loss: 0.0625, LR: 0.000888

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0008
    25th percentile: 0.0852
    50th percentile: 0.1707
    75th percentile: 0.2868
    100th percentile: 0.8229
  - Using 1034 samples with hard negative mining
Epoch 24/100, Train Loss: 0.0374, Val Loss: 0.0231, LR: 0.000878
  → New best validation loss: 0.0231

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0462
    50th percentile: 0.0928
    75th percentile: 0.1600
    100th percentile: 0.5242
  - Using 1034 samples with hard negative mining
Epoch 25/100, Train Loss: 0.0372, Val Loss: 0.0280, LR: 0.000868

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0503
    50th percentile: 0.1044
    75th percentile: 0.1672
    100th percentile: 0.4271
  - Using 1034 samples with hard negative mining
Epoch 26/100, Train Loss: 0.0256, Val Loss: 0.0182, LR: 0.000858
  → New best validation loss: 0.0182

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0337
    50th percentile: 0.0828
    75th percentile: 0.1462
    100th percentile: 0.4322
  - Using 1034 samples with hard negative mining
Epoch 27/100, Train Loss: 0.0226, Val Loss: 0.0321, LR: 0.000848

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0529
    50th percentile: 0.1095
    75th percentile: 0.1802
    100th percentile: 0.4203
  - Using 1034 samples with hard negative mining
Epoch 28/100, Train Loss: 0.0149, Val Loss: 0.0168, LR: 0.000837
  → New best validation loss: 0.0168

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0359
    50th percentile: 0.0771
    75th percentile: 0.1267
    100th percentile: 0.3714
  - Using 1034 samples with hard negative mining
Epoch 29/100, Train Loss: 0.0179, Val Loss: 0.0189, LR: 0.000826

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0006
    25th percentile: 0.0362
    50th percentile: 0.0725
    75th percentile: 0.1198
    100th percentile: 0.3042
  - Using 1034 samples with hard negative mining
Epoch 30/100, Train Loss: 0.0153, Val Loss: 0.0174, LR: 0.000815

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0374
    50th percentile: 0.0757
    75th percentile: 0.1295
    100th percentile: 0.3186
  - Using 1034 samples with hard negative mining
Epoch 31/100, Train Loss: 0.0117, Val Loss: 0.0183, LR: 0.000803

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0297
    50th percentile: 0.0594
    75th percentile: 0.1115
    100th percentile: 0.2999
  - Using 1034 samples with hard negative mining
Epoch 32/100, Train Loss: 0.0097, Val Loss: 0.0167, LR: 0.000791
  → New best validation loss: 0.0167

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0330
    50th percentile: 0.0664
    75th percentile: 0.0998
    100th percentile: 0.2748
  - Using 1034 samples with hard negative mining
Epoch 33/100, Train Loss: 0.0093, Val Loss: 0.0163, LR: 0.000779
  → New best validation loss: 0.0163

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0266
    50th percentile: 0.0516
    75th percentile: 0.0861
    100th percentile: 0.2364
  - Using 1034 samples with hard negative mining
Epoch 34/100, Train Loss: 0.0079, Val Loss: 0.0204, LR: 0.000767

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0377
    50th percentile: 0.0722
    75th percentile: 0.1150
    100th percentile: 0.2906
  - Using 1034 samples with hard negative mining
Epoch 35/100, Train Loss: 0.0070, Val Loss: 0.0183, LR: 0.000754

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0217
    50th percentile: 0.0530
    75th percentile: 0.0986
    100th percentile: 0.2391
  - Using 1034 samples with hard negative mining
Epoch 36/100, Train Loss: 0.0061, Val Loss: 0.0160, LR: 0.000742
  → New best validation loss: 0.0160

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0279
    50th percentile: 0.0529
    75th percentile: 0.0838
    100th percentile: 0.1952
  - Using 1034 samples with hard negative mining
Epoch 37/100, Train Loss: 0.0060, Val Loss: 0.0144, LR: 0.000729
  → New best validation loss: 0.0144

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0215
    50th percentile: 0.0484
    75th percentile: 0.0794
    100th percentile: 0.2470
  - Using 1034 samples with hard negative mining
Epoch 38/100, Train Loss: 0.0053, Val Loss: 0.0183, LR: 0.000716

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0271
    50th percentile: 0.0542
    75th percentile: 0.0964
    100th percentile: 0.2128
  - Using 1034 samples with hard negative mining
Epoch 39/100, Train Loss: 0.0052, Val Loss: 0.0175, LR: 0.000702

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0237
    50th percentile: 0.0534
    75th percentile: 0.0978
    100th percentile: 0.2107
  - Using 1034 samples with hard negative mining
Epoch 40/100, Train Loss: 0.0052, Val Loss: 0.0169, LR: 0.000689

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0267
    50th percentile: 0.0618
    75th percentile: 0.0990
    100th percentile: 0.2084
  - Using 1034 samples with hard negative mining
Epoch 41/100, Train Loss: 0.0048, Val Loss: 0.0154, LR: 0.000676

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0162
    50th percentile: 0.0394
    75th percentile: 0.0639
    100th percentile: 0.2074
  - Using 1034 samples with hard negative mining
Epoch 42/100, Train Loss: 0.0042, Val Loss: 0.0136, LR: 0.000662
  → New best validation loss: 0.0136

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0174
    50th percentile: 0.0389
    75th percentile: 0.0603
    100th percentile: 0.1801
  - Using 1034 samples with hard negative mining
Epoch 43/100, Train Loss: 0.0041, Val Loss: 0.0144, LR: 0.000648

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0183
    50th percentile: 0.0389
    75th percentile: 0.0617
    100th percentile: 0.1395
  - Using 1034 samples with hard negative mining
Epoch 44/100, Train Loss: 0.0034, Val Loss: 0.0141, LR: 0.000634

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0165
    50th percentile: 0.0338
    75th percentile: 0.0578
    100th percentile: 0.1974
  - Using 1034 samples with hard negative mining
Epoch 45/100, Train Loss: 0.0044, Val Loss: 0.0145, LR: 0.000620

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0202
    50th percentile: 0.0390
    75th percentile: 0.0604
    100th percentile: 0.1359
  - Using 1034 samples with hard negative mining
Epoch 46/100, Train Loss: 0.0028, Val Loss: 0.0130, LR: 0.000606
  → New best validation loss: 0.0130

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0173
    50th percentile: 0.0320
    75th percentile: 0.0544
    100th percentile: 0.1148
  - Using 1034 samples with hard negative mining
Epoch 47/100, Train Loss: 0.0030, Val Loss: 0.0139, LR: 0.000592

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0141
    50th percentile: 0.0293
    75th percentile: 0.0478
    100th percentile: 0.1343
  - Using 1034 samples with hard negative mining
Epoch 48/100, Train Loss: 0.0032, Val Loss: 0.0133, LR: 0.000578

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0136
    50th percentile: 0.0279
    75th percentile: 0.0431
    100th percentile: 0.2568
  - Using 1034 samples with hard negative mining
Epoch 49/100, Train Loss: 0.0023, Val Loss: 0.0145, LR: 0.000564

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0156
    50th percentile: 0.0322
    75th percentile: 0.0543
    100th percentile: 0.1657
  - Using 1034 samples with hard negative mining
Epoch 50/100, Train Loss: 0.0021, Val Loss: 0.0148, LR: 0.000550

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0146
    50th percentile: 0.0280
    75th percentile: 0.0452
    100th percentile: 0.1359
  - Using 1034 samples with hard negative mining
Epoch 51/100, Train Loss: 0.0022, Val Loss: 0.0139, LR: 0.000536

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0119
    50th percentile: 0.0286
    75th percentile: 0.0470
    100th percentile: 0.1333
  - Using 1034 samples with hard negative mining
Epoch 52/100, Train Loss: 0.0021, Val Loss: 0.0147, LR: 0.000522

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0116
    50th percentile: 0.0242
    75th percentile: 0.0420
    100th percentile: 0.1001
  - Using 1034 samples with hard negative mining
Epoch 53/100, Train Loss: 0.0021, Val Loss: 0.0147, LR: 0.000508

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0157
    50th percentile: 0.0340
    75th percentile: 0.0569
    100th percentile: 0.2146
  - Using 1034 samples with hard negative mining
Epoch 54/100, Train Loss: 0.0021, Val Loss: 0.0149, LR: 0.000494

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0112
    50th percentile: 0.0236
    75th percentile: 0.0400
    100th percentile: 0.0934
  - Using 1034 samples with hard negative mining
Epoch 55/100, Train Loss: 0.0017, Val Loss: 0.0139, LR: 0.000480

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0099
    50th percentile: 0.0217
    75th percentile: 0.0395
    100th percentile: 0.0904
  - Using 1034 samples with hard negative mining
Epoch 56/100, Train Loss: 0.0015, Val Loss: 0.0157, LR: 0.000466

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0095
    50th percentile: 0.0221
    75th percentile: 0.0400
    100th percentile: 0.1042
  - Using 1034 samples with hard negative mining
Epoch 57/100, Train Loss: 0.0013, Val Loss: 0.0152, LR: 0.000452

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0084
    50th percentile: 0.0190
    75th percentile: 0.0333
    100th percentile: 0.1350
  - Using 1034 samples with hard negative mining
Epoch 58/100, Train Loss: 0.0015, Val Loss: 0.0152, LR: 0.000438

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0112
    50th percentile: 0.0247
    75th percentile: 0.0414
    100th percentile: 0.0913
  - Using 1034 samples with hard negative mining
Epoch 59/100, Train Loss: 0.0012, Val Loss: 0.0139, LR: 0.000424

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0086
    50th percentile: 0.0187
    75th percentile: 0.0298
    100th percentile: 0.0679
  - Using 1034 samples with hard negative mining
Epoch 60/100, Train Loss: 0.0012, Val Loss: 0.0138, LR: 0.000411

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0087
    50th percentile: 0.0187
    75th percentile: 0.0324
    100th percentile: 0.0799
  - Using 1034 samples with hard negative mining
Epoch 61/100, Train Loss: 0.0011, Val Loss: 0.0139, LR: 0.000398

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0097
    50th percentile: 0.0204
    75th percentile: 0.0344
    100th percentile: 0.0928
  - Using 1034 samples with hard negative mining
Epoch 62/100, Train Loss: 0.0011, Val Loss: 0.0144, LR: 0.000384

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0088
    50th percentile: 0.0177
    75th percentile: 0.0324
    100th percentile: 0.0709
  - Using 1034 samples with hard negative mining
Epoch 63/100, Train Loss: 0.0010, Val Loss: 0.0150, LR: 0.000371

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0132
    50th percentile: 0.0272
    75th percentile: 0.0426
    100th percentile: 0.0759
  - Using 1034 samples with hard negative mining
Epoch 64/100, Train Loss: 0.0011, Val Loss: 0.0140, LR: 0.000358

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0078
    50th percentile: 0.0164
    75th percentile: 0.0274
    100th percentile: 0.0890
  - Using 1034 samples with hard negative mining
Epoch 65/100, Train Loss: 0.0008, Val Loss: 0.0144, LR: 0.000346

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0064
    50th percentile: 0.0140
    75th percentile: 0.0256
    100th percentile: 0.0595
  - Using 1034 samples with hard negative mining
Epoch 66/100, Train Loss: 0.0008, Val Loss: 0.0137, LR: 0.000333
Early stopping at epoch 66

Evaluating model...

Applying test-time augmentation...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4
  - TTA iteration 1/3
  - TTA iteration 2/3
  - TTA iteration 3/3
  - Final predictions created from 4 versions

Test Metrics:
RMSE: 0.1174
R²: 0.9285
MAE: 0.0877
Spearman correlation: 0.9455

Metrics by site:
  Site 0: RMSE=0.1242, R²=0.7775, MAE=0.0917, n=81
  Site 1: RMSE=0.0597, R²=0.9266, MAE=0.0528, n=19
  Site 2: RMSE=0.1436, R²=0.8946, MAE=0.1052, n=23
  Site 3: RMSE=0.0953, R²=0.8827, MAE=0.0846, n=20

========================================
Running Hybrid CV Fold 3
========================================

Adding derived features...
Added 5 derived features, new shape: (511, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (123, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (150, 64, 24, 24)

Standardizing features...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4

Creating CNN_COORDINATE model...

Training model...
Epoch 1/100, Train Loss: 2.1239, Val Loss: 2.0793, LR: 0.001000
  → New best validation loss: 2.0793
Epoch 2/100, Train Loss: 0.7908, Val Loss: 0.3695, LR: 0.000999
  → New best validation loss: 0.3695
Epoch 3/100, Train Loss: 0.5588, Val Loss: 0.2879, LR: 0.000998
  → New best validation loss: 0.2879
Epoch 4/100, Train Loss: 0.5682, Val Loss: 0.2777, LR: 0.000996
  → New best validation loss: 0.2777
Epoch 5/100, Train Loss: 0.5603, Val Loss: 0.2807, LR: 0.000994
Epoch 6/100, Train Loss: 0.5325, Val Loss: 0.2841, LR: 0.000992
Epoch 7/100, Train Loss: 0.5540, Val Loss: 0.2761, LR: 0.000989
  → New best validation loss: 0.2761
Epoch 8/100, Train Loss: 0.5098, Val Loss: 0.2901, LR: 0.000986
Epoch 9/100, Train Loss: 0.5204, Val Loss: 0.2773, LR: 0.000982
Epoch 10/100, Train Loss: 0.5077, Val Loss: 0.2796, LR: 0.000978
Epoch 11/100, Train Loss: 0.4874, Val Loss: 0.2776, LR: 0.000973
Epoch 12/100, Train Loss: 0.4827, Val Loss: 0.2806, LR: 0.000968
Epoch 13/100, Train Loss: 0.4594, Val Loss: 0.2752, LR: 0.000963
  → New best validation loss: 0.2752
Epoch 14/100, Train Loss: 0.4800, Val Loss: 0.2835, LR: 0.000957
Epoch 15/100, Train Loss: 0.5539, Val Loss: 0.2925, LR: 0.000951
Epoch 16/100, Train Loss: 0.4936, Val Loss: 0.2880, LR: 0.000944
Epoch 17/100, Train Loss: 0.4798, Val Loss: 0.2731, LR: 0.000937
  → New best validation loss: 0.2731
Epoch 18/100, Train Loss: 0.4705, Val Loss: 0.2809, LR: 0.000930
Epoch 19/100, Train Loss: 0.4943, Val Loss: 0.2868, LR: 0.000922
Epoch 20/100, Train Loss: 0.4965, Val Loss: 0.2759, LR: 0.000914

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0027
    25th percentile: 0.1519
    50th percentile: 0.3359
    75th percentile: 0.4905
    100th percentile: 3.2308
  - Using 1022 samples with hard negative mining
Epoch 21/100, Train Loss: 0.6323, Val Loss: 0.1845, LR: 0.000906
  → New best validation loss: 0.1845

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0014
    25th percentile: 0.1039
    50th percentile: 0.2267
    75th percentile: 0.3944
    100th percentile: 1.5552
  - Using 1022 samples with hard negative mining
Epoch 22/100, Train Loss: 0.1726, Val Loss: 0.1240, LR: 0.000897
  → New best validation loss: 0.1240

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0713
    50th percentile: 0.1526
    75th percentile: 0.2557
    100th percentile: 2.7354
  - Using 1022 samples with hard negative mining
Epoch 23/100, Train Loss: 0.2719, Val Loss: 0.0692, LR: 0.000888
  → New best validation loss: 0.0692

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0746
    50th percentile: 0.1557
    75th percentile: 0.2958
    100th percentile: 1.8693
  - Using 1022 samples with hard negative mining
Epoch 24/100, Train Loss: 0.1006, Val Loss: 0.0771, LR: 0.000878

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0712
    50th percentile: 0.1504
    75th percentile: 0.2698
    100th percentile: 0.7387
  - Using 1022 samples with hard negative mining
Epoch 25/100, Train Loss: 0.0416, Val Loss: 0.0281, LR: 0.000868
  → New best validation loss: 0.0281

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0540
    50th percentile: 0.1082
    75th percentile: 0.1900
    100th percentile: 0.5600
  - Using 1022 samples with hard negative mining
Epoch 26/100, Train Loss: 0.0380, Val Loss: 0.0378, LR: 0.000858

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0470
    50th percentile: 0.1028
    75th percentile: 0.1777
    100th percentile: 0.5872
  - Using 1022 samples with hard negative mining
Epoch 27/100, Train Loss: 0.0272, Val Loss: 0.0257, LR: 0.000848
  → New best validation loss: 0.0257

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0449
    50th percentile: 0.0945
    75th percentile: 0.1527
    100th percentile: 0.3794
  - Using 1022 samples with hard negative mining
Epoch 28/100, Train Loss: 0.0236, Val Loss: 0.0203, LR: 0.000837
  → New best validation loss: 0.0203

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0363
    50th percentile: 0.0798
    75th percentile: 0.1351
    100th percentile: 0.3334
  - Using 1022 samples with hard negative mining
Epoch 29/100, Train Loss: 0.0175, Val Loss: 0.0340, LR: 0.000826

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0006
    25th percentile: 0.0497
    50th percentile: 0.1000
    75th percentile: 0.1671
    100th percentile: 0.4695
  - Using 1022 samples with hard negative mining
Epoch 30/100, Train Loss: 0.0128, Val Loss: 0.0184, LR: 0.000815
  → New best validation loss: 0.0184

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0005
    25th percentile: 0.0319
    50th percentile: 0.0653
    75th percentile: 0.1051
    100th percentile: 0.3282
  - Using 1022 samples with hard negative mining
Epoch 31/100, Train Loss: 0.0129, Val Loss: 0.0237, LR: 0.000803

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0359
    50th percentile: 0.0701
    75th percentile: 0.1125
    100th percentile: 0.2640
  - Using 1022 samples with hard negative mining
Epoch 32/100, Train Loss: 0.0116, Val Loss: 0.0189, LR: 0.000791

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0294
    50th percentile: 0.0622
    75th percentile: 0.1020
    100th percentile: 0.2581
  - Using 1022 samples with hard negative mining
Epoch 33/100, Train Loss: 0.0108, Val Loss: 0.0161, LR: 0.000779
  → New best validation loss: 0.0161

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0006
    25th percentile: 0.0303
    50th percentile: 0.0603
    75th percentile: 0.1001
    100th percentile: 0.2379
  - Using 1022 samples with hard negative mining
Epoch 34/100, Train Loss: 0.0087, Val Loss: 0.0169, LR: 0.000767

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0006
    25th percentile: 0.0301
    50th percentile: 0.0540
    75th percentile: 0.0893
    100th percentile: 0.2713
  - Using 1022 samples with hard negative mining
Epoch 35/100, Train Loss: 0.0090, Val Loss: 0.0191, LR: 0.000754

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0005
    25th percentile: 0.0280
    50th percentile: 0.0535
    75th percentile: 0.0838
    100th percentile: 0.2325
  - Using 1022 samples with hard negative mining
Epoch 36/100, Train Loss: 0.0075, Val Loss: 0.0231, LR: 0.000742

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0308
    50th percentile: 0.0584
    75th percentile: 0.1059
    100th percentile: 0.2859
  - Using 1022 samples with hard negative mining
Epoch 37/100, Train Loss: 0.0063, Val Loss: 0.0163, LR: 0.000729

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0252
    50th percentile: 0.0479
    75th percentile: 0.0786
    100th percentile: 0.1818
  - Using 1022 samples with hard negative mining
Epoch 38/100, Train Loss: 0.0082, Val Loss: 0.0161, LR: 0.000716

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0227
    50th percentile: 0.0450
    75th percentile: 0.0787
    100th percentile: 0.1857
  - Using 1022 samples with hard negative mining
Epoch 39/100, Train Loss: 0.0057, Val Loss: 0.0178, LR: 0.000702

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0238
    50th percentile: 0.0471
    75th percentile: 0.0789
    100th percentile: 0.1774
  - Using 1022 samples with hard negative mining
Epoch 40/100, Train Loss: 0.0048, Val Loss: 0.0180, LR: 0.000689

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0218
    50th percentile: 0.0431
    75th percentile: 0.0703
    100th percentile: 0.2026
  - Using 1022 samples with hard negative mining
Epoch 41/100, Train Loss: 0.0045, Val Loss: 0.0174, LR: 0.000676

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0206
    50th percentile: 0.0456
    75th percentile: 0.0696
    100th percentile: 0.1644
  - Using 1022 samples with hard negative mining
Epoch 42/100, Train Loss: 0.0042, Val Loss: 0.0174, LR: 0.000662

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0166
    50th percentile: 0.0378
    75th percentile: 0.0646
    100th percentile: 0.1520
  - Using 1022 samples with hard negative mining
Epoch 43/100, Train Loss: 0.0040, Val Loss: 0.0172, LR: 0.000648

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0163
    50th percentile: 0.0377
    75th percentile: 0.0645
    100th percentile: 0.1365
  - Using 1022 samples with hard negative mining
Epoch 44/100, Train Loss: 0.0038, Val Loss: 0.0182, LR: 0.000634

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0173
    50th percentile: 0.0393
    75th percentile: 0.0659
    100th percentile: 0.2144
  - Using 1022 samples with hard negative mining
Epoch 45/100, Train Loss: 0.0038, Val Loss: 0.0200, LR: 0.000620

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0179
    50th percentile: 0.0399
    75th percentile: 0.0659
    100th percentile: 0.1514
  - Using 1022 samples with hard negative mining
Epoch 46/100, Train Loss: 0.0030, Val Loss: 0.0155, LR: 0.000606
  → New best validation loss: 0.0155

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0136
    50th percentile: 0.0322
    75th percentile: 0.0577
    100th percentile: 0.1453
  - Using 1022 samples with hard negative mining
Epoch 47/100, Train Loss: 0.0030, Val Loss: 0.0208, LR: 0.000592

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0241
    50th percentile: 0.0522
    75th percentile: 0.0849
    100th percentile: 0.1943
  - Using 1022 samples with hard negative mining
Epoch 48/100, Train Loss: 0.0025, Val Loss: 0.0141, LR: 0.000578
  → New best validation loss: 0.0141

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0157
    50th percentile: 0.0343
    75th percentile: 0.0657
    100th percentile: 0.1611
  - Using 1022 samples with hard negative mining
Epoch 49/100, Train Loss: 0.0026, Val Loss: 0.0168, LR: 0.000564

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0153
    50th percentile: 0.0330
    75th percentile: 0.0571
    100th percentile: 0.1226
  - Using 1022 samples with hard negative mining
Epoch 50/100, Train Loss: 0.0025, Val Loss: 0.0162, LR: 0.000550

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0137
    50th percentile: 0.0298
    75th percentile: 0.0507
    100th percentile: 0.1504
  - Using 1022 samples with hard negative mining
Epoch 51/100, Train Loss: 0.0024, Val Loss: 0.0154, LR: 0.000536

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0123
    50th percentile: 0.0300
    75th percentile: 0.0493
    100th percentile: 0.1300
  - Using 1022 samples with hard negative mining
Epoch 52/100, Train Loss: 0.0024, Val Loss: 0.0147, LR: 0.000522

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0121
    50th percentile: 0.0260
    75th percentile: 0.0464
    100th percentile: 0.1225
  - Using 1022 samples with hard negative mining
Epoch 53/100, Train Loss: 0.0021, Val Loss: 0.0160, LR: 0.000508

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0117
    50th percentile: 0.0266
    75th percentile: 0.0434
    100th percentile: 0.1012
  - Using 1022 samples with hard negative mining
Epoch 54/100, Train Loss: 0.0016, Val Loss: 0.0156, LR: 0.000494

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0135
    50th percentile: 0.0274
    75th percentile: 0.0497
    100th percentile: 0.1328
  - Using 1022 samples with hard negative mining
Epoch 55/100, Train Loss: 0.0019, Val Loss: 0.0158, LR: 0.000480

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0112
    50th percentile: 0.0242
    75th percentile: 0.0426
    100th percentile: 0.1098
  - Using 1022 samples with hard negative mining
Epoch 56/100, Train Loss: 0.0017, Val Loss: 0.0149, LR: 0.000466

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0126
    50th percentile: 0.0279
    75th percentile: 0.0463
    100th percentile: 0.1219
  - Using 1022 samples with hard negative mining
Epoch 57/100, Train Loss: 0.0018, Val Loss: 0.0167, LR: 0.000452

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0140
    50th percentile: 0.0304
    75th percentile: 0.0507
    100th percentile: 0.1196
  - Using 1022 samples with hard negative mining
Epoch 58/100, Train Loss: 0.0017, Val Loss: 0.0157, LR: 0.000438

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0102
    50th percentile: 0.0206
    75th percentile: 0.0362
    100th percentile: 0.1206
  - Using 1022 samples with hard negative mining
Epoch 59/100, Train Loss: 0.0015, Val Loss: 0.0162, LR: 0.000424

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0178
    50th percentile: 0.0313
    75th percentile: 0.0472
    100th percentile: 0.1136
  - Using 1022 samples with hard negative mining
Epoch 60/100, Train Loss: 0.0018, Val Loss: 0.0155, LR: 0.000411

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0112
    50th percentile: 0.0222
    75th percentile: 0.0371
    100th percentile: 0.0972
  - Using 1022 samples with hard negative mining
Epoch 61/100, Train Loss: 0.0014, Val Loss: 0.0155, LR: 0.000398

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0087
    50th percentile: 0.0183
    75th percentile: 0.0341
    100th percentile: 0.0863
  - Using 1022 samples with hard negative mining
Epoch 62/100, Train Loss: 0.0012, Val Loss: 0.0168, LR: 0.000384

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0082
    50th percentile: 0.0170
    75th percentile: 0.0299
    100th percentile: 0.1110
  - Using 1022 samples with hard negative mining
Epoch 63/100, Train Loss: 0.0012, Val Loss: 0.0167, LR: 0.000371

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0085
    50th percentile: 0.0173
    75th percentile: 0.0292
    100th percentile: 0.0763
  - Using 1022 samples with hard negative mining
Epoch 64/100, Train Loss: 0.0011, Val Loss: 0.0169, LR: 0.000358

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0120
    50th percentile: 0.0234
    75th percentile: 0.0389
    100th percentile: 0.0967
  - Using 1022 samples with hard negative mining
Epoch 65/100, Train Loss: 0.0010, Val Loss: 0.0153, LR: 0.000346

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0083
    50th percentile: 0.0165
    75th percentile: 0.0280
    100th percentile: 0.0753
  - Using 1022 samples with hard negative mining
Epoch 66/100, Train Loss: 0.0009, Val Loss: 0.0166, LR: 0.000333

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0090
    50th percentile: 0.0202
    75th percentile: 0.0359
    100th percentile: 0.0771
  - Using 1022 samples with hard negative mining
Epoch 67/100, Train Loss: 0.0009, Val Loss: 0.0150, LR: 0.000321

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0078
    50th percentile: 0.0148
    75th percentile: 0.0258
    100th percentile: 0.0633
  - Using 1022 samples with hard negative mining
Epoch 68/100, Train Loss: 0.0007, Val Loss: 0.0165, LR: 0.000309
Early stopping at epoch 68

Evaluating model...

Applying test-time augmentation...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4
  - TTA iteration 1/3
  - TTA iteration 2/3
  - TTA iteration 3/3
  - Final predictions created from 4 versions

Test Metrics:
RMSE: 0.1583
R²: 0.8687
MAE: 0.1182
Spearman correlation: 0.9315

Metrics by site:
  Site 0: RMSE=0.1869, R²=0.7418, MAE=0.1403, n=83
  Site 1: RMSE=0.1102, R²=0.6999, MAE=0.0838, n=22
  Site 2: RMSE=0.1051, R²=0.8334, MAE=0.0805, n=23
  Site 3: RMSE=0.1237, R²=-0.8028, MAE=0.1083, n=22

========================================
Running Hybrid CV Fold 4
========================================

Adding derived features...
Added 5 derived features, new shape: (513, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (123, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (148, 64, 24, 24)

Standardizing features...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4

Creating CNN_COORDINATE model...

Training model...
Epoch 1/100, Train Loss: 2.5060, Val Loss: 4.0138, LR: 0.001000
  → New best validation loss: 4.0138
Epoch 2/100, Train Loss: 1.1534, Val Loss: 0.8094, LR: 0.000999
  → New best validation loss: 0.8094
Epoch 3/100, Train Loss: 0.6812, Val Loss: 0.2787, LR: 0.000998
  → New best validation loss: 0.2787
Epoch 4/100, Train Loss: 0.6034, Val Loss: 0.2507, LR: 0.000996
  → New best validation loss: 0.2507
Epoch 5/100, Train Loss: 0.6119, Val Loss: 0.2402, LR: 0.000994
  → New best validation loss: 0.2402
Epoch 6/100, Train Loss: 0.5851, Val Loss: 0.2172, LR: 0.000992
  → New best validation loss: 0.2172
Epoch 7/100, Train Loss: 0.6073, Val Loss: 0.2353, LR: 0.000989
Epoch 8/100, Train Loss: 0.5912, Val Loss: 0.2285, LR: 0.000986
Epoch 9/100, Train Loss: 0.5491, Val Loss: 0.2149, LR: 0.000982
  → New best validation loss: 0.2149
Epoch 10/100, Train Loss: 0.5229, Val Loss: 0.2422, LR: 0.000978
Epoch 11/100, Train Loss: 0.5734, Val Loss: 0.2157, LR: 0.000973
Epoch 12/100, Train Loss: 0.5560, Val Loss: 0.2363, LR: 0.000968
Epoch 13/100, Train Loss: 0.5473, Val Loss: 0.2152, LR: 0.000963
Epoch 14/100, Train Loss: 0.5387, Val Loss: 0.2164, LR: 0.000957
Epoch 15/100, Train Loss: 0.5224, Val Loss: 0.2345, LR: 0.000951
Epoch 16/100, Train Loss: 0.5323, Val Loss: 0.2180, LR: 0.000944
Epoch 17/100, Train Loss: 0.5307, Val Loss: 0.2027, LR: 0.000937
  → New best validation loss: 0.2027
Epoch 18/100, Train Loss: 0.5321, Val Loss: 0.1916, LR: 0.000930
  → New best validation loss: 0.1916
Epoch 19/100, Train Loss: 0.4997, Val Loss: 0.2150, LR: 0.000922
Epoch 20/100, Train Loss: 0.5083, Val Loss: 0.1435, LR: 0.000914
  → New best validation loss: 0.1435

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0005
    25th percentile: 0.0854
    50th percentile: 0.2311
    75th percentile: 0.4499
    100th percentile: 2.7665
  - Using 1026 samples with hard negative mining
Epoch 21/100, Train Loss: 0.2801, Val Loss: 0.2443, LR: 0.000906

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0908
    50th percentile: 0.2016
    75th percentile: 0.3894
    100th percentile: 1.8316
  - Using 1026 samples with hard negative mining
Epoch 22/100, Train Loss: 0.1622, Val Loss: 0.0839, LR: 0.000897
  → New best validation loss: 0.0839

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0022
    25th percentile: 0.0740
    50th percentile: 0.1561
    75th percentile: 0.2642
    100th percentile: 2.3530
  - Using 1026 samples with hard negative mining
Epoch 23/100, Train Loss: 0.1437, Val Loss: 0.0602, LR: 0.000888
  → New best validation loss: 0.0602

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0016
    25th percentile: 0.0624
    50th percentile: 0.1370
    75th percentile: 0.2433
    100th percentile: 0.8020
  - Using 1026 samples with hard negative mining
Epoch 24/100, Train Loss: 0.0554, Val Loss: 0.0379, LR: 0.000878
  → New best validation loss: 0.0379

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0630
    50th percentile: 0.1360
    75th percentile: 0.2096
    100th percentile: 0.6667
  - Using 1026 samples with hard negative mining
Epoch 25/100, Train Loss: 0.0372, Val Loss: 0.0263, LR: 0.000868
  → New best validation loss: 0.0263

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0418
    50th percentile: 0.0887
    75th percentile: 0.1577
    100th percentile: 0.4955
  - Using 1026 samples with hard negative mining
Epoch 26/100, Train Loss: 0.0359, Val Loss: 0.0293, LR: 0.000858

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0412
    50th percentile: 0.0952
    75th percentile: 0.1602
    100th percentile: 0.4134
  - Using 1026 samples with hard negative mining
Epoch 27/100, Train Loss: 0.0226, Val Loss: 0.0228, LR: 0.000848
  → New best validation loss: 0.0228

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0370
    50th percentile: 0.0844
    75th percentile: 0.1462
    100th percentile: 0.3751
  - Using 1026 samples with hard negative mining
Epoch 28/100, Train Loss: 0.0177, Val Loss: 0.0262, LR: 0.000837

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0006
    25th percentile: 0.0394
    50th percentile: 0.0771
    75th percentile: 0.1325
    100th percentile: 0.3843
  - Using 1026 samples with hard negative mining
Epoch 29/100, Train Loss: 0.0149, Val Loss: 0.0204, LR: 0.000826
  → New best validation loss: 0.0204

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0007
    25th percentile: 0.0350
    50th percentile: 0.0661
    75th percentile: 0.1125
    100th percentile: 0.3385
  - Using 1026 samples with hard negative mining
Epoch 30/100, Train Loss: 0.0150, Val Loss: 0.0211, LR: 0.000815

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0307
    50th percentile: 0.0700
    75th percentile: 0.1194
    100th percentile: 0.2937
  - Using 1026 samples with hard negative mining
Epoch 31/100, Train Loss: 0.0089, Val Loss: 0.0180, LR: 0.000803
  → New best validation loss: 0.0180

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0305
    50th percentile: 0.0572
    75th percentile: 0.0975
    100th percentile: 0.2683
  - Using 1026 samples with hard negative mining
Epoch 32/100, Train Loss: 0.0095, Val Loss: 0.0216, LR: 0.000791

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0305
    50th percentile: 0.0661
    75th percentile: 0.1101
    100th percentile: 0.2939
  - Using 1026 samples with hard negative mining
Epoch 33/100, Train Loss: 0.0091, Val Loss: 0.0193, LR: 0.000779

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0253
    50th percentile: 0.0585
    75th percentile: 0.1075
    100th percentile: 0.2579
  - Using 1026 samples with hard negative mining
Epoch 34/100, Train Loss: 0.0074, Val Loss: 0.0187, LR: 0.000767

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0259
    50th percentile: 0.0550
    75th percentile: 0.0912
    100th percentile: 0.2215
  - Using 1026 samples with hard negative mining
Epoch 35/100, Train Loss: 0.0076, Val Loss: 0.0199, LR: 0.000754

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0261
    50th percentile: 0.0580
    75th percentile: 0.1059
    100th percentile: 0.2369
  - Using 1026 samples with hard negative mining
Epoch 36/100, Train Loss: 0.0064, Val Loss: 0.0182, LR: 0.000742

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0239
    50th percentile: 0.0534
    75th percentile: 0.0932
    100th percentile: 0.2333
  - Using 1026 samples with hard negative mining
Epoch 37/100, Train Loss: 0.0055, Val Loss: 0.0176, LR: 0.000729
  → New best validation loss: 0.0176

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0233
    50th percentile: 0.0509
    75th percentile: 0.0911
    100th percentile: 0.2144
  - Using 1026 samples with hard negative mining
Epoch 38/100, Train Loss: 0.0061, Val Loss: 0.0187, LR: 0.000716

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0208
    50th percentile: 0.0533
    75th percentile: 0.0936
    100th percentile: 0.2405
  - Using 1026 samples with hard negative mining
Epoch 39/100, Train Loss: 0.0057, Val Loss: 0.0201, LR: 0.000702

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0005
    25th percentile: 0.0309
    50th percentile: 0.0643
    75th percentile: 0.1088
    100th percentile: 0.2352
  - Using 1026 samples with hard negative mining
Epoch 40/100, Train Loss: 0.0054, Val Loss: 0.0188, LR: 0.000689

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0218
    50th percentile: 0.0468
    75th percentile: 0.0861
    100th percentile: 0.2974
  - Using 1026 samples with hard negative mining
Epoch 41/100, Train Loss: 0.0048, Val Loss: 0.0180, LR: 0.000676

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0202
    50th percentile: 0.0430
    75th percentile: 0.0746
    100th percentile: 0.1869
  - Using 1026 samples with hard negative mining
Epoch 42/100, Train Loss: 0.0041, Val Loss: 0.0175, LR: 0.000662
  → New best validation loss: 0.0175

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0199
    50th percentile: 0.0472
    75th percentile: 0.0833
    100th percentile: 0.2024
  - Using 1026 samples with hard negative mining
Epoch 43/100, Train Loss: 0.0039, Val Loss: 0.0156, LR: 0.000648
  → New best validation loss: 0.0156

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0175
    50th percentile: 0.0370
    75th percentile: 0.0600
    100th percentile: 0.1894
  - Using 1026 samples with hard negative mining
Epoch 44/100, Train Loss: 0.0048, Val Loss: 0.0172, LR: 0.000634

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0201
    50th percentile: 0.0370
    75th percentile: 0.0631
    100th percentile: 0.1584
  - Using 1026 samples with hard negative mining
Epoch 45/100, Train Loss: 0.0042, Val Loss: 0.0157, LR: 0.000620

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0171
    50th percentile: 0.0352
    75th percentile: 0.0586
    100th percentile: 0.1412
  - Using 1026 samples with hard negative mining
Epoch 46/100, Train Loss: 0.0035, Val Loss: 0.0150, LR: 0.000606
  → New best validation loss: 0.0150

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0149
    50th percentile: 0.0316
    75th percentile: 0.0526
    100th percentile: 0.1290
  - Using 1026 samples with hard negative mining
Epoch 47/100, Train Loss: 0.0033, Val Loss: 0.0160, LR: 0.000592

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0148
    50th percentile: 0.0281
    75th percentile: 0.0515
    100th percentile: 0.1420
  - Using 1026 samples with hard negative mining
Epoch 48/100, Train Loss: 0.0028, Val Loss: 0.0169, LR: 0.000578

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0168
    50th percentile: 0.0381
    75th percentile: 0.0656
    100th percentile: 0.1585
  - Using 1026 samples with hard negative mining
Epoch 49/100, Train Loss: 0.0024, Val Loss: 0.0167, LR: 0.000564

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0152
    50th percentile: 0.0298
    75th percentile: 0.0478
    100th percentile: 0.1749
  - Using 1026 samples with hard negative mining
Epoch 50/100, Train Loss: 0.0028, Val Loss: 0.0145, LR: 0.000550
  → New best validation loss: 0.0145

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0128
    50th percentile: 0.0263
    75th percentile: 0.0443
    100th percentile: 0.1062
  - Using 1026 samples with hard negative mining
Epoch 51/100, Train Loss: 0.0024, Val Loss: 0.0152, LR: 0.000536

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0120
    50th percentile: 0.0284
    75th percentile: 0.0516
    100th percentile: 0.1357
  - Using 1026 samples with hard negative mining
Epoch 52/100, Train Loss: 0.0024, Val Loss: 0.0148, LR: 0.000522

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0132
    50th percentile: 0.0249
    75th percentile: 0.0410
    100th percentile: 0.1058
  - Using 1026 samples with hard negative mining
Epoch 53/100, Train Loss: 0.0023, Val Loss: 0.0159, LR: 0.000508

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0172
    50th percentile: 0.0355
    75th percentile: 0.0576
    100th percentile: 0.1396
  - Using 1026 samples with hard negative mining
Epoch 54/100, Train Loss: 0.0020, Val Loss: 0.0149, LR: 0.000494

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0135
    50th percentile: 0.0253
    75th percentile: 0.0408
    100th percentile: 0.1097
  - Using 1026 samples with hard negative mining
Epoch 55/100, Train Loss: 0.0020, Val Loss: 0.0149, LR: 0.000480

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0126
    50th percentile: 0.0262
    75th percentile: 0.0466
    100th percentile: 0.1212
  - Using 1026 samples with hard negative mining
Epoch 56/100, Train Loss: 0.0015, Val Loss: 0.0150, LR: 0.000466

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0130
    50th percentile: 0.0267
    75th percentile: 0.0457
    100th percentile: 0.1120
  - Using 1026 samples with hard negative mining
Epoch 57/100, Train Loss: 0.0014, Val Loss: 0.0146, LR: 0.000452

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0128
    50th percentile: 0.0258
    75th percentile: 0.0436
    100th percentile: 0.1120
  - Using 1026 samples with hard negative mining
Epoch 58/100, Train Loss: 0.0014, Val Loss: 0.0138, LR: 0.000438
  → New best validation loss: 0.0138

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0116
    50th percentile: 0.0205
    75th percentile: 0.0321
    100th percentile: 0.0908
  - Using 1026 samples with hard negative mining
Epoch 59/100, Train Loss: 0.0015, Val Loss: 0.0141, LR: 0.000424

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0112
    50th percentile: 0.0240
    75th percentile: 0.0388
    100th percentile: 0.1000
  - Using 1026 samples with hard negative mining
Epoch 60/100, Train Loss: 0.0017, Val Loss: 0.0133, LR: 0.000411
  → New best validation loss: 0.0133

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0079
    50th percentile: 0.0189
    75th percentile: 0.0322
    100th percentile: 0.0839
  - Using 1026 samples with hard negative mining
Epoch 61/100, Train Loss: 0.0013, Val Loss: 0.0144, LR: 0.000398

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0076
    50th percentile: 0.0165
    75th percentile: 0.0292
    100th percentile: 0.0809
  - Using 1026 samples with hard negative mining
Epoch 62/100, Train Loss: 0.0012, Val Loss: 0.0142, LR: 0.000384

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0082
    50th percentile: 0.0166
    75th percentile: 0.0272
    100th percentile: 0.0937
  - Using 1026 samples with hard negative mining
Epoch 63/100, Train Loss: 0.0014, Val Loss: 0.0151, LR: 0.000371

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0087
    50th percentile: 0.0185
    75th percentile: 0.0300
    100th percentile: 0.0817
  - Using 1026 samples with hard negative mining
Epoch 64/100, Train Loss: 0.0011, Val Loss: 0.0148, LR: 0.000358

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0111
    50th percentile: 0.0226
    75th percentile: 0.0361
    100th percentile: 0.0837
  - Using 1026 samples with hard negative mining
Epoch 65/100, Train Loss: 0.0008, Val Loss: 0.0145, LR: 0.000346

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0086
    50th percentile: 0.0177
    75th percentile: 0.0292
    100th percentile: 0.0862
  - Using 1026 samples with hard negative mining
Epoch 66/100, Train Loss: 0.0009, Val Loss: 0.0137, LR: 0.000333

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0074
    50th percentile: 0.0147
    75th percentile: 0.0230
    100th percentile: 0.0694
  - Using 1026 samples with hard negative mining
Epoch 67/100, Train Loss: 0.0008, Val Loss: 0.0145, LR: 0.000321

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0087
    50th percentile: 0.0174
    75th percentile: 0.0302
    100th percentile: 0.0739
  - Using 1026 samples with hard negative mining
Epoch 68/100, Train Loss: 0.0008, Val Loss: 0.0144, LR: 0.000309

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0067
    50th percentile: 0.0128
    75th percentile: 0.0219
    100th percentile: 0.0624
  - Using 1026 samples with hard negative mining
Epoch 69/100, Train Loss: 0.0012, Val Loss: 0.0145, LR: 0.000297

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0076
    50th percentile: 0.0168
    75th percentile: 0.0264
    100th percentile: 0.0665
  - Using 1026 samples with hard negative mining
Epoch 70/100, Train Loss: 0.0008, Val Loss: 0.0144, LR: 0.000285

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0059
    50th percentile: 0.0123
    75th percentile: 0.0211
    100th percentile: 0.0596
  - Using 1026 samples with hard negative mining
Epoch 71/100, Train Loss: 0.0009, Val Loss: 0.0145, LR: 0.000274

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0093
    50th percentile: 0.0194
    75th percentile: 0.0300
    100th percentile: 0.0667
  - Using 1026 samples with hard negative mining
Epoch 72/100, Train Loss: 0.0007, Val Loss: 0.0143, LR: 0.000263

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0055
    50th percentile: 0.0116
    75th percentile: 0.0198
    100th percentile: 0.0582
  - Using 1026 samples with hard negative mining
Epoch 73/100, Train Loss: 0.0006, Val Loss: 0.0141, LR: 0.000252

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0061
    50th percentile: 0.0126
    75th percentile: 0.0211
    100th percentile: 0.0668
  - Using 1026 samples with hard negative mining
Epoch 74/100, Train Loss: 0.0007, Val Loss: 0.0140, LR: 0.000242

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0051
    50th percentile: 0.0100
    75th percentile: 0.0177
    100th percentile: 0.0413
  - Using 1026 samples with hard negative mining
Epoch 75/100, Train Loss: 0.0005, Val Loss: 0.0141, LR: 0.000232

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0062
    50th percentile: 0.0132
    75th percentile: 0.0221
    100th percentile: 0.0510
  - Using 1026 samples with hard negative mining
Epoch 76/100, Train Loss: 0.0005, Val Loss: 0.0140, LR: 0.000222

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0061
    50th percentile: 0.0128
    75th percentile: 0.0216
    100th percentile: 0.0501
  - Using 1026 samples with hard negative mining
Epoch 77/100, Train Loss: 0.0005, Val Loss: 0.0142, LR: 0.000212

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0049
    50th percentile: 0.0113
    75th percentile: 0.0202
    100th percentile: 0.0490
  - Using 1026 samples with hard negative mining
Epoch 78/100, Train Loss: 0.0005, Val Loss: 0.0142, LR: 0.000203

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0050
    50th percentile: 0.0102
    75th percentile: 0.0171
    100th percentile: 0.0393
  - Using 1026 samples with hard negative mining
Epoch 79/100, Train Loss: 0.0005, Val Loss: 0.0142, LR: 0.000194

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0053
    50th percentile: 0.0106
    75th percentile: 0.0189
    100th percentile: 0.0485
  - Using 1026 samples with hard negative mining
Epoch 80/100, Train Loss: 0.0004, Val Loss: 0.0145, LR: 0.000186
Early stopping at epoch 80

Evaluating model...

Applying test-time augmentation...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4
  - TTA iteration 1/3
  - TTA iteration 2/3
  - TTA iteration 3/3
  - Final predictions created from 4 versions

Test Metrics:
RMSE: 0.1291
R²: 0.8824
MAE: 0.1009
Spearman correlation: 0.9167

Metrics by site:
  Site 0: RMSE=0.1324, R²=0.6157, MAE=0.1017, n=83
  Site 1: RMSE=0.1084, R²=0.6909, MAE=0.0839, n=24
  Site 2: RMSE=0.1239, R²=0.5978, MAE=0.1051, n=21
  Site 3: RMSE=0.1428, R²=0.8184, MAE=0.1137, n=20

========================================
Running Hybrid CV Fold 5
========================================

Adding derived features...
Added 5 derived features, new shape: (499, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (121, 64, 24, 24)

Adding derived features...
Added 5 derived features, new shape: (164, 64, 24, 24)

Standardizing features...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4

Creating CNN_COORDINATE model...

Training model...
Epoch 1/100, Train Loss: 2.1374, Val Loss: 2.4304, LR: 0.001000
  → New best validation loss: 2.4304
Epoch 2/100, Train Loss: 0.9148, Val Loss: 0.5438, LR: 0.000999
  → New best validation loss: 0.5438
Epoch 3/100, Train Loss: 0.6844, Val Loss: 0.2983, LR: 0.000998
  → New best validation loss: 0.2983
Epoch 4/100, Train Loss: 0.5943, Val Loss: 0.2617, LR: 0.000996
  → New best validation loss: 0.2617
Epoch 5/100, Train Loss: 0.6331, Val Loss: 0.2523, LR: 0.000994
  → New best validation loss: 0.2523
Epoch 6/100, Train Loss: 0.6197, Val Loss: 0.2870, LR: 0.000992
Epoch 7/100, Train Loss: 0.6266, Val Loss: 0.2526, LR: 0.000989
Epoch 8/100, Train Loss: 0.5758, Val Loss: 0.2790, LR: 0.000986
Epoch 9/100, Train Loss: 0.5846, Val Loss: 0.2722, LR: 0.000982
Epoch 10/100, Train Loss: 0.5353, Val Loss: 0.2587, LR: 0.000978
Epoch 11/100, Train Loss: 0.5652, Val Loss: 0.2387, LR: 0.000973
  → New best validation loss: 0.2387
Epoch 12/100, Train Loss: 0.5248, Val Loss: 0.2344, LR: 0.000968
  → New best validation loss: 0.2344
Epoch 13/100, Train Loss: 0.4970, Val Loss: 0.2733, LR: 0.000963
Epoch 14/100, Train Loss: 0.5271, Val Loss: 0.2239, LR: 0.000957
  → New best validation loss: 0.2239
Epoch 15/100, Train Loss: 0.4974, Val Loss: 0.2354, LR: 0.000951
Epoch 16/100, Train Loss: 0.5123, Val Loss: 0.2131, LR: 0.000944
  → New best validation loss: 0.2131
Epoch 17/100, Train Loss: 0.5689, Val Loss: 0.1492, LR: 0.000937
  → New best validation loss: 0.1492
Epoch 18/100, Train Loss: 0.4716, Val Loss: 0.1230, LR: 0.000930
  → New best validation loss: 0.1230
Epoch 19/100, Train Loss: 0.4793, Val Loss: 0.1062, LR: 0.000922
  → New best validation loss: 0.1062
Epoch 20/100, Train Loss: 0.4818, Val Loss: 0.1109, LR: 0.000914

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.1130
    50th percentile: 0.2188
    75th percentile: 0.3327
    100th percentile: 2.4591
  - Using 998 samples with hard negative mining
Epoch 21/100, Train Loss: 0.2631, Val Loss: 0.0589, LR: 0.000906
  → New best validation loss: 0.0589

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0874
    50th percentile: 0.1708
    75th percentile: 0.2793
    100th percentile: 0.7739
  - Using 998 samples with hard negative mining
Epoch 22/100, Train Loss: 0.0807, Val Loss: 0.0311, LR: 0.000897
  → New best validation loss: 0.0311

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0007
    25th percentile: 0.0545
    50th percentile: 0.1110
    75th percentile: 0.1897
    100th percentile: 0.7773
  - Using 998 samples with hard negative mining
Epoch 23/100, Train Loss: 0.0462, Val Loss: 0.0241, LR: 0.000888
  → New best validation loss: 0.0241

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0433
    50th percentile: 0.0904
    75th percentile: 0.1596
    100th percentile: 0.4053
  - Using 998 samples with hard negative mining
Epoch 24/100, Train Loss: 0.0346, Val Loss: 0.0225, LR: 0.000878
  → New best validation loss: 0.0225

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0458
    50th percentile: 0.0908
    75th percentile: 0.1491
    100th percentile: 0.3866
  - Using 998 samples with hard negative mining
Epoch 25/100, Train Loss: 0.0219, Val Loss: 0.0243, LR: 0.000868

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0366
    50th percentile: 0.0794
    75th percentile: 0.1287
    100th percentile: 0.4115
  - Using 998 samples with hard negative mining
Epoch 26/100, Train Loss: 0.0182, Val Loss: 0.0201, LR: 0.000858
  → New best validation loss: 0.0201

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0004
    25th percentile: 0.0369
    50th percentile: 0.0760
    75th percentile: 0.1191
    100th percentile: 0.4177
  - Using 998 samples with hard negative mining
Epoch 27/100, Train Loss: 0.0165, Val Loss: 0.0219, LR: 0.000848

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0339
    50th percentile: 0.0674
    75th percentile: 0.1136
    100th percentile: 0.3143
  - Using 998 samples with hard negative mining
Epoch 28/100, Train Loss: 0.0130, Val Loss: 0.0157, LR: 0.000837
  → New best validation loss: 0.0157

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0342
    50th percentile: 0.0650
    75th percentile: 0.1036
    100th percentile: 0.2564
  - Using 998 samples with hard negative mining
Epoch 29/100, Train Loss: 0.0114, Val Loss: 0.0194, LR: 0.000826

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0298
    50th percentile: 0.0601
    75th percentile: 0.1049
    100th percentile: 0.2424
  - Using 998 samples with hard negative mining
Epoch 30/100, Train Loss: 0.0104, Val Loss: 0.0166, LR: 0.000815

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0318
    50th percentile: 0.0621
    75th percentile: 0.1038
    100th percentile: 0.4900
  - Using 998 samples with hard negative mining
Epoch 31/100, Train Loss: 0.0125, Val Loss: 0.0179, LR: 0.000803

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0243
    50th percentile: 0.0519
    75th percentile: 0.0883
    100th percentile: 0.2634
  - Using 998 samples with hard negative mining
Epoch 32/100, Train Loss: 0.0085, Val Loss: 0.0206, LR: 0.000791

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0008
    25th percentile: 0.0313
    50th percentile: 0.0603
    75th percentile: 0.1054
    100th percentile: 0.3641
  - Using 998 samples with hard negative mining
Epoch 33/100, Train Loss: 0.0072, Val Loss: 0.0165, LR: 0.000779

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0238
    50th percentile: 0.0512
    75th percentile: 0.0878
    100th percentile: 0.2657
  - Using 998 samples with hard negative mining
Epoch 34/100, Train Loss: 0.0066, Val Loss: 0.0167, LR: 0.000767

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0260
    50th percentile: 0.0533
    75th percentile: 0.0826
    100th percentile: 0.1909
  - Using 998 samples with hard negative mining
Epoch 35/100, Train Loss: 0.0066, Val Loss: 0.0160, LR: 0.000754

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0202
    50th percentile: 0.0458
    75th percentile: 0.0751
    100th percentile: 0.1787
  - Using 998 samples with hard negative mining
Epoch 36/100, Train Loss: 0.0066, Val Loss: 0.0183, LR: 0.000742

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0237
    50th percentile: 0.0469
    75th percentile: 0.0836
    100th percentile: 0.2095
  - Using 998 samples with hard negative mining
Epoch 37/100, Train Loss: 0.0054, Val Loss: 0.0165, LR: 0.000729

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0215
    50th percentile: 0.0437
    75th percentile: 0.0821
    100th percentile: 0.1901
  - Using 998 samples with hard negative mining
Epoch 38/100, Train Loss: 0.0059, Val Loss: 0.0160, LR: 0.000716

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0217
    50th percentile: 0.0394
    75th percentile: 0.0653
    100th percentile: 0.1615
  - Using 998 samples with hard negative mining
Epoch 39/100, Train Loss: 0.0046, Val Loss: 0.0160, LR: 0.000702

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0172
    50th percentile: 0.0387
    75th percentile: 0.0685
    100th percentile: 0.1782
  - Using 998 samples with hard negative mining
Epoch 40/100, Train Loss: 0.0054, Val Loss: 0.0154, LR: 0.000689
  → New best validation loss: 0.0154

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0193
    50th percentile: 0.0389
    75th percentile: 0.0656
    100th percentile: 0.1493
  - Using 998 samples with hard negative mining
Epoch 41/100, Train Loss: 0.0042, Val Loss: 0.0159, LR: 0.000676

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0003
    25th percentile: 0.0168
    50th percentile: 0.0351
    75th percentile: 0.0558
    100th percentile: 0.1529
  - Using 998 samples with hard negative mining
Epoch 42/100, Train Loss: 0.0041, Val Loss: 0.0161, LR: 0.000662

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0205
    50th percentile: 0.0387
    75th percentile: 0.0630
    100th percentile: 0.1486
  - Using 998 samples with hard negative mining
Epoch 43/100, Train Loss: 0.0033, Val Loss: 0.0158, LR: 0.000648

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0179
    50th percentile: 0.0328
    75th percentile: 0.0581
    100th percentile: 0.2426
  - Using 998 samples with hard negative mining
Epoch 44/100, Train Loss: 0.0038, Val Loss: 0.0164, LR: 0.000634

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0178
    50th percentile: 0.0378
    75th percentile: 0.0654
    100th percentile: 0.1560
  - Using 998 samples with hard negative mining
Epoch 45/100, Train Loss: 0.0030, Val Loss: 0.0149, LR: 0.000620
  → New best validation loss: 0.0149

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0159
    50th percentile: 0.0320
    75th percentile: 0.0526
    100th percentile: 0.1220
  - Using 998 samples with hard negative mining
Epoch 46/100, Train Loss: 0.0027, Val Loss: 0.0167, LR: 0.000606

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0141
    50th percentile: 0.0309
    75th percentile: 0.0545
    100th percentile: 0.2478
  - Using 998 samples with hard negative mining
Epoch 47/100, Train Loss: 0.0039, Val Loss: 0.0161, LR: 0.000592

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0159
    50th percentile: 0.0344
    75th percentile: 0.0560
    100th percentile: 0.2071
  - Using 998 samples with hard negative mining
Epoch 48/100, Train Loss: 0.0033, Val Loss: 0.0153, LR: 0.000578

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0145
    50th percentile: 0.0309
    75th percentile: 0.0543
    100th percentile: 0.1423
  - Using 998 samples with hard negative mining
Epoch 49/100, Train Loss: 0.0023, Val Loss: 0.0153, LR: 0.000564

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0199
    50th percentile: 0.0403
    75th percentile: 0.0619
    100th percentile: 0.1304
  - Using 998 samples with hard negative mining
Epoch 50/100, Train Loss: 0.0021, Val Loss: 0.0152, LR: 0.000550

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0117
    50th percentile: 0.0234
    75th percentile: 0.0405
    100th percentile: 0.1173
  - Using 998 samples with hard negative mining
Epoch 51/100, Train Loss: 0.0020, Val Loss: 0.0147, LR: 0.000536
  → New best validation loss: 0.0147

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0111
    50th percentile: 0.0225
    75th percentile: 0.0403
    100th percentile: 0.1119
  - Using 998 samples with hard negative mining
Epoch 52/100, Train Loss: 0.0018, Val Loss: 0.0147, LR: 0.000522
  → New best validation loss: 0.0147

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0107
    50th percentile: 0.0226
    75th percentile: 0.0365
    100th percentile: 0.1128
  - Using 998 samples with hard negative mining
Epoch 53/100, Train Loss: 0.0017, Val Loss: 0.0142, LR: 0.000508
  → New best validation loss: 0.0142

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0130
    50th percentile: 0.0288
    75th percentile: 0.0446
    100th percentile: 0.1333
  - Using 998 samples with hard negative mining
Epoch 54/100, Train Loss: 0.0019, Val Loss: 0.0150, LR: 0.000494

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0110
    50th percentile: 0.0233
    75th percentile: 0.0368
    100th percentile: 0.1032
  - Using 998 samples with hard negative mining
Epoch 55/100, Train Loss: 0.0016, Val Loss: 0.0148, LR: 0.000480

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0121
    50th percentile: 0.0251
    75th percentile: 0.0403
    100th percentile: 0.0856
  - Using 998 samples with hard negative mining
Epoch 56/100, Train Loss: 0.0014, Val Loss: 0.0149, LR: 0.000466

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0102
    50th percentile: 0.0207
    75th percentile: 0.0347
    100th percentile: 0.0951
  - Using 998 samples with hard negative mining
Epoch 57/100, Train Loss: 0.0015, Val Loss: 0.0155, LR: 0.000452

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0100
    50th percentile: 0.0200
    75th percentile: 0.0342
    100th percentile: 0.1067
  - Using 998 samples with hard negative mining
Epoch 58/100, Train Loss: 0.0017, Val Loss: 0.0166, LR: 0.000438

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0209
    50th percentile: 0.0393
    75th percentile: 0.0584
    100th percentile: 0.1076
  - Using 998 samples with hard negative mining
Epoch 59/100, Train Loss: 0.0014, Val Loss: 0.0149, LR: 0.000424

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0089
    50th percentile: 0.0183
    75th percentile: 0.0324
    100th percentile: 0.1471
  - Using 998 samples with hard negative mining
Epoch 60/100, Train Loss: 0.0016, Val Loss: 0.0151, LR: 0.000411

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0089
    50th percentile: 0.0181
    75th percentile: 0.0307
    100th percentile: 0.1076
  - Using 998 samples with hard negative mining
Epoch 61/100, Train Loss: 0.0011, Val Loss: 0.0138, LR: 0.000398
  → New best validation loss: 0.0138

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0096
    50th percentile: 0.0196
    75th percentile: 0.0324
    100th percentile: 0.0865
  - Using 998 samples with hard negative mining
Epoch 62/100, Train Loss: 0.0010, Val Loss: 0.0145, LR: 0.000384

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0088
    50th percentile: 0.0177
    75th percentile: 0.0295
    100th percentile: 0.0755
  - Using 998 samples with hard negative mining
Epoch 63/100, Train Loss: 0.0008, Val Loss: 0.0149, LR: 0.000371

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0083
    50th percentile: 0.0196
    75th percentile: 0.0334
    100th percentile: 0.0790
  - Using 998 samples with hard negative mining
Epoch 64/100, Train Loss: 0.0009, Val Loss: 0.0147, LR: 0.000358

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0125
    50th percentile: 0.0245
    75th percentile: 0.0371
    100th percentile: 0.0909
  - Using 998 samples with hard negative mining
Epoch 65/100, Train Loss: 0.0016, Val Loss: 0.0142, LR: 0.000346

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0078
    50th percentile: 0.0159
    75th percentile: 0.0274
    100th percentile: 0.1030
  - Using 998 samples with hard negative mining
Epoch 66/100, Train Loss: 0.0010, Val Loss: 0.0146, LR: 0.000333

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0072
    50th percentile: 0.0153
    75th percentile: 0.0249
    100th percentile: 0.0702
  - Using 998 samples with hard negative mining
Epoch 67/100, Train Loss: 0.0013, Val Loss: 0.0148, LR: 0.000321

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0067
    50th percentile: 0.0150
    75th percentile: 0.0286
    100th percentile: 0.1285
  - Using 998 samples with hard negative mining
Epoch 68/100, Train Loss: 0.0009, Val Loss: 0.0138, LR: 0.000309

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0073
    50th percentile: 0.0143
    75th percentile: 0.0233
    100th percentile: 0.0634
  - Using 998 samples with hard negative mining
Epoch 69/100, Train Loss: 0.0006, Val Loss: 0.0141, LR: 0.000297

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0060
    50th percentile: 0.0126
    75th percentile: 0.0208
    100th percentile: 0.0716
  - Using 998 samples with hard negative mining
Epoch 70/100, Train Loss: 0.0007, Val Loss: 0.0141, LR: 0.000285

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0074
    50th percentile: 0.0155
    75th percentile: 0.0252
    100th percentile: 0.0732
  - Using 998 samples with hard negative mining
Epoch 71/100, Train Loss: 0.0007, Val Loss: 0.0139, LR: 0.000274

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0063
    50th percentile: 0.0133
    75th percentile: 0.0229
    100th percentile: 0.0702
  - Using 998 samples with hard negative mining
Epoch 72/100, Train Loss: 0.0006, Val Loss: 0.0143, LR: 0.000263

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0058
    50th percentile: 0.0118
    75th percentile: 0.0188
    100th percentile: 0.0473
  - Using 998 samples with hard negative mining
Epoch 73/100, Train Loss: 0.0005, Val Loss: 0.0144, LR: 0.000252

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0001
    25th percentile: 0.0049
    50th percentile: 0.0097
    75th percentile: 0.0151
    100th percentile: 0.0414
  - Using 998 samples with hard negative mining
Epoch 74/100, Train Loss: 0.0004, Val Loss: 0.0146, LR: 0.000242

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0002
    25th percentile: 0.0050
    50th percentile: 0.0110
    75th percentile: 0.0189
    100th percentile: 0.0495
  - Using 998 samples with hard negative mining
Epoch 75/100, Train Loss: 0.0004, Val Loss: 0.0144, LR: 0.000232

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0099
    50th percentile: 0.0186
    75th percentile: 0.0283
    100th percentile: 0.0620
  - Using 998 samples with hard negative mining
Epoch 76/100, Train Loss: 0.0006, Val Loss: 0.0139, LR: 0.000222

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0049
    50th percentile: 0.0099
    75th percentile: 0.0174
    100th percentile: 0.0457
  - Using 998 samples with hard negative mining
Epoch 77/100, Train Loss: 0.0004, Val Loss: 0.0142, LR: 0.000212

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0051
    50th percentile: 0.0102
    75th percentile: 0.0164
    100th percentile: 0.0392
  - Using 998 samples with hard negative mining
Epoch 78/100, Train Loss: 0.0003, Val Loss: 0.0141, LR: 0.000203

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0043
    50th percentile: 0.0089
    75th percentile: 0.0148
    100th percentile: 0.0368
  - Using 998 samples with hard negative mining
Epoch 79/100, Train Loss: 0.0003, Val Loss: 0.0141, LR: 0.000194

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0048
    50th percentile: 0.0101
    75th percentile: 0.0171
    100th percentile: 0.0411
  - Using 998 samples with hard negative mining
Epoch 80/100, Train Loss: 0.0003, Val Loss: 0.0145, LR: 0.000186

Creating hard negative mining sampler:
  - Error distribution:
    0th percentile: 0.0000
    25th percentile: 0.0041
    50th percentile: 0.0078
    75th percentile: 0.0134
    100th percentile: 0.0392
  - Using 998 samples with hard negative mining
Epoch 81/100, Train Loss: 0.0003, Val Loss: 0.0143, LR: 0.000178
Early stopping at epoch 81

Evaluating model...

Applying test-time augmentation...

Setting up data augmentation:
  - Using geometric augmentation (flips, rotations)
  - Using spectral augmentation (band jittering)
  - Augmentation probability: 0.7
  - Total transformations: 4
  - TTA iteration 1/3
  - TTA iteration 2/3
  - TTA iteration 3/3
  - Final predictions created from 4 versions

Test Metrics:
RMSE: 0.1460
R²: 0.8883
MAE: 0.1032
Spearman correlation: 0.9408

Metrics by site:
  Site 0: RMSE=0.1735, R²=0.8161, MAE=0.1229, n=96
  Site 1: RMSE=0.1170, R²=0.6928, MAE=0.0917, n=21
  Site 2: RMSE=0.0753, R²=0.9093, MAE=0.0642, n=25
  Site 3: RMSE=0.0895, R²=0.5854, MAE=0.0721, n=22

Hybrid CV Summary:
RMSE: 0.1308 ± 0.0197
R²: 0.8994 ± 0.0249
MAE: 0.0983 ± 0.0128
Spearman: 0.9387 ± 0.0142

Creating visualizations...

Hybrid CV complete. Results saved to spatial_biomass_results/hybrid_cv/20250528_044727

Hybrid cross-validation complete!
"""
