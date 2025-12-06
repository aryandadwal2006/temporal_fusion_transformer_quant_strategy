import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Tuple, Optional
import time

try:
    import cudf
    import dask_cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    import dask.dataframe as dd
    CUDF_AVAILABLE = False

from config import Config

class StreamingDataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.use_gpu = config.USE_GPU and CUDF_AVAILABLE
        self.feature_stats = {}
        self.stats_computed = False
        print(f"StreamingDataProcessor initialized with GPU: {self.use_gpu}")
    
    def load_parquet_lazy(self, day_num: int):
        parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
        if not os.path.exists(parquet_path):
            return None
        try:
            if self.use_gpu:
                ddf = dask_cudf.read_parquet(parquet_path)
            else:
                ddf = dd.read_parquet(parquet_path)
            return ddf
        except Exception as e:
            print(f"Error loading day{day_num}: {e}")
            return None
    
    def filter_stable_period(self, ddf):
        stable_time = self.config.STABLE_START_TIME
        ddf_filtered = ddf[ddf['Time'] >= stable_time]
        return ddf_filtered
    
    def compute_feature_statistics(self, day_numbers: List[int], sample_rate: float = None):
        """
        OPTIMIZED: Parallel feature statistics computation using Dask
        """
        if sample_rate is None:
            sample_rate = self.config.FEATURE_STATS_SAMPLE_RATE
        
        print(f"Computing feature statistics (parallel mode, {sample_rate*100}% sample)...")
        start_time = time.time()
        
        feature_cols = self.config.get_feature_columns()
        
        # Limit number of days if configured
        if self.config.FEATURE_STATS_MAX_DAYS and len(day_numbers) > self.config.FEATURE_STATS_MAX_DAYS:
            # Sample evenly distributed days
            step = len(day_numbers) // self.config.FEATURE_STATS_MAX_DAYS
            day_numbers = day_numbers[::step][:self.config.FEATURE_STATS_MAX_DAYS]
            print(f"  Using {len(day_numbers)} evenly distributed days for statistics")
        
        # Load all days into a single Dask dataframe for parallel processing
        ddf_list = []
        for day_num in day_numbers:
            ddf = self.load_parquet_lazy(day_num)
            if ddf is None:
                continue
            ddf = self.filter_stable_period(ddf)
            ddf_list.append(ddf)
        
        if not ddf_list:
            print("No valid data found for statistics computation")
            return
        
        # Concatenate all dataframes
        print(f"  Concatenating {len(ddf_list)} days...")
        combined_ddf = dd.concat(ddf_list, axis=0, interleave_partitions=True) if not self.use_gpu else dask_cudf.concat(ddf_list, axis=0, interleave_partitions=True)
        
        # Sample if needed
        if sample_rate < 1.0:
            print(f"  Sampling {sample_rate*100}% of data...")
            combined_ddf = combined_ddf.sample(frac=sample_rate, random_state=self.config.RANDOM_SEED)
        
        # Select only feature columns that exist
        available_cols = [col for col in feature_cols if col in combined_ddf.columns]
        print(f"  Found {len(available_cols)}/{len(feature_cols)} feature columns")
        
        if not available_cols:
            print("No feature columns found!")
            return
        
        # PARALLEL COMPUTATION: Use Dask's native mean and std
        print("  Computing statistics in parallel...")
        feature_ddf = combined_ddf[available_cols]
        
        # Replace inf values before computation
        feature_ddf = feature_ddf.replace([np.inf, -np.inf], np.nan)
        
        # Compute mean and std in parallel (this triggers Dask computation)
        means = feature_ddf.mean().compute()
        stds = feature_ddf.std().compute()
        
        # Build statistics dictionary
        self.feature_stats = {}
        if self.use_gpu:
            means_dict = means.to_pandas().to_dict()
            stds_dict = stds.to_pandas().to_dict()
        else:
            means_dict = means.to_dict()
            stds_dict = stds.to_dict()
        
        for col in available_cols:
            mean_val = means_dict.get(col, 0.0)
            std_val = stds_dict.get(col, 1.0)
            
            # Handle NaN/invalid values
            if np.isnan(mean_val) or np.isinf(mean_val):
                mean_val = 0.0
            if np.isnan(std_val) or np.isinf(std_val) or std_val < 1e-8:
                std_val = 1.0
            
            self.feature_stats[col] = {'mean': float(mean_val), 'std': float(std_val)}
        
        # Fill in missing columns with default values
        for col in feature_cols:
            if col not in self.feature_stats:
                self.feature_stats[col] = {'mean': 0.0, 'std': 1.0}
        
        self.stats_computed = True
        
        elapsed_time = time.time() - start_time
        print(f"  Statistics computed in {elapsed_time:.2f} seconds")
        print(f"  Statistics ready for {len(self.feature_stats)} features")
        
        # Cleanup
        del combined_ddf, feature_ddf, ddf_list
        gc.collect()
    
    def normalize_partition(self, partition):
        feature_cols = self.config.get_feature_columns()
        for col in feature_cols:
            if col in partition.columns and col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                partition[col] = (partition[col] - mean) / std
                if self.use_gpu:
                    partition[col] = partition[col].replace([np.inf, -np.inf], 0)
                    partition[col] = partition[col].fillna(0)
                else:
                    partition[col] = partition[col].replace([np.inf, -np.inf], np.nan)
                    partition[col] = partition[col].fillna(0)
                partition[col] = partition[col].clip(-5, 5)
        return partition
    
    def create_target_variable_partition(self, partition):
        """
        MODIFIED: Create slope-based target instead of return-based target
        Slope = (Price_t+15 - Price_t) / 15 seconds
        """
        slope_window = self.config.SLOPE_PREDICTION_WINDOW
        
        # Calculate slope: (future_price - current_price) / time_window
        partition['future_price'] = partition['Price'].shift(-slope_window)
        partition['price_slope'] = (partition['future_price'] - partition['Price']) / slope_window
        
        # Handle inf and nan
        if self.use_gpu:
            partition['price_slope'] = partition['price_slope'].replace([np.inf, -np.inf], 0)
            partition['price_slope'] = partition['price_slope'].fillna(0)
        else:
            partition['price_slope'] = partition['price_slope'].replace([np.inf, -np.inf], np.nan)
            partition['price_slope'] = partition['price_slope'].fillna(0)
        
        # Classify slope into directions: -1 (down), 0 (neutral), 1 (up)
        partition['target_direction'] = 0
        partition.loc[partition['price_slope'] > self.config.SLOPE_THRESHOLD_POSITIVE, 'target_direction'] = 1
        partition.loc[partition['price_slope'] < self.config.SLOPE_THRESHOLD_NEGATIVE, 'target_direction'] = -1
        
        # Drop temporary columns
        partition = partition.drop(columns=['future_price'])
        
        return partition
    
    def process_and_save_sequences(self, day_numbers: List[int], output_prefix: str):
        """
        FIXED: Better memory management to prevent file access errors on Windows
        """
        print(f"Processing days {day_numbers[0]} to {day_numbers[-1]}...")
        feature_cols = self.config.get_feature_columns()
        lookback = self.config.LOOKBACK_WINDOW
        slope_window = self.config.SLOPE_PREDICTION_WINDOW
        
        # More accurate sequence estimation
        sequences_per_day = 15000  # approximate
        max_sequences_estimate = len(day_numbers) * sequences_per_day
        
        # Create memory-mapped arrays
        X_memmap_path = f"{self.config.CACHE_DIR}/{output_prefix}_X.dat"
        y_memmap_path = f"{self.config.CACHE_DIR}/{output_prefix}_y.dat"
        
        print(f"  Allocating memory-mapped arrays for ~{max_sequences_estimate:,} sequences...")
        X_memmap = np.memmap(X_memmap_path, dtype='float32', mode='w+', 
                            shape=(max_sequences_estimate, lookback, len(feature_cols)))
        y_memmap = np.memmap(y_memmap_path, dtype='int8', mode='w+', 
                            shape=(max_sequences_estimate,))
        
        sequence_idx = 0
        
        for day_idx, day_num in enumerate(day_numbers):
            day_start_time = time.time()
            
            ddf = self.load_parquet_lazy(day_num)
            if ddf is None:
                continue
            
            ddf = self.filter_stable_period(ddf)
            ddf = ddf.map_partitions(self.normalize_partition)
            ddf = ddf.map_partitions(self.create_target_variable_partition)
            
            # Compute to pandas/cudf
            df = ddf.compute()
            
            # Convert to numpy
            available_cols = [col for col in feature_cols if col in df.columns]
            if self.use_gpu:
                df_features = df[available_cols].to_numpy()
                df_targets = df['target_direction'].to_numpy()
            else:
                df_features = df[available_cols].values
                df_targets = df['target_direction'].values
            
            # Create sequences
            sequences_this_day = 0
            for i in range(lookback, len(df) - slope_window):
                seq = df_features[i - lookback:i]
                target = df_targets[i]
                
                # Validate sequence and target
                if not np.isnan(target) and not np.isnan(seq).any():
                    if sequence_idx >= max_sequences_estimate:
                        print(f"\nWARNING: Reached maximum sequence estimate at day {day_num}!")
                        break
                    
                    X_memmap[sequence_idx] = seq
                    y_memmap[sequence_idx] = target
                    sequence_idx += 1
                    sequences_this_day += 1
            
            # Progress update
            day_elapsed = time.time() - day_start_time
            print(f"  Day {day_num} ({day_idx+1}/{len(day_numbers)}): "
                  f"+{sequences_this_day:,} sequences | "
                  f"Total: {sequence_idx:,} | "
                  f"Time: {day_elapsed:.1f}s")
            
            # Cleanup
            del df, ddf
            gc.collect()
        
        # Trim arrays to actual size
        print(f"\nFinalizing {sequence_idx:,} sequences...")
        X_final = X_memmap[:sequence_idx].copy()  # Create copy before deleting
        y_final = y_memmap[:sequence_idx].copy()  # Create copy before deleting
        
        # CRITICAL FIX: Delete memmap objects and flush before removing files
        del X_memmap, y_memmap
        gc.collect()  # Force garbage collection
        
        # Small delay to ensure file handles are released (Windows issue)
        time.sleep(0.1)
        
        # Now save as numpy arrays
        np.save(f"{self.config.MODEL_DIR}/{output_prefix}_X.npy", X_final)
        np.save(f"{self.config.MODEL_DIR}/{output_prefix}_y.npy", y_final)
        
        # Try to cleanup memory-mapped files with retry logic
        max_retries = 3
        for retry in range(max_retries):
            try:
                if os.path.exists(X_memmap_path):
                    os.remove(X_memmap_path)
                if os.path.exists(y_memmap_path):
                    os.remove(y_memmap_path)
                break
            except PermissionError:
                if retry < max_retries - 1:
                    print(f"  Retrying file cleanup (attempt {retry + 1}/{max_retries})...")
                    time.sleep(0.5)
                    gc.collect()
                else:
                    print(f"  Warning: Could not remove temporary files (will be cleaned on next run)")
        
        gc.collect()
        
        print(f"Saved {sequence_idx:,} sequences to {output_prefix}")
        return sequence_idx
    
    def prepare_training_data(self, train_days: List[int], val_days: List[int], test_days: List[int]):
        print("\n" + "="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
        # Compute statistics only on training data
        if not self.stats_computed:
            print("\nStep 1: Computing feature statistics...")
            self.compute_feature_statistics(train_days)
        
        print("\nStep 2: Processing training data...")
        train_count = self.process_and_save_sequences(train_days, "train")
        
        print("\nStep 3: Processing validation data...")
        val_count = self.process_and_save_sequences(val_days, "val")
        
        print("\nStep 4: Processing test data...")
        test_count = self.process_and_save_sequences(test_days, "test")
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"Training sequences:   {train_count:,}")
        print(f"Validation sequences: {val_count:,}")
        print(f"Test sequences:       {test_count:,}")
        print(f"Total sequences:      {train_count + val_count + test_count:,}")
        print("="*60)
        
        return train_count, val_count, test_count

def main():
    config = Config()
    processor = StreamingDataProcessor(config)
    train_days = list(range(0, 10))
    val_days = list(range(10, 12))
    test_days = list(range(12, 15))
    processor.prepare_training_data(train_days, val_days, test_days)

if __name__ == "__main__":
    main()