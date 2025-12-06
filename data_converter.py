import os
import gc
from pathlib import Path
import numpy as np
try:
    import cudf
    import dask_cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    import dask.dataframe as dd
    CUDF_AVAILABLE = False
    print("WARNING: cuDF not available, falling back to pandas")

from config import Config

class CSVToParquetConverter:
    """
    OPTIMIZED: Enhanced memory management for large CSV conversions
    """
    def __init__(self, config: Config):
        self.config = config
        self.use_gpu = config.USE_GPU and CUDF_AVAILABLE
        print(f"Converter initialized with GPU: {self.use_gpu}")
    
    def convert_single_file(self, day_num: int, verbose: bool = True):
        """
        Convert a single CSV file to Parquet format with memory-efficient processing
        """
        csv_path = f"{self.config.DATA_DIR}/day{day_num}.csv"
        parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
        
        if not os.path.exists(csv_path):
            if verbose:
                print(f"File not found: {csv_path}")
            return False
        
        if os.path.exists(parquet_path):
            if verbose:
                print(f"Already exists: {parquet_path} (skipping)")
            return True
        
        try:
            if verbose:
                print(f"Converting day{day_num}.csv to parquet...")
            
            # Use appropriate chunk size based on available memory
            chunk_size_kb = self.config.CHUNK_SIZE
            
            if self.use_gpu:
                ddf = dask_cudf.read_csv(
                    csv_path,
                    blocksize=f'{chunk_size_kb}KB',
                    dtype={'Time': 'str', 'Price': 'float32'}
                )
            else:
                ddf = dd.read_csv(
                    csv_path,
                    blocksize=f'{chunk_size_kb}KB',
                    dtype={'Time': 'str', 'Price': 'float32'},
                    engine='python',  # More robust for large files
                    on_bad_lines='skip'  # Skip problematic lines
                )
            
            # Add day column
            ddf['day'] = day_num
            
            # Write to Parquet with optimized settings
            ddf.to_parquet(
                parquet_path,
                engine='pyarrow',
                compression='snappy',
                write_index=False,
                row_group_size=self.config.PARQUET_ROW_GROUP_SIZE,
                write_metadata_file=False  # Reduce overhead
            )
            
            if verbose:
                size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                print(f"  ✓ day{day_num} converted ({size_mb:.1f} MB)")
            
            # Explicit cleanup
            del ddf
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error converting day{day_num}: {str(e)}")
            # Remove partial file if exists
            if os.path.exists(parquet_path):
                try:
                    os.remove(parquet_path)
                except:
                    pass
            return False
    
    def convert_all_files(self, start_day: int = 0, end_day: int = 279):
        """
        Convert all CSV files in the specified range
        """
        print("="*60)
        print("CONVERTING CSV FILES TO PARQUET")
        print("="*60)
        print(f"Range: day{start_day} to day{end_day-1}")
        print(f"GPU Acceleration: {self.use_gpu}")
        print("="*60 + "\n")
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        for day_num in range(start_day, end_day):
            # Check if already exists before attempting conversion
            parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
            if os.path.exists(parquet_path):
                skip_count += 1
                if day_num % 10 == 0:  # Print progress every 10 files
                    print(f"Day {day_num}: already exists (skipped)")
                continue
            
            success = self.convert_single_file(day_num, verbose=True)
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            # Periodic garbage collection for long runs
            if (day_num - start_day + 1) % 20 == 0:
                gc.collect()
        
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Successful:  {success_count:3d}")
        print(f"Skipped:     {skip_count:3d}")
        print(f"Failed:      {fail_count:3d}")
        print(f"Total:       {end_day - start_day:3d}")
        print("="*60 + "\n")
        
        return success_count, fail_count
    
    def verify_conversion(self, day_num: int):
        """
        Verify that a converted Parquet file can be read
        """
        parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
        
        try:
            if self.use_gpu:
                ddf = dask_cudf.read_parquet(parquet_path)
            else:
                ddf = dd.read_parquet(parquet_path)
            
            nrows = len(ddf)
            ncols = len(ddf.columns)
            
            print(f"✓ day{day_num}: {nrows:,} rows, {ncols} columns")
            print(f"  Columns: {list(ddf.columns[:10])}...")
            
            # Cleanup
            del ddf
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"✗ Verification failed for day{day_num}: {str(e)}")
            return False
    
    def verify_all_files(self, start_day: int = 0, end_day: int = 279, sample_every: int = 50):
        """
        Verify a sample of converted files
        """
        print("\n" + "="*60)
        print("VERIFYING PARQUET FILES")
        print("="*60)
        
        sample_days = list(range(start_day, end_day, sample_every))
        verified = 0
        failed = 0
        
        for day_num in sample_days:
            if self.verify_conversion(day_num):
                verified += 1
            else:
                failed += 1
        
        print("\n" + "="*60)
        print(f"Verified: {verified}/{len(sample_days)} files")
        if failed > 0:
            print(f"⚠️  Failed: {failed} files")
        else:
            print("✓ All sampled files verified successfully")
        print("="*60 + "\n")
        
        return failed == 0


def main():
    config = Config()
    converter = CSVToParquetConverter(config)
    
    # Convert all files
    converter.convert_all_files(start_day=0, end_day=279)
    
    # Verify sample files
    print("\nVerifying sample files...")
    converter.verify_all_files(start_day=0, end_day=279, sample_every=50)


if __name__ == "__main__":
    main()