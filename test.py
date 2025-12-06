"""
Quick Test Script - Test Complete Workflow on Single Day
=========================================================

This script tests the entire workflow on minimal data:
- Converts 1 CSV to Parquet
- Prepares data from 3 days (1 train, 1 val, 1 test)
- Trains model for 3 epochs
- Backtests on 1 day

Use this to verify everything works before running the full pipeline.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_converter import CSVToParquetConverter
from data_processor_streaming import StreamingDataProcessor
from tft_model_enhanced import TemporalFusionTransformer, TradingDataset
from trainer_enhanced import EnhancedTFTTrainer
from trading_strategy_enhanced import EnhancedTFTTradingStrategy, PerformanceEvaluator


class QuickTestConfig(Config):
    """Test configuration with minimal settings"""
    # Reduce model size for faster testing
    HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    ATTENTION_HEADS = 4
    BATCH_SIZE = 256
    MAX_EPOCHS = 3
    
    # Faster feature statistics
    FEATURE_STATS_SAMPLE_RATE = 0.02  # Only 2%
    FEATURE_STATS_MAX_DAYS = 1  # Just 1 day for stats
    
    # Quick training
    EARLY_STOPPING_PATIENCE = 2
    LOG_INTERVAL = 50


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")


def test_step1_conversion(config, test_day=0):
    """Test: Convert single CSV to Parquet"""
    print_header("TEST STEP 1: CSV TO PARQUET")
    
    converter = CSVToParquetConverter(config)
    success = converter.convert_single_file(test_day, verbose=True)
    
    if success:
        print(f"✓ Conversion successful for day {test_day}")
        # Verify
        converter.verify_conversion(test_day)
    else:
        print(f"✗ Conversion failed for day {test_day}")
        return False
    
    del converter
    gc.collect()
    return True


def test_step2_data_prep(config, train_day=0, val_day=1, test_day=2):
    """Test: Prepare minimal dataset"""
    print_header("TEST STEP 2: DATA PREPARATION")
    
    processor = StreamingDataProcessor(config)
    
    train_days = [train_day]
    val_days = [val_day]
    test_days = [test_day]
    
    print(f"Using days: train={train_day}, val={val_day}, test={test_day}")
    
    try:
        train_count, val_count, test_count = processor.prepare_training_data(
            train_days, val_days, test_days
        )
        
        print(f"\n✓ Data preparation successful!")
        print(f"  Training sequences:   {train_count:,}")
        print(f"  Validation sequences: {val_count:,}")
        print(f"  Test sequences:       {test_count:,}")
        
        del processor
        gc.collect()
        return True
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step3_training(config, device):
    """Test: Train model for few epochs"""
    print_header("TEST STEP 3: MODEL TRAINING")
    
    try:
        # Create small model
        model = TemporalFusionTransformer(
            num_features=config.get_num_features(),
            hidden_size=config.HIDDEN_SIZE,
            lstm_layers=config.LSTM_LAYERS,
            num_attention_heads=config.ATTENTION_HEADS,
            dropout=config.DROPOUT,
            ffn_hidden_size=config.HIDDEN_SIZE * 2
        )
        
        print("Loading datasets...")
        train_dataset = TradingDataset(
            f"{config.MODEL_DIR}/train_X.npy",
            f"{config.MODEL_DIR}/train_y.npy",
            mmap_mode=True
        )
        val_dataset = TradingDataset(
            f"{config.MODEL_DIR}/val_X.npy",
            f"{config.MODEL_DIR}/val_y.npy",
            mmap_mode=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=config.USE_GPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=config.USE_GPU
        )
        
        print(f"Training for {config.MAX_EPOCHS} epochs...")
        trainer = EnhancedTFTTrainer(model, config, device)
        trainer.train(train_loader, val_loader, config.MAX_EPOCHS)
        
        print(f"\n✓ Training completed!")
        print(f"  Best val loss: {trainer.best_val_loss:.4f}")
        print(f"  Best val acc:  {trainer.best_val_acc:.2f}%")
        
        # Cleanup
        del trainer, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return model
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_step4_backtest(config, model, device, test_day=2):
    """Test: Backtest on single day"""
    print_header("TEST STEP 4: BACKTESTING")
    
    try:
        strategy = EnhancedTFTTradingStrategy(model, config, device)
        evaluator = PerformanceEvaluator(config)
        processor = StreamingDataProcessor(config)
        
        # Need to load feature statistics
        if not processor.stats_computed:
            print("Loading feature statistics...")
            processor.compute_feature_statistics([0])  # Use day 0 for stats
        
        # Load and process test day
        print(f"Loading day {test_day}...")
        ddf = processor.load_parquet_lazy(test_day)
        if ddf is None:
            print(f"✗ Could not load day {test_day}")
            return False
        
        ddf = processor.filter_stable_period(ddf)
        ddf = ddf.map_partitions(processor.normalize_partition)
        df = ddf.compute()
        
        print(f"Running strategy on {len(df):,} data points...")
        trades = strategy.run_day(df, test_day)
        
        print(f"\n✓ Backtesting completed!")
        print(f"  Trades executed: {len(trades)}")
        
        if trades:
            # Calculate metrics
            results = evaluator.calculate_metrics(trades, 1)
            
            # Print summary
            print("\nQuick Results:")
            print(f"  Win rate:      {results['win_rate']*100:.1f}%")
            print(f"  Total return:  {results['total_return']*100:+.2f}%")
            print(f"  Avg win:       {results['avg_win']*100:+.2f}%")
            print(f"  Avg loss:      {results['avg_loss']*100:.2f}%")
            print(f"  Profit factor: {results['profit_factor']:.2f}")
            
            # Show sample trades
            print("\nSample Trades:")
            for i, trade in enumerate(trades[:3]):
                print(f"  Trade {i+1}: {trade['position']:+d} | "
                      f"P&L: {trade['pnl_pct']*100:+.2f}% | "
                      f"Hold: {trade['holding_time']:.0f}s | "
                      f"Exit: {trade['exit_reason']}")
        else:
            print("  No trades executed (model may need more training)")
        
        # Strategy stats
        stats = strategy.get_statistics()
        print(f"\nPrediction Statistics:")
        print(f"  Total predictions:     {stats['total_predictions']:,}")
        print(f"  Confident predictions: {stats['confident_predictions']:,}")
        print(f"  Confidence rate:       {stats['confidence_rate']*100:.1f}%")
        
        # Cleanup
        del strategy, processor, df, ddf
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"✗ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """Run complete quick test"""
    print("\n" + "="*60)
    print("QUICK TEST - COMPLETE WORKFLOW".center(60))
    print("="*60)
    print("\nThis will test the entire pipeline on minimal data:")
    print("  - Convert 1 CSV file")
    print("  - Prepare data from 3 days")
    print("  - Train model for 3 epochs")
    print("  - Backtest on 1 day")
    print("\nEstimated time: 2-5 minutes")
    print("="*60)
    
    # Initialize
    config = QuickTestConfig()
    device = torch.device('cuda' if config.USE_GPU and torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test days to use
    test_days = {
        'convert': [0, 1, 2],  # Convert 3 days
        'train': 0,
        'val': 1,
        'test': 2
    }
    
    try:
        # Step 1: Convert CSV files
        print("\nConverting test files...")
        for day in test_days['convert']:
            if not test_step1_conversion(config, day):
                print(f"\n❌ Test failed at conversion step (day {day})")
                return False
        
        # Step 2: Prepare data
        if not test_step2_data_prep(
            config, 
            test_days['train'], 
            test_days['val'], 
            test_days['test']
        ):
            print("\n❌ Test failed at data preparation step")
            return False
        
        # Step 3: Train model
        model = test_step3_training(config, device)
        if model is None:
            print("\n❌ Test failed at training step")
            return False
        
        model = model.to(device)
        model.eval()
        
        # Step 4: Backtest
        if not test_step4_backtest(config, model, device, test_days['test']):
            print("\n❌ Test failed at backtesting step")
            return False
        
        # Success!
        print_header("✓ ALL TESTS PASSED!")
        print("The complete workflow is functioning correctly.")
        print("\nYou can now run the full pipeline with:")
        print("  python main_enhanced.py")
        print("\nOr with specific options:")
        print("  python main_enhanced.py --skip-conversion")
        print("  python main_enhanced.py --skip-tuning")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_test_files():
    """Clean up test files (optional)"""
    print("\nCleaning up test files...")
    
    config = Config()
    
    # Files to remove
    test_files = [
        f"{config.MODEL_DIR}/train_X.npy",
        f"{config.MODEL_DIR}/train_y.npy",
        f"{config.MODEL_DIR}/val_X.npy",
        f"{config.MODEL_DIR}/val_y.npy",
        f"{config.MODEL_DIR}/test_X.npy",
        f"{config.MODEL_DIR}/test_y.npy",
        f"{config.MODEL_DIR}/best_tft_model.pt",
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  Removed: {file_path}")
            except Exception as e:
                print(f"  Could not remove {file_path}: {e}")
    
    print("Cleanup complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test of complete workflow')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up test files after testing')
    args = parser.parse_args()
    
    # Run test
    success = run_quick_test()
    
    # Optional cleanup
    if args.cleanup and success:
        response = input("\nDo you want to clean up test files? (y/n): ")
        if response.lower() == 'y':
            cleanup_test_files()
    
    # Exit with appropriate code
    exit(0 if success else 1)