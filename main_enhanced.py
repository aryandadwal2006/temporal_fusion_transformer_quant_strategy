import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

from config import Config
from data_converter import CSVToParquetConverter
from data_processor_streaming import StreamingDataProcessor
from tft_model_enhanced import TemporalFusionTransformer, TradingDataset, count_parameters
from trainer_enhanced import EnhancedTFTTrainer
from trading_strategy_enhanced import EnhancedTFTTradingStrategy, PerformanceEvaluator
from hyperparameter_tuning import OptunaHyperparameterTuner


def setup_environment(config: Config):
    """Setup random seeds and environment for reproducibility"""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if config.USE_GPU and torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = config.DETERMINISTIC
        torch.backends.cudnn.benchmark = not config.DETERMINISTIC
    
    print("\n" + "="*60)
    print("ENVIRONMENT SETUP")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"CUDA Available: No (using CPU)")
    print("="*60 + "\n")


def step1_convert_to_parquet(config: Config):
    """Step 1: Convert CSV files to Parquet format"""
    print("\n" + "="*60)
    print("STEP 1: CONVERT CSV TO PARQUET")
    print("="*60)
    converter = CSVToParquetConverter(config)
    success, failed = converter.convert_all_files(start_day=0, end_day=279)
    print(f"\nConversion complete: {success} successful, {failed} failed")
    print("="*60 + "\n")
    return success > 0


def step2_prepare_data(config: Config, train_end: int = 195, val_end: int = 237):
    """
    Step 2: Prepare training data with slope-based targets
    MODIFIED: Now uses slope prediction instead of returns
    """
    print("\n" + "="*60)
    print("STEP 2: PREPARE DATA (SLOPE-BASED PREDICTION)")
    print("="*60)
    processor = StreamingDataProcessor(config)
    
    train_days = list(range(0, train_end))
    val_days = list(range(train_end, val_end))
    test_days = list(range(val_end, 279))
    
    print(f"Training days:   {len(train_days):3d} (days 0-{train_end-1})")
    print(f"Validation days: {len(val_days):3d} (days {train_end}-{val_end-1})")
    print(f"Test days:       {len(test_days):3d} (days {val_end}-278)")
    print(f"\nSlope prediction window: {config.SLOPE_PREDICTION_WINDOW} seconds")
    
    train_count, val_count, test_count = processor.prepare_training_data(
        train_days, val_days, test_days
    )
    
    print(f"\nData preparation complete")
    print("="*60 + "\n")
    
    # Cleanup
    del processor
    gc.collect()
    
    return train_count > 0


def step3_hyperparameter_tuning(config: Config, device: torch.device, n_trials: int = 50):
    """Step 3: Hyperparameter tuning with Optuna"""
    print("\n" + "="*60)
    print("STEP 3: HYPERPARAMETER TUNING")
    print("="*60)
    tuner = OptunaHyperparameterTuner(config, device)
    study = tuner.run_optimization(n_trials=n_trials)
    print(f"\nHyperparameter tuning complete")
    print("="*60 + "\n")
    
    # Cleanup
    del tuner
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return study


def step4_train_model(config: Config, device: torch.device, use_best_params: bool = True):
    """Step 4: Train the TFT model"""
    print("\n" + "="*60)
    print("STEP 4: TRAIN MODEL")
    print("="*60)
    
    # Load model with best parameters if available
    if use_best_params:
        try:
            tuner = OptunaHyperparameterTuner(config, device)
            model, best_params = tuner.create_model_from_best_params()
            config.LEARNING_RATE = best_params['learning_rate']
            config.BATCH_SIZE = best_params['batch_size']
            print("\n✓ Using optimized hyperparameters")
            del tuner
        except Exception as e:
            print(f"\n⚠ Could not load best params: {e}")
            print("Using default configuration")
            model = TemporalFusionTransformer(
                num_features=config.get_num_features(),
                hidden_size=config.HIDDEN_SIZE,
                lstm_layers=config.LSTM_LAYERS,
                num_attention_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT,
                ffn_hidden_size=config.FFN_HIDDEN_SIZE
            )
    else:
        model = TemporalFusionTransformer(
            num_features=config.get_num_features(),
            hidden_size=config.HIDDEN_SIZE,
            lstm_layers=config.LSTM_LAYERS,
            num_attention_heads=config.ATTENTION_HEADS,
            dropout=config.DROPOUT,
            ffn_hidden_size=config.FFN_HIDDEN_SIZE
        )
    
    print("\nLoading datasets...")
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
    
    # Train model
    trainer = EnhancedTFTTrainer(model, config, device)
    trainer.train(train_loader, val_loader, config.MAX_EPOCHS)
    
    # Load best checkpoint
    checkpoint_path = f"{config.MODEL_DIR}/best_tft_model.pt"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Best model loaded from checkpoint")
    
    print(f"\nModel training complete")
    print("="*60 + "\n")
    
    # Cleanup
    del trainer, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model


def step5_backtest_strategy(config: Config, model, device: torch.device):
    """
    Step 5: Backtest the trading strategy
    MODIFIED: Now uses slope-based predictions
    """
    print("\n" + "="*60)
    print("STEP 5: BACKTEST STRATEGY (SLOPE-BASED)")
    print("="*60)
    
    strategy = EnhancedTFTTradingStrategy(model, config, device)
    evaluator = PerformanceEvaluator(config)
    processor = StreamingDataProcessor(config)
    
    # Load feature statistics (needed for normalization)
    if not processor.stats_computed:
        print("Loading feature statistics for normalization...")
        train_days = list(range(0, 195))
        processor.compute_feature_statistics(train_days)
    
    test_days = list(range(237, 279))
    all_trades = []
    
    print(f"\nBacktesting on {len(test_days)} days...")
    print(f"Slope prediction window: {config.SLOPE_PREDICTION_WINDOW} seconds\n")
    
    for idx, day_num in enumerate(test_days):
        print(f"\rProcessing day {day_num} ({idx+1}/{len(test_days)})...", end='', flush=True)
        
        ddf = processor.load_parquet_lazy(day_num)
        if ddf is None:
            continue
        
        ddf = processor.filter_stable_period(ddf)
        ddf = ddf.map_partitions(processor.normalize_partition)
        df = ddf.compute()
        
        # Run strategy on this day
        day_trades = strategy.run_day(df, day_num)
        all_trades.extend(day_trades)
        
        # Cleanup
        del df, ddf
        gc.collect()
    
    print(f"\n\n✓ Backtesting complete: {len(all_trades)} trades executed")
    
    # Calculate and print results
    results = evaluator.calculate_metrics(all_trades, len(test_days))
    evaluator.print_results(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{config.RESULTS_DIR}/backtest_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json_results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results.items()}
        json.dump(json_results, f, indent=4)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save trades
    import pandas as pd
    trades_df = pd.DataFrame(all_trades)
    trades_file = f"{config.RESULTS_DIR}/trades_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"✓ Trades saved to: {trades_file}")
    
    # Print strategy statistics
    stats = strategy.get_statistics()
    print(f"\n" + "="*60)
    print("STRATEGY STATISTICS")
    print("="*60)
    print(f"Total predictions:      {stats['total_predictions']:,}")
    print(f"Confident predictions:  {stats['confident_predictions']:,}")
    print(f"Confidence rate:        {stats['confidence_rate']*100:.2f}%")
    print("="*60 + "\n")
    
    # Cleanup
    del strategy, processor
    gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='TFT Trading Strategy - Slope-Based Prediction')
    parser.add_argument('--skip-conversion', action='store_true', 
                       help='Skip CSV to Parquet conversion')
    parser.add_argument('--skip-data-prep', action='store_true', 
                       help='Skip data preparation')
    parser.add_argument('--skip-tuning', action='store_true', 
                       help='Skip hyperparameter tuning')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training')
    parser.add_argument('--tuning-trials', type=int, default=50, 
                       help='Number of Optuna trials')
    parser.add_argument('--train-end', type=int, default=195, 
                       help='Last day for training (exclusive)')
    parser.add_argument('--val-end', type=int, default=237, 
                       help='Last day for validation (exclusive)')
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.print_config()
    
    # Setup environment
    setup_environment(config)
    
    device = torch.device('cuda' if config.USE_GPU and torch.cuda.is_available() else 'cpu')
    
    try:
        # Step 1: Convert CSV to Parquet
        if not args.skip_conversion:
            if not step1_convert_to_parquet(config):
                print("\n❌ Conversion failed!")
                return
        
        # Step 2: Prepare data with slope-based targets
        if not args.skip_data_prep:
            if not step2_prepare_data(config, args.train_end, args.val_end):
                print("\n❌ Data preparation failed!")
                return
        
        # Step 3: Hyperparameter tuning
        if not args.skip_tuning:
            step3_hyperparameter_tuning(config, device, args.tuning_trials)
        
        # Step 4: Train model
        if not args.skip_training:
            model = step4_train_model(config, device, use_best_params=not args.skip_tuning)
        else:
            print("\n" + "="*60)
            print("Loading existing model...")
            print("="*60)
            checkpoint = torch.load(f"{config.MODEL_DIR}/best_tft_model.pt", 
                                   map_location=device)
            model = TemporalFusionTransformer(
                num_features=config.get_num_features(),
                hidden_size=config.HIDDEN_SIZE,
                lstm_layers=config.LSTM_LAYERS,
                num_attention_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT,
                ffn_hidden_size=config.FFN_HIDDEN_SIZE
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            print("✓ Model loaded")
            print("="*60 + "\n")
        
        # Step 5: Backtest strategy
        results = step5_backtest_strategy(config, model, device)
        
        # Final summary
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)
        print(f"Annual Return:    {results['annual_return']*100:+.2f}%")
        print(f"Max Drawdown:     {results['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
        print(f"Calmar Ratio:     {results['calmar_ratio']:.3f}")
        print(f"Win Rate:         {results['win_rate']*100:.2f}%")
        print(f"Total Trades:     {results['total_trades']:,}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()