import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import gc
from datetime import datetime
from pathlib import Path

from config import Config
from tft_model_enhanced import TemporalFusionTransformer, TradingDataset
from trainer_enhanced import EnhancedTFTTrainer


class OptunaHyperparameterTuner:
    """
    OPTIMIZED: Hyperparameter tuning with better memory management
    Optimizes model for slope-based prediction task
    """
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.train_X_path = f"{config.MODEL_DIR}/train_X.npy"
        self.train_y_path = f"{config.MODEL_DIR}/train_y.npy"
        self.val_X_path = f"{config.MODEL_DIR}/val_X.npy"
        self.val_y_path = f"{config.MODEL_DIR}/val_y.npy"
        self.study_path = f"{config.OPTUNA_DIR}/study.db"
        
        print("="*60)
        print("OPTUNA HYPERPARAMETER TUNER INITIALIZED")
        print("="*60)
        print(f"Device: {device}")
        print(f"Study path: {self.study_path}")
        print(f"Task: {config.SLOPE_PREDICTION_WINDOW}s slope prediction")
        print("="*60 + "\n")
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization
        Memory-efficient trial execution
        """
        # Sample hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', self.config.HP_HIDDEN_SIZE)
        lstm_layers = trial.suggest_categorical('lstm_layers', self.config.HP_LSTM_LAYERS)
        attention_heads = trial.suggest_categorical('attention_heads', self.config.HP_ATTENTION_HEADS)
        dropout = trial.suggest_categorical('dropout', self.config.HP_DROPOUT)
        learning_rate = trial.suggest_categorical('learning_rate', self.config.HP_LEARNING_RATE)
        batch_size = trial.suggest_categorical('batch_size', self.config.HP_BATCH_SIZE)
        
        ffn_hidden_size = hidden_size * 2
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}:")
        print(f"{'='*60}")
        print(f"  hidden_size={hidden_size}, lstm_layers={lstm_layers}")
        print(f"  attention_heads={attention_heads}, dropout={dropout}")
        print(f"  learning_rate={learning_rate}, batch_size={batch_size}")
        print(f"{'='*60}\n")
        
        try:
            # Create model
            model = TemporalFusionTransformer(
                num_features=self.config.get_num_features(),
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                num_attention_heads=attention_heads,
                dropout=dropout,
                ffn_hidden_size=ffn_hidden_size,
                num_classes=3
            )
            
            # Load datasets with memory mapping
            train_dataset = TradingDataset(self.train_X_path, self.train_y_path, mmap_mode=True)
            val_dataset = TradingDataset(self.val_X_path, self.val_y_path, mmap_mode=True)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.config.USE_GPU else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.config.USE_GPU else False
            )
            
            # Create trial config
            trial_config = self.create_trial_config(
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            # Train model
            trainer = EnhancedTFTTrainer(model, trial_config, self.device)
            
            max_epochs = 15  # Reduced for faster tuning
            best_val_loss = float('inf')
            
            for epoch in range(max_epochs):
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc, _ = trainer.validate(val_loader)
                
                # Report to Optuna
                trial.report(val_loss, epoch)
                
                # Check for pruning
                if trial.should_prune():
                    print(f"  ✗ Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
                
                # Track best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Early stopping if not improving
                if epoch > 5 and val_loss > best_val_loss * 1.2:
                    print(f"  ⚠️  Early stopping at epoch {epoch+1}")
                    break
                
                # Memory cleanup
                if epoch % 3 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"  ✓ Best val loss: {best_val_loss:.4f}")
            
            # Cleanup
            del model, trainer, train_loader, val_loader, train_dataset, val_dataset
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return best_val_loss
            
        except optuna.TrialPruned:
            # Cleanup before re-raising
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            raise
            
        except Exception as e:
            print(f"  ✗ Trial failed with error: {str(e)}")
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return float('inf')
    
    def create_trial_config(self, learning_rate: float, batch_size: int):
        """
        Create a configuration object for the trial
        """
        class TrialConfig(Config):
            LEARNING_RATE = learning_rate
            BATCH_SIZE = batch_size
            MAX_EPOCHS = 15
            LOG_INTERVAL = 200
            VERBOSE = False
        return TrialConfig()
    
    def run_optimization(self, n_trials: int = None, timeout: int = None):
        """
        Run hyperparameter optimization with Optuna
        """
        if n_trials is None:
            n_trials = self.config.OPTUNA_N_TRIALS
        if timeout is None:
            timeout = self.config.OPTUNA_TIMEOUT
        
        print("="*60)
        print("STARTING HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout if timeout else 'None'}")
        print("="*60 + "\n")
        
        # Create study
        study = optuna.create_study(
            study_name=f"tft_slope_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction='minimize',
            sampler=TPESampler(seed=self.config.RANDOM_SEED),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            ),
            storage=f'sqlite:///{self.study_path}',
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config.OPTUNA_N_JOBS,
            show_progress_bar=True,
            gc_after_trial=True  # Automatic garbage collection
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (val_loss): {trial.value:.4f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("="*60 + "\n")
        
        # Save results
        self.save_results(study)
        
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return study
    
    def save_results(self, study: optuna.Study):
        """
        Save optimization results to files
        """
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'best_params': best_params,
            'best_value': float(best_value),
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat(),
            'prediction_task': f'{self.config.SLOPE_PREDICTION_WINDOW}s_slope'
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = f"{self.config.OPTUNA_DIR}/best_params_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Best parameters saved to: {json_path}")
        
        # Save all trials
        df = study.trials_dataframe()
        csv_path = f"{self.config.OPTUNA_DIR}/all_trials_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ All trials saved to: {csv_path}")
    
    def load_best_params(self, json_path: str = None) -> dict:
        """
        Load best parameters from a JSON file
        """
        if json_path is None:
            json_files = sorted(Path(self.config.OPTUNA_DIR).glob("best_params_*.json"))
            if not json_files:
                raise FileNotFoundError("No best_params files found")
            json_path = str(json_files[-1])
        
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        print(f"✓ Loaded parameters from: {json_path}")
        return results['best_params']
    
    def create_model_from_best_params(self, json_path: str = None):
        """
        Create a model using the best parameters
        """
        best_params = self.load_best_params(json_path)
        
        print("\n" + "="*60)
        print("Creating model with best parameters:")
        print("="*60)
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
        
        model = TemporalFusionTransformer(
            num_features=self.config.get_num_features(),
            hidden_size=best_params['hidden_size'],
            lstm_layers=best_params['lstm_layers'],
            num_attention_heads=best_params['attention_heads'],
            dropout=best_params['dropout'],
            ffn_hidden_size=best_params['hidden_size'] * 2,
            num_classes=3
        )
        
        return model, best_params


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tuner = OptunaHyperparameterTuner(config, device)
    study = tuner.run_optimization(n_trials=50)
    
    best_model, best_params = tuner.create_model_from_best_params()
    print("\n✓ Best model created successfully!")
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()