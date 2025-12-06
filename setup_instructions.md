# Setup Instructions - Windows

## 📋 Prerequisites

Before you begin, make sure you have:

1. **Python 3.8 or higher** installed
   - Download from: https://www.python.org/downloads/
   - ✅ During installation, check "Add Python to PATH"

2. **Git Bash** installed (for running .sh scripts)
   - Download from: https://git-scm.com/download/win
   - Or use WSL (Windows Subsystem for Linux)

3. **NVIDIA GPU** (optional but recommended)
   - With latest drivers installed
   - CUDA 11.8 or higher

4. **At least 16GB RAM** recommended

5. **Your data files** in `D:\tft-quant-strategy-main\EBY\`
   - Expected: `day0.csv`, `day1.csv`, ..., `day278.csv`

---

## 🚀 Quick Setup (Recommended)

### Method 1: Using Git Bash (Easiest)

1. **Open Git Bash** in your project directory:
   ```bash
   cd /d/tft-quant-strategy-main
   ```

2. **Run the setup script**:
   ```bash
   bash setup.sh
   ```

3. **Follow the prompts** - it will:
   - ✅ Check Python and dependencies
   - ✅ Create virtual environment
   - ✅ Install all required packages
   - ✅ Verify installation
   - ✅ Optionally run a quick test

4. **Done!** The script will tell you when everything is ready.

### Method 2: Using Windows CMD/PowerShell (Manual)

1. **Open Command Prompt** as Administrator

2. **Navigate to project**:
   ```cmd
   D:
   cd D:\tft-quant-strategy-main
   ```

3. **Create virtual environment**:
   ```cmd
   python -m venv venv
   ```

4. **Activate virtual environment**:
   ```cmd
   venv\Scripts\activate
   ```

5. **Upgrade pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

6. **Install PyTorch** (with CUDA if you have GPU):
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   Or without CUDA (CPU only):
   ```cmd
   pip install torch torchvision torchaudio
   ```

7. **Install other dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

8. **Verify installation**:
   ```cmd
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

---

## 🎮 Running the Project

### Option 1: Using the Interactive Menu (Easiest)

**Double-click** `run.bat` or run in CMD:
```cmd
run.bat
```

This gives you a menu:
```
1. Run Quick Test (2-5 minutes)
2. Run Full Pipeline (several hours)
3. Run Full Pipeline (skip conversion)
4. Run Full Pipeline (skip tuning)
5. Run Full Pipeline (skip training, backtest only)
6. Open Python Shell
7. Check GPU Status
8. Exit
```

### Option 2: Command Line

**Quick test** (recommended first):
```cmd
python test.py
```

**Full pipeline**:
```cmd
python main_enhanced.py
```

**With options**:
```cmd
python main_enhanced.py --skip-conversion
python main_enhanced.py --skip-tuning
python main_enhanced.py --skip-training
```

---

## 📁 Project Structure

After setup, your directory should look like:

```
D:\tft-quant-strategy-main\
│
├── EBY\                          # Your CSV data files
│   ├── day0.csv
│   ├── day1.csv
│   └── ...
│
├── venv\                         # Virtual environment (created by setup)
│
├── models\                       # Saved models (created automatically)
├── results\                      # Results and trades (created automatically)
├── optuna_studies\               # Hyperparameter tuning (created automatically)
├── cache\                        # Temporary files (created automatically)
├── EBY_parquet\                  # Converted data (created automatically)
│
├── config.py                     # Configuration
├── data_converter.py             # CSV to Parquet converter
├── data_processor_streaming.py   # Data preprocessing
├── tft_model_enhanced.py         # Model architecture
├── trainer_enhanced.py           # Training logic
├── trading_strategy_enhanced.py  # Trading strategy
├── hyperparameter_tuning.py      # Optuna tuning
├── main_enhanced.py              # Main pipeline
├── test.py                       # Quick test script
│
├── requirements.txt              # Python dependencies
├── setup.sh                      # Setup script (for Git Bash)
├── run.bat                       # Windows runner script
├── SETUP_INSTRUCTIONS.md         # This file
└── MODIFICATIONS_SUMMARY.md      # Detailed changes documentation
```

---

## 🔧 Troubleshooting

### "Python not found"
- Make sure Python is installed and in PATH
- Try: `python --version`
- Reinstall Python and check "Add to PATH"

### "Virtual environment activation failed"
- On Windows CMD: `venv\Scripts\activate.bat`
- On PowerShell: `venv\Scripts\Activate.ps1`
- On Git Bash: `source venv/Scripts/activate`

### "CUDA not available" (but you have GPU)
- Update NVIDIA drivers
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA:
  ```cmd
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### "Out of Memory" errors
- Reduce `BATCH_SIZE` in `config.py` (try 256 or 128)
- Reduce `FEATURE_STATS_SAMPLE_RATE` to 0.01
- Close other applications
- Use CPU instead of GPU (slower but more stable)

### "Module not found" errors
- Make sure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

### "Data files not found"
- Make sure CSV files are in `EBY\` directory
- Check filenames: must be `day0.csv`, `day1.csv`, etc.

### Setup script fails
- Try manual installation (Method 2 above)
- Check Python version: `python --version` (need 3.8+)
- Check pip: `pip --version`

---

## 📊 Expected Timeline

With a modern GPU (RTX 3080 or better):

| Step | Time | What It Does |
|------|------|--------------|
| Quick Test | 2-5 min | Tests on 3 days |
| CSV Conversion | 10-30 min | Converts 279 CSV files |
| Data Preparation | 30-60 min | Creates sequences |
| Hyperparameter Tuning | 2-4 hours | 50 trials |
| Model Training | 1-2 hours | Full dataset |
| Backtesting | 20-40 min | 42 test days |
| **Total** | **4-8 hours** | Complete pipeline |

Without GPU (CPU only): Add 2-3x more time

---

## 🎯 Recommended Workflow

### First Time:
1. ✅ Run `setup.sh` (in Git Bash)
2. ✅ Run `python test.py` (quick test)
3. ✅ If test passes, run full pipeline

### After Changes:
1. Modify config in `config.py`
2. Re-run specific steps:
   ```cmd
   python main_enhanced.py --skip-conversion --skip-data-prep
   ```

### For Experimentation:
1. Use `test.py` for quick iterations
2. Only run full pipeline when confident

---

## 💡 Tips

1. **Always activate virtual environment first**:
   ```cmd
   venv\Scripts\activate
   ```

2. **Check GPU status**:
   ```cmd
   nvidia-smi
   ```

3. **Monitor GPU usage**:
   ```cmd
   nvidia-smi -l 1
   ```

4. **Use `run.bat` for convenience** - it handles activation automatically

5. **Start with quick test** - don't run full pipeline until test passes

6. **Save your configs** - back up `config.py` before experimenting

---

## 📞 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Look in the Troubleshooting section above
3. Verify all prerequisites are installed
4. Try the quick test first: `python test.py`
5. Check that data files exist in correct location

---

## 🎓 Learning the Code

After setup, explore:
- `config.py` - All configuration options
- `test.py` - See how each component works
- `MODIFICATIONS_SUMMARY.md` - Understand the changes
- Run components individually for learning

---

## ✅ Verification Checklist

Before running full pipeline:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] GPU recognized (if available)
- [ ] Data files in `EBY\` directory
- [ ] Quick test passed
- [ ] At least 16GB RAM available
- [ ] At least 50GB disk space free

---

**Ready to go!** 🚀

Use `run.bat` for the easiest experience, or run commands directly if you prefer.

Good luck with your trading strategy!