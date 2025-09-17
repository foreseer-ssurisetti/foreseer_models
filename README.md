# Hierarchical Model Library

## Directory Structure

```
foreseer_models/
├── __init__.py
├── base.py
├── factory.py
├── models/                     # Auto-created
│   ├── greedy/                # Greedy algorithm models
│   │   ├── model_v1.pkl
│   │   ├── production.pkl
│   │   └── experiment_1.pkl
│   ├── greedy_daily/          # Greedy daily algorithm models
│   │   ├── daily_v1.pkl
│   │   └── weekly_rebal.pkl
│   └── neural_net/            # Future neural net models
└── algorithms/
    ├── greedy.py
    ├── greedy_daily.py
    └── ...
```

## Usage
### 1. Run and save models:
```python
from foreseer_models import foreseer_models

# Save with simple filename - automatically goes to foreseer_models/models/greedy/
results = foreseer_models("greedy", 
                         data=log_returns, 
                         save_path="production_v1")  # Just the filename!

# Save greedy_daily model
daily_results = foreseer_models("greedy_daily",
                               data=log_returns,
                               save_path="daily_experiment_1")
```
### 2. List Saved Models
```python
from foreseer_models import list_saved_models

# List all saved models
all_models = list_saved_models()
print(all_models)
# Output: {'greedy': ['production_v1.pkl', 'experiment_1.pkl'], 
#          'greedy_daily': ['daily_experiment_1.pkl']}

# List models for specific algorithm
greedy_models = list_saved_models("greedy")
print(greedy_models)
# Output: {'greedy': ['production_v1.pkl', 'experiment_1.pkl']}
```
### 3. Load Saved Models:
```python
from foreseer_models import load_saved_model

# Load a saved model
loaded_algo = load_saved_model("greedy", "production_v1")

# Use the loaded model on new data
new_results = loaded_algo.backtest(new_data)
new_weights = loaded_algo.predict(latest_data)

# Check the loaded model's parameters
print(f"Loaded model gross_cap: {loaded_algo.gross_cap}")
print(f"Loaded model lookback: {loaded_algo.lookback}")
```
### 4. Load pre-trained and Continue Training:
```python
from foreseer_models import load_saved_model

# Load existing model
algo = load_saved_model("greedy_daily", "daily_experiment_1")

# Run backtest on new data with same parameters
results = algo.backtest(new_market_data)

# Save updated version
updated_path = algo.save_model("daily_experiment_1_updated")
```
