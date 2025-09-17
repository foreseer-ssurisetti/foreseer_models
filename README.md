# Test Hierarchical Model Repo

To call a specific model use the following code:

```python
from foreseer_models import foreseer_models

# Save automatically during backtest
results = foreseer_models("greedy",
                         data=log_returns,
                         save_path="models/greedy_model_v1") # Create a models folder if not there already

# The model is automatically saved, and the path is stored in results
print(f"Model saved to: {results.model_path}")
```

