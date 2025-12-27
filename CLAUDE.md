# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pytorchers is a PyTorch library for tabular datasets with sklearn compatibility. Currently implements regression models; classification support is planned next.

**Critical Constraint:** The `fit()` method in reg.py (lines 93, 95) expects `y` to have a `.values` attribute (pandas Series/DataFrame). This will fail with plain numpy arrays and needs fixing before widespread use.

## Development Commands

### Environment Setup
```bash
# Install in editable mode (recommended: use uv)
uv pip install -e .

# Install with dev dependencies for notebooks
uv pip install -e ".[dev]"

# Install with pre-commit dependencies
uv pip install -e . --group pre-commit

# Install pre-commit hooks
pre-commit install
```

### Code Quality
```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run ruff linting (auto-fix)
ruff check --fix .

# Run ruff formatting
ruff format .

# Run on notebooks
nbqa ruff notebooks/ --fix
```

**Ruff Config:** Line length 100, target Python 3.12, includes pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade.

**Pre-commit Config Highlights:**
- nbstripout keeps outputs but strips execution metadata from notebooks
- detect-secrets excludes .ipynb files and package-lock.json
- nbqa-ruff applies ruff checks to notebooks

### No Tests Yet
There is currently no test suite. When adding tests, use pytest and ensure sklearn compatibility with `sklearn.utils.estimator_checks.check_estimator()`.

## Architecture

### Design Pattern: Wrapper + Mixin

The library uses two key patterns to achieve sklearn compatibility while maintaining PyTorch flexibility:

1. **Wrapper Pattern (reg.py)**
   - `NNRegressorEstimator` wraps any PyTorch nn.Module
   - Implements `BaseEstimator` and `RegressorMixin` from sklearn
   - Handles all numpy ↔ tensor conversions
   - Manages training loop, validation split, and loss tracking
   - `NNRegressor` is a convenience subclass that auto-creates `BaseNNRegressor`

2. **Mixin Pattern (viz.py)**
   - `ForwardTracker` adds visualization via multiple inheritance
   - Must be combined with nn.Module: `class MyModel(nn.Module, ForwardTracker)`
   - Uses PyTorch forward hooks to capture layer activations
   - Only tracks direct child Linear layers (uses `named_children()`, not `named_modules()`)

### Critical Implementation Details

#### BaseNNRegressor (base.py)
Uses `setattr`/`getattr` for dynamic layer creation, NOT ModuleList:
```python
# Layers created as: self.fc0, self.fc1, self.fc2, ...
setattr(self, f"fc{i}", nn.Linear(...))

# Accessed in forward() as:
x = getattr(self, f"fc{i}")(x)
```

#### NNRegressorEstimator.fit() (reg.py)
Loss and optimizer are hardcoded inline (lines 81-84), not in helper methods:
```python
if self.loss == "mse":
    loss = nn.MSELoss()
if self.optimizer == "adam":
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
```

**TODO:** Only MSE loss and Adam optimizer are implemented despite documentation mentioning "mae", "huber", "sgd", "rmsprop".

#### ForwardTracker (viz.py)
- Uses `named_children()` so only tracks direct child layers, not nested modules
- Checks `isinstance(child, nn.Linear)` - won't work with Conv2d, LSTM, etc.
- Sets `self.forward_tracking` flag to prevent duplicate hook registration
- Stores activations with `{name}_activations` keys in `self.activations` dict

### sklearn Compatibility Requirements

For GridSearchCV and Pipeline to work:
1. All `__init__` parameters must be stored as instance attributes (e.g., `self.epochs = epochs`)
2. `get_params(deep=True)` must return dict of all parameters
3. `set_params(**params)` must update attributes and return self
4. `fit()` must return self
5. `predict()` should use `check_is_fitted()` (currently missing in reg.py)

**Important:** `NNRegressor.fit()` creates the model in fit(), not `__init__()`, so the model can be reconstructed during cross-validation.

## Key Gaps and Known Issues

1. **Data type assumption (HIGH PRIORITY):** reg.py lines 93, 95 call `y.values` - fails on numpy arrays
   - Fix: Add `y.values if hasattr(y, 'values') else y`
   - This affects both `NNRegressorEstimator.fit()` and indirectly `NNRegressor.fit()`

2. **Loss/optimizer limited:** Only MSE and Adam work despite accepting other strings
   - Extract inline code (lines 81-84) to helper methods
   - Add implementations for "mae", "huber" losses and "sgd", "rmsprop" optimizers

3. **No fitted check:** `predict()` doesn't verify model is fitted
   - Add `from sklearn.utils.validation import check_is_fitted`
   - Call `check_is_fitted(self, ['model'])` at start of `predict()`

4. **Empty `__init__.py`:** No public API exports
   - Should export: BaseNNRegressor, NNRegressor, NNRegressorEstimator, ForwardTracker

5. **ForwardTracker limitation:** Only works with direct child Linear layers
   - Consider using `named_modules()` for nested architectures
   - Won't work with models using nn.Sequential or deeply nested modules

## Project Structure

```
pytorchers/
├── src/pytorchers/
│   ├── __init__.py          # Currently empty - should export public API
│   ├── base.py              # BaseNNRegressor - uses setattr/getattr for layers
│   ├── reg.py               # NNRegressorEstimator + NNRegressor
│   ├── viz.py               # ForwardTracker mixin for visualization
│   └── main.py              # CLI entry point
├── notebooks/
│   └── boston.ipynb         # Canonical usage example
├── pyproject.toml           # uv-based build config
├── .pre-commit-config.yaml  # Ruff, nbqa, nbstripout, detect-secrets
└── CLAUDE.md                # This file
```

## Adding Classification Support

When implementing classification (next priority):

1. Create `BaseNNClassifier` in base.py with softmax/sigmoid output
2. Create `clf.py` with `NNClassifierEstimator(BaseEstimator, ClassifierMixin)`
3. Key differences from regression:
   - Use `nn.CrossEntropyLoss()` or `nn.BCEWithLogitsLoss()`
   - `predict()` returns class labels (argmax)
   - Add `predict_proba()` method for probabilities
   - Store `classes_` attribute in `fit()` (sklearn requirement)
   - Handle label encoding for string/categorical targets
4. Test with `check_estimator(NNClassifier(...))`
5. Follow same patterns: wrapper pattern for sklearn compat, dynamic layer creation with setattr/getattr

## Example Notebook

See `notebooks/boston.ipynb` for comprehensive usage examples on the Boston Housing dataset, demonstrating:
- sklearn Pipeline integration
- Cross-validation and GridSearchCV
- Visualization with ForwardTracker
- Train/validation comparison

This notebook is the canonical reference for how the library should be used.
