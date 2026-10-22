# ROBT 407 - Machine Learning Midterm Project

Coursework exploring linear and logistic regression through synthetic experiments and real datasets. The regression notebooks share reusable functions in `mlcore.py`, with NumPy implementations of losses, gradients, preprocessing, and evaluation metrics. SciPy provides L-BFGS optimization, and scikit-learn supplies datasets and reference models.

## Project files

| File or folder | Contents |
| --- | --- |
| [part2_linear_regression.ipynb](part2_linear_regression.ipynb) | Linear regression experiments, Sections 2.2-2.7 |
| [part3_logistic_regression.ipynb](part3_logistic_regression.ipynb) | Logistic regression theory and experiments, Sections 3.1-3.6 |
| [mlcore.py](mlcore.py) | Shared solvers, feature transforms, preprocessing, and metrics |
| [test_mlcore.py](test_mlcore.py) | Checks for the shared numerical functions |
| [requirements.txt](requirements.txt) | Python dependencies |
| [figures/](figures/) | Exported plots, including PNG and PDF files |
| [results/](results/) | Numerical outputs in CSV, JSON, and NumPy archive formats |

## Setup

Use a Python 3 environment. From the project folder, create a virtual environment and install the dependencies. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open the folder in VS Code with the Python and Jupyter extensions, open a notebook, and select `.venv` as its Python kernel. Run cells from top to bottom, keeping the project root as the working directory so imports and output paths resolve correctly.

You can also use an existing Jupyter installation with the same dependencies installed in its notebook environment.

## Experiments

### Part 2: Linear regression

- **2.2 - Five solvers:** compare the normal-equation inverse, SVD pseudoinverse, least squares, batch gradient descent, and L-BFGS-B; check the analytical gradient and plot convergence.
- **2.3 - Conditioning:** study coefficient error, training residuals, and gradient descent iterations as two features become nearly collinear.
- **2.4 - Learning curves:** compare simulated training and test errors with theoretical predictions.
- **2.5 - California Housing:** evaluate multiple regression and a single-feature baseline, with scaling fitted only on training data.
- **2.6 - Library comparison:** compare coefficients with scikit-learn and benchmark the five solvers as sample size and feature count vary.
- **2.7 - Digits classification:** classify digits 1 and 5 using intensity and left-right symmetry; compare linear regression with Pocket initialized from zero or regression weights.

### Part 3: Logistic regression

- **3.1-3.2 - Theory and numerical stability:** derive the loss, gradient, and Hessian, then check stable sigmoid and cross-entropy calculations.
- **3.3 - Optimizers:** compare batch gradient descent, stochastic gradient descent, and L-BFGS, including learning-rate and runtime experiments.
- **3.4 - Separable data:** track loss, weight norm, and weight direction with and without L2 regularization.
- **3.5 - Breast cancer classification:** report classification metrics, ROC/AUC, and a threshold selected using validation data.
- **3.6 - Library comparison:** compare unregularized and L2-regularized models with scikit-learn.

Binary classification helpers use labels `-1` and `+1`. Confusion matrices have true labels in rows and predicted labels in columns, both ordered `[-1, +1]`. In Part 3's breast cancer experiment, malignant cases are the positive class.

## Quick runs and full experiments

Both regression notebooks currently set `SEED = 407` and `FAST = False` in their setup cells. Set `FAST = True` for shorter runs, or use `FAST = False` and rerun all cells for the full experiments.

| Experiment | Quick mode | Full mode |
| --- | --- | --- |
| Part 2.3: conditioning | 10 runs per delta | 100 runs per delta |
| Part 2.4: learning curves | 20 runs per training size | 500 runs per training size |
| Part 2.6: solver timing | 3 repeats per case | 20 repeats per case |
| Part 3.3: SGD simulation | 12 runs | 100 runs |
| Part 3.3: optimizer timing | 5 repeats | 100 repeats |
| Part 3.4: separable data | 5,000 iterations | 100,000 iterations |

The full Part 2.6 benchmark varies sample size up to 1,000,000 at 20 features, and separately varies feature count up to 1,000 at 10,000 samples. These experiments can take substantial time and memory. Runtime measurements depend on the machine and background activity.

## Datasets and saved outputs

- **California Housing:** downloaded through scikit-learn on first use and cached for subsequent runs.
- **Digits and breast cancer:** loaded from datasets bundled with scikit-learn.

Part 2 saves outputs directly to `figures/` and `results/` in both modes, so a quick run can overwrite existing full-run files. Part 3 saves quick-mode outputs to `figures/preview/` and `results/preview/`, keeping them separate from its full-run outputs. Preserve full-run outputs before preparing final report results.

Part 2 uses scikit-learn's 8 x 8 digits dataset. For comparisons with other team notebooks, use the same dataset, training/test split, feature definitions, and training-fitted preprocessing.

## Check the shared functions

Run from the project root after installing dependencies:

```powershell
.\.venv\Scripts\python.exe test_mlcore.py
```

The script checks regression solvers, analytical gradients, training-only standardization, polynomial and Legendre features, folds, stable logistic calculations, classification metrics, ROC/AUC, and Pocket behavior. On success, it prints:

```text
All shared mlcore tests passed.
```
