"""Shared NumPy toolkit for ROBT 407 Midterm Project (Parts 0 to 5).

Use one copy of this module in the project root for all notebooks.
Algorithms/transforms/metrics below are implemented with NumPy (no sklearn).
The original Parts 2.2 to 2.5 helper API is preserved:
    add_bias, gradient_check, standardize, linear_lstsq, mse, r2_score_numpy.

Label convention for classification: positive class = +1, negative = -1.
Confusion matrix row=true and column=predicted, ordered [-1, +1].
"""
from __future__ import annotations

from itertools import combinations_with_replacement
import numpy as np


def add_bias(X):
    """Return [1, X] for an (N, d) feature matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (N, d)")
    return np.column_stack((np.ones(X.shape[0]), X))


def standardize(X_train, X_other=None):
    """Fit on training features ONLY and transform train and optional other data.

    Returns (Z_train, Z_other_or_None, train_means, train_scales).
    Constant columns use a safe scale of 1.0; call before adding bias.
    """
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError("X_train must be a nonempty 2D array")
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Z_train = (X_train - mu) / sigma
    if X_other is None:
        Z_other = None
    else:
        X_other = np.asarray(X_other, dtype=float)
        if X_other.ndim != 2 or X_other.shape[1] != X_train.shape[1]:
            raise ValueError("Other data must have matching number of features")
        Z_other = (X_other - mu) / sigma
    return Z_train, Z_other, mu, sigma


def transform_standardize(X, means, scales):
    """Reuse training-fitted means and scales on validation/test features."""
    return (np.asarray(X, dtype=float) - means) / scales


def legendre_features(x, Q):
    """Columns L_0(x)...L_Q(x) via Legendre three-term recurrence."""
    if Q < 0:
        raise ValueError("Q must be nonnegative")
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError("x must be a vector or an (N,1) matrix")
    Z = np.empty((x.size, Q + 1), dtype=float)
    Z[:, 0] = 1.0
    if Q >= 1:
        Z[:, 1] = x
    for k in range(2, Q + 1):
        Z[:, k] = ((2*k-1)*x*Z[:, k-1] - (k-1)*Z[:, k-2]) / k
    return Z


def poly_features(X, Q):
    """All monomials of total degree <=Q, including intercept column.

    Columns use increasing total degree and combinations with replacement;
    for d=2 and Q=2: [1, x1, x2, x1**2, x1*x2, x2**2].
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or Q < 0:
        raise ValueError("X must be 2D and Q must be nonnegative")
    N, d = X.shape
    columns = [np.ones(N)]
    for degree in range(1, Q + 1):
        for indices in combinations_with_replacement(range(d), degree):
            columns.append(np.prod(X[:, indices], axis=1))
    return np.column_stack(columns)


def kfold_indices(N, K, rng):
    """List of (training_indices, validation_indices) with disjoint folds."""
    if not (2 <= K <= N):
        raise ValueError("Require 2 <= K <= N")
    order = rng.permutation(N)
    folds = np.array_split(order, K)
    return [(np.concatenate([folds[j] for j in range(K) if j != k]),
             folds[k].copy()) for k in range(K)]


def gradient_check(f, grad, w, h=1e-5):
    """Central-difference relative gradient error; expects f(w) and grad(w)."""
    w = np.asarray(w, dtype=float)
    analytical = np.asarray(grad(w), dtype=float)
    if analytical.shape != w.shape:
        raise ValueError("Gradient shape differs from weight-vector shape")
    numerical = np.empty_like(w)
    for j in range(w.size):
        direction = np.zeros_like(w)
        direction[j] = h
        numerical[j] = (f(w + direction) - f(w - direction)) / (2*h)
    denominator = max(1e-12, np.linalg.norm(analytical) + np.linalg.norm(numerical))
    return float(np.linalg.norm(analytical - numerical) / denominator)


def linear_lstsq(X, y):
    """Stable least-squares coefficients; do not invert (X.T @ X)."""
    return np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)[0]


def linear_loss(w, X, y):
    return mse(y, np.asarray(X, float) @ w)


def linear_grad(w, X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    return (2.0 / len(y)) * X.T @ (X @ w - y)


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and target shapes differ")
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_numpy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and target shapes differ")
    sst = np.sum((y_true - y_true.mean()) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else (1.0 if sse == 0 else 0.0)


def sigmoid(s):
    """Overflow-free sigmoid for extreme positive/negative scores."""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    positive = s >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-s[positive]))
    exp_s = np.exp(s[~positive])
    out[~positive] = exp_s / (1.0 + exp_s)
    return out.item() if out.ndim == 0 else out


def logistic_loss(w, X, y, lam=0.0, penalize_bias=False):
    """Mean logistic cross entropy + lam*||w_nonbias||^2 (by default)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    margin = y * (X @ w)
    reg = w if penalize_bias else w[1:]
    return float(np.mean(np.logaddexp(0, -margin)) + lam * np.dot(reg, reg))


def logistic_grad(w, X, y, lam=0.0, penalize_bias=False):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    margin = y * (X @ w)
    grad = -(X.T @ (y * sigmoid(-margin))) / len(y)
    penalized = np.asarray(w if penalize_bias else w[1:], float)
    if penalize_bias:
        grad += 2*lam*penalized
    else:
        grad[1:] += 2*lam*penalized
    return grad


def _check_binary_labels(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [-1, 1]).all():
        raise ValueError("Labels must be a 1D array of -1/+1")
    return y


def confusion_matrix(y_true, y_pred):
    """Rows true [-1,+1], columns predicted [-1,+1]."""
    yt, yp = _check_binary_labels(y_true), _check_binary_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError("Length mismatch")
    tn = np.sum((yt == -1) & (yp == -1))
    fp = np.sum((yt == -1) & (yp ==  1))
    fn = np.sum((yt ==  1) & (yp == -1))
    tp = np.sum((yt ==  1) & (yp ==  1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.0
    accuracy = (tp + tn)/total if total else float('nan')
    return {"error": 1.0 - accuracy, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": cm}


def roc_curve(y_true, scores):
    """Return (fpr,tpr,thresholds), with scores higher => +1 prediction."""
    yt = _check_binary_labels(y_true)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != yt.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite and match label vector")
    positives = np.sum(yt == 1)
    negatives = np.sum(yt == -1)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC needs both positive and negative classes")
    sorted_indices = np.argsort(scores, kind='mergesort')[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_y = (yt[sorted_indices] == 1)
    boundary = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(yt)-1]
    tp = np.cumsum(sorted_y)[boundary]
    fp = 1 + boundary - tp
    tpr = np.r_[0., tp / positives]
    fpr = np.r_[0., fp / negatives]
    thresholds = np.r_[np.inf, sorted_scores[boundary]]
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Trapezoidal ROC AUC for monotonically increasing FPR."""
    fpr = np.asarray(fpr, float)
    tpr = np.asarray(tpr, float)
    if fpr.shape != tpr.shape or np.any(np.diff(fpr) < 0):
        raise ValueError("Invalid ROC coordinates")
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))



# Five solvers for Part 2
from scipy.optimize import minimize


def solve_naive_inverse(X, y):
    """Normal equation inverse. Kept only to show stability problems."""
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def solve_svd(X, y):
    """Least squares using an SVD pseudoinverse."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    threshold = np.finfo(float).eps * max(X.shape) * s[0]
    inverse_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > threshold)
    return Vt.T @ (inverse_s * (U.T @ y))


def solve_lstsq(X, y):
    """Least squares through numpy.linalg.lstsq."""
    return linear_lstsq(X, y)


def solve_gd(X, y, tol=1e-10, max_iter=100000):
    """Full batch gradient descent on the mean squared error."""
    w = np.zeros(X.shape[1])
    L = 2.0 * np.linalg.norm(X, ord=2)**2 / X.shape[0]
    eta = 0.8 / L
    history = [linear_loss(w, X, y)]
    for step in range(max_iter):
        grad = linear_grad(w, X, y)
        if np.linalg.norm(grad) < tol:
            return w, np.asarray(history), step, True, eta
        w = w - eta * grad
        history.append(linear_loss(w, X, y))
    return w, np.asarray(history), max_iter, False, eta


def solve_lbfgs(X, y, tol=1e-10, max_iter=10000):
    """SciPy L-BFGS-B using our MSE and our analytical gradient."""
    return minimize(
        fun=linear_loss, x0=np.zeros(X.shape[1]), args=(X, y),
        jac=linear_grad, method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-14}
    )


def stratified_split(y, test_fraction, rng):
    """Return train and test indices with the same class proportions."""
    y = np.asarray(y)
    train_indices, test_indices = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_test = int(round(len(indices) * test_fraction))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)


def pocket_pla(X, y, steps, rng, w0=None):
    """Pocket PLA. Pick a misclassified training point at random each step."""
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, float, copy=True)
    best_w = w.copy()
    best_error = np.mean(np.sign(X @ w) != y)
    history = [best_error]
    for _ in range(steps):
        wrong = np.flatnonzero(np.sign(X @ w) != y)
        if len(wrong) == 0:
            break
        i = rng.choice(wrong)
        w = w + y[i] * X[i]
        error = np.mean(np.sign(X @ w) != y)
        if error < best_error:
            best_w = w.copy()
            best_error = error
        history.append(best_error)
    return best_w, np.asarray(history)

__all__ = [
    'add_bias', 'standardize', 'transform_standardize', 'legendre_features',
    'poly_features', 'kfold_indices', 'gradient_check', 'linear_lstsq',
    'linear_loss', 'linear_grad', 'mse', 'r2_score_numpy', 'sigmoid',
    'logistic_loss', 'logistic_grad', 'solve_naive_inverse',
    'solve_svd', 'solve_lstsq', 'solve_gd', 'solve_lbfgs',
    'stratified_split', 'pocket_pla', 'confusion_matrix',
    'classification_metrics', 'roc_curve', 'auc'
]
