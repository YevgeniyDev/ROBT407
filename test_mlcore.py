"""Run from project folder: python test_mlcore.py"""
import numpy as np
from numpy.polynomial.legendre import legval

from mlcore import (
    add_bias, gradient_check, standardize, transform_standardize,
    linear_lstsq, linear_loss, linear_grad, mse, r2_score_numpy,
    legendre_features, poly_features, kfold_indices,
    sigmoid, logistic_loss, logistic_grad,
    confusion_matrix, classification_metrics, roc_curve, auc,
)

rng = np.random.default_rng(407)

# Design and regression checks
X_raw = rng.normal(size=(100, 2))
X = add_bias(X_raw)
w_star = np.array([1.5, -2.0, 0.8])
y = X @ w_star
assert X.shape == (100, 3)
assert np.allclose(linear_lstsq(X, y), w_star, atol=1e-12)
assert mse(y, X @ w_star) < 1e-25
assert np.isclose(r2_score_numpy(y, X @ w_star), 1)
assert gradient_check(lambda w: linear_loss(w, X, y),
                      lambda w: linear_grad(w, X, y),
                      w_star + .3) < 1e-6

# Training-only standardization: giant holdout value must not affect mean
train = np.array([[1., 10.], [3., 30.], [5., 50.]])
holdout = np.array([[100., 1000.]])
Z, Zh, mu, scale = standardize(train, holdout)
assert np.allclose(mu, [3, 30])
assert np.allclose(Z.mean(axis=0), 0)
assert np.allclose(Zh, transform_standardize(holdout, mu, scale))
assert np.allclose(standardize(np.ones((2, 1)))[0], 0)

# Legendre recursion against numpy.polynomial
x = np.linspace(-1., 1., 21)
for Q in range(8):
    Z = legendre_features(x, Q)
    for q in range(Q+1):
        c = np.eye(Q+1)[q]
        assert np.allclose(Z[:, q], legval(x, c), atol=1e-12)

assert np.allclose(poly_features(np.array([[2., 3.]]), 2),
                   [[1., 2., 3., 4., 6., 9.]])
assert poly_features(np.zeros((7, 2)), 3).shape == (7, 10)

folds = kfold_indices(37, 5, np.random.default_rng(407))
assert np.array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(37))
assert all(len(set(t) & set(v)) == 0 for t, v in folds)

# Extreme logits safe; gradient accurate
assert np.array_equal(sigmoid(np.array([-1000., 1000.])), [0., 1.])
assert sigmoid(0.) == 0.5
Xlog = add_bias(rng.normal(size=(40, 2)))
ylog = rng.choice([-1,1], size=40)
w = rng.normal(size=3)
assert np.isfinite(logistic_loss(np.array([1.]), np.array([[-1000.], [1000.]]),
                                 np.array([1., -1.])))
assert gradient_check(lambda t: logistic_loss(t, Xlog, ylog, lam=.07),
                      lambda t: logistic_grad(t, Xlog, ylog, lam=.07), w) < 1e-6

# Binary metrics hand-checked: TP=2, FP=1, FN=1, TN=2
yt = np.array([-1, -1, -1, +1, +1, +1])
yp = np.array([-1, -1, +1, -1, +1, +1])
m = classification_metrics(yt, yp)
assert np.array_equal(confusion_matrix(yt, yp), [[2,1],[1,2]])
assert np.isclose(m['f1'], 2/3) and np.isclose(m['accuracy'], 2/3)
scores = np.array([.1, .2, .8, .5, .7, .9])
fpr, tpr, thresholds = roc_curve(yt, scores)
assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
assert np.isclose(auc(fpr, tpr), 7/9)  # 7 of 9 pos/neg pairs ranked correctly

# Check the five linear regression solvers and Pocket initialization.
from mlcore import (solve_naive_inverse, solve_svd, solve_lstsq,
                    solve_gd, solve_lbfgs, stratified_split, pocket_pla)
ref = solve_lstsq(X, y)
for estimate in [solve_naive_inverse(X, y), solve_svd(X, y),
                 solve_gd(X, y)[0], solve_lbfgs(X, y).x]:
    assert np.max(np.abs(ref - estimate)) < 1e-4
tr, te = stratified_split(np.array([1]*10 + [-1]*10), 0.3, rng)
assert len(tr) == 14 and len(te) == 6
weights, history = pocket_pla(X[:20], np.where(y[:20]>0,1,-1), 20, rng)
assert len(weights) == X.shape[1] and np.all(np.diff(history) <= 0)
print('All shared mlcore tests passed.')
