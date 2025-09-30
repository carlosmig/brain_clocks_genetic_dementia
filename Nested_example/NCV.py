# Nested CV demo with polynomial-kernel SVR (toy data)
# - Inner 5-fold GridSearch over C and polynomial degree
# - Outer 5-fold (single repeat) evaluation
# - Metrics: Pearson r and MAE on the OUTER test folds

import numpy as np
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
rng = np.random.default_rng(42)

# ------------------------------------------------------------
# 1) Make a toy regression dataset with a non-linear structure
#    (think of "age" as the target to keep the vocabulary familiar)
# ------------------------------------------------------------
n_samples  = 400
n_features = 50

X = rng.normal(size=(n_samples, n_features))

# Construct a non-linear target (toy "age"): 20–90-ish with noise
# Use a few features to build polynomial structure; rest are noise
y = (
    40
    + 12*(X[:, 0]**2)
    -  7*(X[:, 1]**3)
    +  5*(X[:, 2]*X[:, 3])
    +  3*X[:, 4]
    + rng.normal(scale=8.0, size=n_samples)
)

# Sprinkle a few NaNs to mimic real data quirks, then zero-fill like your code
nan_mask = rng.random(X.shape) < 0.01
X = X.copy()
X[nan_mask] = np.nan
X = np.nan_to_num(X, nan=0.0)

# ------------------------------------------------------------
# 2) Nested cross-validation setup (matches your scheme)
#    - Inner: 5-fold KFold
#    - Outer: 5-fold, 1 repeat (i.e., just 5 folds total)
#    - Grid: C (log-spaced per MS), degree 1..5 (per MS)
# ------------------------------------------------------------
param_grid = {
    "svr__C":      [0.1, 1, 10, 100, 1000],     # MS info
    "svr__degree": [1, 2, 3, 4, 5],             # MS info
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

pearson_nested, mae_nested = [], []
chosen_C, chosen_degree = [], []

# Base pipeline: polynomial SVR (gamma='scale' is standard; epsilon small)
pipe = Pipeline([
    ("svr", SVR(kernel="poly", gamma="scale", epsilon=0.001, coef0=0.0))
])

# ------------------------------------------------------------
# 3) Outer loop: split, tune on inner CV, evaluate on held-out fold
# ------------------------------------------------------------
print("Nested CV (outer=5 folds, inner=5 folds) — scoring: neg MAE\n")
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    search.fit(X_train, y_train)

    best = search.best_params_
    chosen_C.append(best["svr__C"])
    chosen_degree.append(best["svr__degree"])

    # Evaluate on the *outer* test fold
    y_pred = search.predict(X_test)
    r_out  = pearsonr(y_test, y_pred)[0]
    mae_out = mean_absolute_error(y_test, y_pred)

    pearson_nested.append(r_out)
    mae_nested.append(mae_out)

    print(f"[Outer fold {fold_idx}]  best C={best['svr__C']:<6}  "
          f"deg={best['svr__degree']}   r={r_out: .3f}   MAE={mae_out: .2f}")

# ------------------------------------------------------------
# 4) Summary
# ------------------------------------------------------------
pearson_nested = np.asarray(pearson_nested, float)
mae_nested     = np.asarray(mae_nested, float)

print("\nSummary across outer folds:")
print(f"  mean r   = {pearson_nested.mean():.3f} ± {pearson_nested.std(ddof=1):.3f}")
print(f"  mean MAE = {mae_nested.mean():.2f} ± {mae_nested.std(ddof=1):.2f}")

print("\nSelected (C, degree) per outer fold:")
for c, d in zip(chosen_C, chosen_degree):
    print(f"  C={c}, degree={d}")


