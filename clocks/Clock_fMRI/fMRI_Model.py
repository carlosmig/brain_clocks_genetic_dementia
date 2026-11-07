"""Train the fMRI brain clock from HCs data."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import svm
from sklearn.model_selection import KFold
import statsmodels.api as sm


def main():
    folder = Path(__file__).resolve().parent
    parts = sorted(folder.glob("fMRI_data_part*.csv"))
    if not parts:
        raise FileNotFoundError(f"No fMRI CSV fragments found in {folder}")
    data = pd.concat((pd.read_csv(part) for part in parts), ignore_index=True)
    if not data["Diagnosis"].eq("HCs").all():
        raise ValueError("The input fragments must contain only healthy controls.")

    feature_columns = [column for column in data if column.startswith("FC_")]
    if not feature_columns:
        raise ValueError("No FC_ features found in the input data.")
    features = np.nan_to_num(data[feature_columns].to_numpy(dtype=float), nan=0.0)
    age = data["Age"].to_numpy(dtype=float)
    if len(age) < 10 or not np.isfinite(age).all():
        raise ValueError("At least 10 controls with finite ages are required.")

    repetitions = 15
    folds = 5
    prediction_sum = np.zeros(len(age))
    all_fold_gap_sum = np.zeros(len(age))
    held_out_gap_sum = np.zeros(len(age))
    fold_correlations, fold_errors = [], []

    for repetition in range(repetitions):
        cv = KFold(n_splits=folds, shuffle=True, random_state=repetition)
        for train, test in cv.split(features):
            x_train, y_train = features[train], age[train]
            # Compute Pearson feature selection on training data only.
            correlations = np.empty(x_train.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                for start in range(0, x_train.shape[1], 256):
                    block = slice(start, start + 256)
                    correlations[block] = stats.pearsonr(
                        x_train[:, block], y_train[:, None], axis=0
                    ).statistic
            keep = np.abs(correlations) > 0.0
            if not keep.any():
                raise ValueError("No features passed the training-fold correlation filter.")
            model = svm.SVR(max_iter=10000, C=2, kernel='poly', degree=2, epsilon=1e-4)
            model.fit(x_train[:, keep], y_train)

            prediction = model.predict(features[:, keep])
            held_out = prediction[test]
            prediction_sum[test] += held_out
            fold_correlations.append(stats.pearsonr(age[test], held_out).statistic)
            fold_errors.append(np.mean(np.abs(age[test] - held_out)))

            train_gap = prediction[train] - y_train
            bias = sm.OLS(train_gap, sm.add_constant(y_train)).fit()
            corrected_gap = prediction - age - bias.predict(sm.add_constant(age))
            all_fold_gap_sum += corrected_gap
            held_out_gap_sum[test] += corrected_gap[test]

        print(f"fMRI repetition {repetition + 1}/{repetitions}", flush=True)

    predicted_age = prediction_sum / repetitions
    # All-fold BAGs reproduce the original script's aggregation, which includes
    # in-training predictions. Held-out BAGs use only test-fold predictions.
    pd.DataFrame({
        "Age": age,
        "PredictedAge_OOF": predicted_age,
        "BAG_raw_OOF": predicted_age - age,
        "BAG_biasCorr_OOF": held_out_gap_sum / repetitions,
        "BAG_biasCorr_all_folds": all_fold_gap_sum / (repetitions * folds),
    }).to_csv(folder / "fMRI_HC_BAGs.csv", index=False)

    mean_r = float(np.mean(fold_correlations))
    mean_mae = float(np.mean(fold_errors))
    f2 = mean_r**2 / (1 - mean_r**2) if abs(mean_r) < 1 else np.nan
    print(f"fMRI HC performance (mean across held-out folds): r={mean_r:.3f}, MAE={mean_mae:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(age, predicted_age, s=40, alpha=0.7, edgecolor="black")
    fit = stats.linregress(age, predicted_age)
    limits = np.array([age.min(), age.max()])
    ax.plot(limits, fit.intercept + fit.slope * limits, color="crimson", ls="--")
    ax.set(xlabel="Chronological age (years)", ylabel="Predicted age (years)",
           title="fMRI brain clock performance in healthy controls")
    ax.text(0.03, 0.97, f"Mean fold r = {mean_r:.3f}\nMean fold MAE = {mean_mae:.3f}\nCohen's f² = {f2:.3f}",
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    fig.tight_layout()

if __name__ == "__main__":
    main()
