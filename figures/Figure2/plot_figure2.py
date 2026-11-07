"""Plot the structural and functional brain-clock performance in Figure 2a.

Run with Python and install numpy, pandas, openpyxl, scipy, and matplotlib.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_performance(ax, data, title):
    age, prediction = data[["RealAge", "PredictedAge"]].to_numpy().T
    coordinates = np.vstack((age, prediction))
    density = stats.gaussian_kde(coordinates)(coordinates)
    density = (density - density.min()) / np.ptp(density)
    order = density.argsort()
    scatter = ax.scatter(
        age[order], prediction[order], c=density[order], cmap="RdBu_r",
        s=20, linewidths=0, alpha=0.8,
    )
    limits = np.array([10, 100])
    slope, intercept, *_ = stats.linregress(age, prediction)
    ax.plot(limits, limits, "--", color="gray", linewidth=1.2)
    ax.plot(limits, intercept + slope * limits, color="black", linewidth=1.5)
    ax.set(
        title=title, xlabel="Chronological age (years)", ylabel="Predicted age (years)",
        xticks=range(20, 101, 20), yticks=range(20, 101, 20),
    )
    r, p = stats.pearsonr(age, prediction)
    p_label = "p < 0.001" if p < 0.001 else f"p = {p:.3g}"
    mae = np.mean(np.abs(age - prediction))
    f2 = r**2 / (1 - r**2)
    ax.text(
        0.04, 0.96, f"r = {r:.3f}\n{p_label}\nCohen's f² = {f2:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.figure.colorbar(
        scatter, ax=ax, orientation="horizontal", shrink=0.65, ticks=[0, 1],
        label="Kernel density estimation",
    )


def main():
    folder = Path(__file__).resolve().parent
    book = folder / "Figure 2 Source Data.xlsx"
    sheets = ["2a Structural", "2a Functional"]
    inputs = pd.read_excel(book, sheet_name=sheets)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    titles = ["Structural", "Functional (M/EEG + fMRI)"]
    for ax, sheet, title in zip(axes, sheets, titles):
        plot_performance(ax, inputs[sheet], title)


if __name__ == "__main__":
    main()
