"""Plot the regional effects (3c) and gene associations (3d) from the workbook.

Install dependencies from the repository's requirements.txt.
The fsLR surfaces are fetched by neuromaps on first use.
"""
from pathlib import Path
import argparse

import dcor
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from brainsmash.mapgen.base import Base
from neuromaps.datasets import fetch_fslr
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
from surfplot import Plot


def plot_regional_effects(data, folder, output):
    atlas = folder / "atlas"
    labels = {
        side: nib.load(atlas / f"AAL.32k.{side[0]}.label.gii").darrays[0].data.astype(int)
        for side in ("LH", "RH")
    }
    left, right = fetch_fslr(density="32k")["inflated"]
    for contrast, group in data.groupby("Contrast", sort=False):
        vertex_data = {}
        for side, name in (("LH", "left"), ("RH", "right")):
            effects = group.loc[group["Hemisphere"].eq(side)].set_index("AAL_region_index_1to41")["Cohens_d"]
            if len(effects) != 41 or set(effects.index) != set(range(1, 42)):
                raise ValueError(f"{contrast}: expected labels 1–41 once each for {side}")
            lookup = np.zeros(labels[side].max() + 1)
            lookup[effects.index.to_numpy()] = effects.to_numpy()
            vertex_data[name] = lookup[labels[side]]

        plot = Plot(left, right, size=(1000, 800))
        plot.add_layer(vertex_data, cmap="RdBu_r", cbar=True, color_range=(-1.2, 1.2), cbar_label="Cohen's d")
        fig = plot.build()
        fig.suptitle(contrast.replace("_", " "))



def plot_gene_associations(data, output):
    panels = [
        ("PSEN1_aCar", "PSEN1 aCar", "#D7AAD6"),
        ("sAD_patients", "sAD patients", "#8E73D2"),
        ("dDS_patients", "dDS patients", "#D5BE45"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, layout="constrained")
    for ax, (group, title, color) in zip(axes, panels):
        rows = data.loc[data["Group"].eq(group)].iloc[::-1]
        bars = ax.barh(rows["Gene"], rows["Distance_correlation"], color=color)
        ax.set(title=title, xlabel="Distance correlation", xlim=(0, 0.53))
        for bar, p in zip(bars, rows["FDR_p"]):
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if stars:
                bar.set_edgecolor("black")
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, stars, va="center")



def run_spatial_test(inputs):
    """Recompute the BrainSMASH surrogate tests for all six group maps."""
    groups = ["HCs", "AD", "aCar", "aDS", "pDS", "dDS"]
    genes = inputs.loc[:, "MAPT":"RCAN1"].copy()
    genes[["PSEN1", "PSEN2"]] = genes[["PSEN1", "PSEN2"]].div(genes["APP"], axis=0)
    distances = squareform(pdist(inputs[["X", "Y", "Z"]]))
    results = []
    for group in groups:
        excitability = inputs[group].to_numpy()
        generator = Base(x=excitability, D=distances, pv=100, nh=2, resample=True, seed=0)
        surrogates = generator(n=100)
        rows = []
        for gene in genes:
            expression = genes[gene].to_numpy()
            mask = expression > 0
            expression = expression[mask]
            observed = dcor.distance_correlation(expression, excitability[mask], exponent=0.5)
            null = np.array([
                dcor.distance_correlation(expression, sample, exponent=0.5)
                for sample in surrogates[:, mask]
            ])
            rows.append((group, gene, observed, np.mean(null >= observed)))
        tests = pd.DataFrame(rows, columns=["Group", "Gene", "Distance_correlation", "Surrogate_p"])
        tests["FDR_p"] = multipletests(tests["Surrogate_p"], method="fdr_bh")[1]
        results.append(tests)
    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-test-only", action="store_true", help="Run 3d without rendering the 3c maps.")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    output = folder / "plots"
    output.mkdir(exist_ok=True)
    book = folder / "Figure 3 Source Data.xlsx"
    results = run_spatial_test(pd.read_excel(book, sheet_name="3d Spatial test inputs"))
    results.to_csv(output / "Fig3d_spatial_test.csv", index=False)
    shown = results.assign(Group=results["Group"].replace({
        "aCar": "PSEN1_aCar", "AD": "sAD_patients", "dDS": "dDS_patients",
    }))
    plot_gene_associations(shown, output)
    if not args.spatial_test_only:
        plot_regional_effects(pd.read_excel(book, sheet_name="3c Regional effects"), folder, output)


if __name__ == "__main__":
    main()
