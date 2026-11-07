"""Plot Figure 4b–f. Requires pandas, openpyxl, matplotlib, and seaborn."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BOOK = HERE / "Figure 4 Source Data.xlsx"
COLORS = {
    "HCs": "#A8DFA0", "aCar": "plum", "sAD": "#7560A8",
    "aDS": "antiquewhite", "pDS": "burlywood", "dDS": "goldenrod",
}
ORDER = ["aCar", "sAD", "aDS", "pDS", "dDS"]


def plot_line(sheet, title, order, offset_group=None):
    d = pd.read_excel(BOOK, sheet_name=f"4b {sheet}")
    d = d.set_index("Group").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8,2.2))
    y_base = 1.0
    for _, row in d.iterrows():
        y = y_base + (0.07 if row["Group"] == offset_group else 0.0)
        ax.plot(row["Z_BAG_centroid"], y, "o", markersize=14,
                color=COLORS[row["Group"]],
                markeredgecolor="black", markeredgewidth=1.0)
        ax.text(row["Z_BAG_centroid"], y+0.07, row["Group"],
                ha="center", va="bottom", fontsize=12)

    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axhline(y_base, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-2,2)
    ax.set_ylim(0.55,1.45)
    ax.set_xticks([-2,-1,0,1,2])
    ax.set_xlabel("Z-scored BAGs")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig

def scatter_space(sheet, title, xlim, ylim):
    d = pd.read_excel(BOOK, sheet_name=f"4c {sheet}")

    fig, ax = plt.subplots(figsize=(6,5))
    for group, g in d.groupby("Group", sort=False):
        ax.scatter(g["Z_BAG"], g["Z_FC"], s=20, alpha=.75,
                   color=COLORS[group], edgecolor="none")
        cx = g["Z_BAG"].mean()
        cy = g["Z_FC"].mean()
        ax.scatter(cx, cy, s=115, color=COLORS[group],
                   edgecolor="black", linewidth=1.5, zorder=3)

    ax.set_xlabel("Z-scored BAGs")
    ax.set_ylabel("Z-scored FC")
    ax.set_title(title)
    ax.set_xticks([-1.5,0,1.5])
    ax.set_yticks([-1.5,0,1.5])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig

def arrow_plot(panel, sheet, title):
    d = pd.read_excel(BOOK, sheet_name=f"{panel} {sheet}")
    d = d.set_index("Group").loc[ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(7.5,4))
    for y,row in enumerate(d.itertuples()):
        ax.scatter(row.HC_BAG, y, color=COLORS["HCs"], edgecolor="k", s=110, zorder=3)
        ax.scatter(row.Group_BAG, y, color=COLORS[row.Group], edgecolor="k", s=110, zorder=3)
        ax.annotate("", xy=(row.Best_target_BAG,y), xytext=(row.Group_BAG,y),
                    arrowprops=dict(arrowstyle="->",lw=2.5,color="black"))
        ax.annotate("", xy=(row.Worst_target_BAG,y), xytext=(row.Group_BAG,y),
                    arrowprops=dict(arrowstyle="->",lw=2.5,linestyle="--",color="gray"))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["Group"])
    ax.set_xlabel("Z-scored BAGs")
    ax.set_title(title)
    ax.set_xlim(-2,3)
    ax.grid(True,axis="x",linestyle="--",alpha=.25)
    ax.legend(handles=[
        Line2D([0],[0],color="black",lw=2.5,label="Best target"),
        Line2D([0],[0],color="gray",lw=2.5,linestyle="--",label="Worst target"),
    ],loc="best",fontsize=9)
    fig.tight_layout()
    return fig

def plot_perturbations():
    df = pd.read_excel(BOOK, sheet_name="4f ViolinData")
    b = pd.read_excel(BOOK, sheet_name="4f Baselines")

    group_order = ORDER
    cond_order = ["DRUG-Positive","DRUG-Negative","STIM-Positive","STIM-Negative"]
    labels = {"DRUG-Positive":"Drug+","DRUG-Negative":"Drug−",
              "STIM-Positive":"Stim+","STIM-Negative":"Stim−"}
    colors = {"aCar":"plum","sAD":"#7560A8","aDS":"antiquewhite",
              "pDS":"burlywood","dDS":"goldenrod"}

    df["Combined"] = df["Group"] + "_" + df["Condition"]
    order = [f"{g}_{c}" for g in group_order for c in cond_order]
    palette = [colors[x.split("_",1)[0]] for x in order]

    fig, ax = plt.subplots(figsize=(15,6))
    sns.violinplot(data=df,x="Combined",y="Z_BAG",order=order,hue="Combined",hue_order=order,palette=palette,legend=False,inner=None,ax=ax)
    sns.stripplot(data=df,x="Combined",y="Z_BAG",order=order,color="k",alpha=.2,size=2,jitter=.15,ax=ax)
    sns.boxplot(data=df,x="Combined",y="Z_BAG",order=order,width=.18,showcaps=True,
                boxprops={"facecolor":"ivory","edgecolor":"k"},
                whiskerprops={"color":"k"},capprops={"color":"k"},
                medianprops={"color":"k"},showfliers=False,ax=ax)

    ax.set_xticks(range(len(order)), [labels[c] for g in group_order for c in cond_order],rotation=45,ha="right")
    ax.set_xlabel("Perturbations")
    ax.set_ylabel("Z-scored BAGs")
    ax.set_title("BAG-distance from HCs")
    ax.set_ylim(-2,3)

    # Baseline group BAG levels used in the original figure.
    baseline_map=dict(zip(b["Group"],b["Baseline_Z_BAG"]))
    baseline_colors={"HCs":"#A8DFA0","aCar":"plum","sAD":"#7560A8",
                     "aDS":"antiquewhite","pDS":"burlywood","dDS":"goldenrod"}
    for g,val in baseline_map.items():
        ax.axhline(val,color=baseline_colors[g],linewidth=2,zorder=0)

    fig.tight_layout()
    return fig



def main():
    with plt.rc_context({"font.size": 18}):
        for sheet, title, groups in [
            ("Combined", "Combined", ["HCs", *ORDER]),
            ("MEEG", "M/EEG", ["HCs", *ORDER]),
            ("fMRI", "fMRI", ["HCs", *ORDER[1:]]),
        ]:
            plot_line(sheet, title, groups, offset_group="pDS")
        for sheet, title, limit in [("MEEG_space", "M/EEG space", 2), ("fMRI_space", "fMRI space", 2.2)]:
            scatter_space(sheet, title, (-limit, limit), (-limit, limit))
    with plt.rc_context({"font.size": 14}):
        for panel, sheet, title in [
            ("4d", "Acetylcholine_DrugPlus", "Acetylcholine (Drug+)"),
            ("4d", "GABA_DrugMinus", "GABA (Drug−)"),
            ("4e", "Excitatory_StimPlus", "Excitatory modulation (Stim+)"),
            ("4e", "Inhibitory_StimMinus", "Inhibitory modulation (Stim−)"),
        ]:
            arrow_plot(panel, sheet, title)
        plot_perturbations()
    plt.show()


if __name__ == "__main__":
    main()
