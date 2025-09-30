
# ============================================================
# Clean & Documented Figure Pipeline (OUTLIER-CLEANED RESULTS)
# - Model performance (scatter KDE): MRI, fMRI, MEG&EEG, EEG+fMRI+MEG
# - Group BAG comparisons: MRI, fMRI, EEG, MEG, EEG+fMRI+MEG
# - Associations (Pearson r with FDR-BH): PET, TRS, Longitudinal (struct/fMRI/EEG),
#   plasma p-tau217, plasma NfL
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import itertools, warnings
from scipy import stats
from scipy.stats import gaussian_kde, t
from statsmodels.stats.oneway import anova_oneway  # Welch ANOVA
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

warnings.filterwarnings("ignore")
rcParams.update({'font.size': 20})
FONT_LEGEND = 14

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def remove_outliers_groupwise(df, value_col="BAGs", group_col="Group", z=2.0):
    """Remove ±z*SD outliers within each group for group comparisons."""
    def _f(g):
        m, s = g[value_col].mean(), g[value_col].std(ddof=1)
        if not np.isfinite(s) or s == 0:
            return g
        return g[(g[value_col] >= m - z*s) & (g[value_col] <= m + z*s)]
    return df.groupby(group_col, group_keys=False).apply(_f)

def _robust_z(x):
    med = np.median(x)
    mad = stats.median_abs_deviation(x, scale='normal')
    if not np.isfinite(mad) or mad == 0:
        mad = np.std(x, ddof=1) or 1.0
    return (x - med) / mad

def _mahal_inlier_mask(x, y, alpha=0.001):
    """Bivariate inlier mask using Mahalanobis distance with chi-square cutoff (df=2)."""
    X = np.column_stack([x, y])
    mu = np.nanmean(X, axis=0)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    inv = np.linalg.pinv(cov)
    m2 = np.einsum('ij,jk,ik->i', Xc, inv, Xc)
    thr = stats.chi2.ppf(1 - alpha, df=2)
    return m2 <= thr

def remove_outliers_pair(x, y, z_thresh=3.5, alpha_maha=0.001):
    """Bivariate outlier removal for correlations (robust-z + Mahalanobis)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    zx, zy = _robust_z(x), _robust_z(y)
    uni_ok = (np.abs(zx) <= z_thresh) & (np.abs(zy) <= z_thresh)
    bi_ok  = _mahal_inlier_mask(x, y, alpha=alpha_maha)
    keep = uni_ok & bi_ok
    return x[keep], y[keep], int(np.count_nonzero(~keep)), int(len(x))

def pearson_safe(x, y):
    """Pearson r with guards for tiny N or zero variance."""
    if len(x) < 3 or np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return np.nan, np.nan
    return stats.pearsonr(x, y)

def welch_df(var1, n1, var2, n2):
    num = (var1 / n1 + var2 / n2) ** 2
    den = (var1**2) / (n1**2 * (n1 - 1)) + (var2**2) / (n2**2 * (n2 - 1))
    return num / den

def cohen_d(m1, m2, s1, s2, n1, n2):
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    return (m1 - m2) / np.sqrt(pooled_var) if pooled_var > 0 else np.nan

def pairwise_welch_table(df, order, value_col="BAGs"):
    """Welch t-tests for all group pairs + FDR-BH correction."""
    groups = {g: df.loc[df["Group"] == g, value_col].dropna().values for g in order}
    rows = []
    for g1, g2 in itertools.combinations(order, 2):
        x, y = groups[g1], groups[g2]
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            rows.append(dict(Comparison=f"{g1} vs {g2}", p_raw=np.nan))
            continue
        m1, m2 = x.mean(), y.mean()
        v1, v2 = x.var(ddof=1), y.var(ddof=1)
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        t_stat, p_raw = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
        df_w = welch_df(v1, n1, v2, n2)
        se = np.sqrt(v1/n1 + v2/n2)
        ci_low, ci_high = (m1 - m2) + np.array([-1, 1]) * t.ppf(0.975, df_w) * se
        d_val = cohen_d(m1, m2, s1, s2, n1, n2)
        rows.append(dict(Comparison=f"{g1} vs {g2}",
                         t=np.round(t_stat, 3),
                         df=np.round(df_w, 1),
                         CI95_low=np.round(ci_low, 3),
                         CI95_high=np.round(ci_high, 3),
                         p_raw=p_raw,
                         Cohen_d=np.round(d_val, 3)))
    out = pd.DataFrame(rows)
    out["p_FDR"] = multipletests(out["p_raw"], method="fdr_bh")[1]
    return out[["Comparison","t","df","CI95_low","CI95_high","p_raw","p_FDR","Cohen_d"]]

def signif_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'NS'

# ------------------------------------------------------------
# Plot helpers 
# ------------------------------------------------------------
def plot_model_performance(real_age, pred_age, per_mae, per_r,
                           title, xticks=(20,40,60,80,100), yticks=(20,40,60,80,100)):
    """Density-coloured scatter with identity + regression line."""
    mean_r = np.nanmean(per_r) if np.ndim(per_r) else per_r
    f2 = mean_r**2 / (1 - mean_r**2) if mean_r**2 < 1 else np.inf
    mae = np.nanmean(per_mae) if np.ndim(per_mae) else per_mae
    r_data, p_val = stats.pearsonr(real_age, pred_age)
    p_str = "p < 0.001" if p_val < 1e-3 else f"p = {p_val:.3g}"

    fig, ax = plt.subplots(figsize=(6, 6))
    xy = np.vstack([real_age, pred_age])
    z = gaussian_kde(xy)(xy)
    z = (z - z.min()) / (z.max() - z.min())
    idx = z.argsort()
    x, y, z = real_age[idx], pred_age[idx], z[idx]
    sc = ax.scatter(x, y, c=z, cmap='RdBu_r', s=20, edgecolor='none', alpha=0.8)

    lims = [10, 10, 100, 100]
    ax.plot(lims, lims, ls='--', lw=1.2, color='gray')

    slope, intercept, *_ = stats.linregress(real_age, pred_age)
    ax.plot(lims, intercept + slope*np.array(lims), lw=1.5, color='black')

    ax.set_xlabel('Chronological age (years)')
    ax.set_ylabel('Predicted age (years)')
    ax.set_xticks(xticks); ax.set_yticks(yticks)

    annot = (fr'$r={mean_r:.3f}$' '\n'
             f'{p_str}' '\n'
             fr"Cohen's $f^2={f2:.2f}$" '\n'
             fr'MAE={mae:.2f}')
    ax.text(0.04, 0.96, annot, transform=ax.transAxes,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=FONT_LEGEND)
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.15, fraction=0.1, aspect=15, shrink=0.55)
    cbar.set_ticks([0,1]); cbar.set_ticklabels(['0','1'])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_violin_with_stats(df, order, title, ylim):
    """
    Violin + strip + box; Welch ANOVA + pairwise Welch with FDR-BH; significance bars.
    Assumes df has columns: Group, BAGs. NO SAVING.
    """
    # Clean outliers within group
    df = remove_outliers_groupwise(df, "BAGs", "Group", z=2.0)
    print(f"{title}: N={len(df)} after outlier removal")

    # Plot
    plt.figure(figsize=(6, 6) if len(order) > 2 else (4, 6))
    x_pos = {g: i for i, g in enumerate(order)}
    sns.violinplot(data=df, x='Group', y='BAGs',
                   order=order,
                   palette=['skyblue','indianred','antiquewhite','burlywood','goldenrod'][:len(order)],
                   inner=None, zorder=1)
    sns.stripplot(data=df, x='Group', y='BAGs',
                  order=order, color='k', alpha=0.2, size=3, jitter=0.15, zorder=2)
    sns.boxplot(data=df, x='Group', y='BAGs', order=order, width=0.2, showcaps=True,
                boxprops={'facecolor': 'ivory', 'edgecolor': 'k'},
                whiskerprops={'color': 'k'}, capprops={'color': 'k'},
                medianprops={'color': 'k'}, showfliers=False, zorder=3)
    ax = plt.gca()
    ax.set_ylim(*ylim); ax.set_ylabel('BAG (years)'); plt.title(title)

    # Welch ANOVA
    groups = [df.loc[df['Group']==g, 'BAGs'].values for g in order]
    anova_res = anova_oneway(groups, use_var='unequal', welch_correction=True)
    print(f"\n{title} — Welch ANOVA:\n{anova_res}")

    # Pairwise Welch + FDR
    posthoc = pairwise_welch_table(df, order, "BAGs")
    print(f"\n{title} — Pairwise Welch (FDR-BH):\n{posthoc.to_string(index=False)}")

    # Significance bars (use FDR p)
    y_offset = df['BAGs'].max() + (5 if len(order) <= 2 else -2)
    bar_gap, bar_tick = 4.1, 0.2
    for k, row in posthoc.iterrows():
        if not np.isfinite(row['p_FDR']): continue
        g1, g2 = row['Comparison'].split(' vs ')
        if g1 not in x_pos or g2 not in x_pos: continue
        x1, x2 = x_pos[g1], x_pos[g2]
        y = y_offset + k*bar_gap
        ax.plot([x1, x1, x2, x2], [y, y+bar_tick, y+bar_tick, y], color='k', linewidth=1.2)
        ax.text((x1+x2)/2, y+bar_tick-0.5, signif_label(row['p_FDR']), ha='center', va='bottom', fontsize=13)

    top_bar = y_offset + (len(posthoc)-1)*bar_gap + bar_tick + 2
    if ax.get_ylim()[1] < top_bar:
        ax.set_ylim(ax.get_ylim()[0], top_bar)

    plt.tight_layout()
    plt.show()

def scatter_reg(x, y, xlabel, ylabel, title, p_display=None, identity=False):
    """Correlation plot with regression line and optional identity line. NO SAVING."""
    r, p_raw = pearson_safe(x, y)
    if not np.isfinite(r):
        r, p_raw = np.nan, np.nan
    r2 = r**2 if np.isfinite(r) else np.nan
    f2 = (r2/(1-r2)) if (np.isfinite(r2) and r2 < 1) else (np.inf if np.isfinite(r2) and r2 == 1 else np.nan)
    p_show = p_display if (p_display is not None and np.isfinite(p_display)) else p_raw
    p_str = "p < 0.001" if (np.isfinite(p_show) and p_show < 1e-3) else (f"p = {p_show:.3g}" if np.isfinite(p_show) else "p = nan")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(x, y, s=75, alpha=0.7, edgecolor='none', color='tab:blue')
    slope, intercept, *_ = stats.linregress(x, y)
    xx = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xx, intercept + slope*xx, color='k', lw=1.5)
    if identity:
        lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        ax.plot(lims, lims, ls='--', lw=1, color='gray')
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    txt = (fr"$r={r:.3f}$" "\n" f"{p_str}\n" + (fr"Cohen's $f^2={f2:.2f}$" if np.isfinite(f2) else "Cohen's $f^2$=nan"))
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top',
            fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    plt.show()
    return r, p_raw, f2

# ============================================================
# A) MODEL PERFORMANCE (SVM) — scatter KDE plots
# ============================================================
base_mri   = Path(r"Zdata_MRI")
base_fmri  = Path(r"Zdata_fMRI")
base_megeg = Path(r"Zdata_MEG&EEG")

# MRI
real = np.load(base_mri / "real_age.npy")
pred = np.load(base_mri / "pred_age.npy")
mae  = np.load(base_mri / "performance_mae.npy")
rr   = np.load(base_mri / "performance_pearson.npy")
plot_model_performance(real, pred, mae, rr, "Structural (MRI)")

# fMRI
real = np.load(base_fmri / "real_age.npy")
pred = np.load(base_fmri / "pred_age.npy")
mae  = np.load(base_fmri / "performance_mae.npy")
rr   = np.load(base_fmri / "performance_pearson.npy")
plot_model_performance(real, pred, mae, rr, "Functional (fMRI)")

# MEG&EEG
real = np.load(base_megeg / "real_age.npy")
pred = np.load(base_megeg / "pred_age.npy")
mae  = np.load(base_megeg / "performance_mae.npy")
rr   = np.load(base_megeg / "performance_pearson.npy")
plot_model_performance(real, pred, mae, rr, "MEG & EEG")

# Combined EEG+fMRI+MEG
real_f = np.load(base_fmri  / "real_age.npy")
pred_f = np.load(base_fmri  / "pred_age.npy")
real_g = np.load(base_megeg / "real_age.npy")
pred_g = np.load(base_megeg / "pred_age.npy")
real_c = np.append(real_f, real_g)
pred_c = np.append(pred_f, pred_g)
mae_c  = np.mean(np.abs(real_c - pred_c))
r_c    = stats.pearsonr(real_c, pred_c)[0]
plot_model_performance(real_c, pred_c, mae_c, r_c, "EEG + fMRI + MEG")

# ============================================================
# B) GROUP BAG COMPARISONS (OUTLIERS REMOVED)
# ============================================================
# Structural (MRI)
df_mri = pd.read_csv(r"C:/Users/carlo/OneDrive/Down Syndrome/Figure2/Data/Zdata_MRI/bags_mri.csv")
plot_violin_with_stats(df_mri,
                       order=['HCs','AD','aDS','pDS','dDS'],
                       title="Structural", ylim=(-30, 80))

# Functional combined (fMRI + MEG&EEG)
df_fmri = pd.read_csv(r"C:/Users/carlo/OneDrive/Down Syndrome/Figure2/Data/Zdata_fMRI/bags_fmri.csv")
df_meg_eeg_north = pd.read_csv(r"C:/Users/carlo/OneDrive/Down Syndrome/Figure2/Data/Zdata_MEG&EEG/bags_meg&eeg_north.csv")
df_func = pd.concat([df_fmri, df_meg_eeg_north], ignore_index=True)
plot_violin_with_stats(df_func,
                       order=['HCs','AD','aDS','pDS','dDS'],
                       title="Functional", ylim=(-30, 65))

# fMRI alone
plot_violin_with_stats(df_fmri,
                       order=['HCs','AD','aDS','pDS','dDS'],
                       title="fMRI", ylim=(-30, 65))

# MEG (HCs vs DS)
df_meg = pd.read_csv(r"C:/Users/carlo/OneDrive/Down Syndrome/Figure2/Data/Zdata_MEG&EEG/bags_meg.csv")
plot_violin_with_stats(df_meg,
                       order=['HCs','DS'],
                       title="MEG", ylim=(-30, 65))

# EEG (HCs, nCar, aCar, AD, DS)
df_eeg = pd.read_csv(r"C:/Users/carlo/OneDrive/Down Syndrome/Figure2/Data/Zdata_MEG&EEG/bags_carriers.csv")
df_eeg['Group'] = df_eeg['Group'].replace({'carriers':'aCar','non-carriers':'nCar'})
plot_violin_with_stats(df_eeg,
                       order=['HCs','nCar','aCar','AD','DS'],
                       title="EEG", ylim=(-30, 65))

# ============================================================
# C) ASSOCIATIONS (OUTLIER-CLEANED) — PET, TRS, Longitudinal
#     FDR-BH ACROSS THIS BLOCK (5 tests)
# ============================================================
centiloids = np.load(base_mri / "centiloids.npy")
bags_cent  = np.load(base_mri / "bags_for_centiloids.npy")
trs        = np.load(base_mri / "trs.npy")
bags_trs   = np.load(base_mri / "bags_for_trs.npy")

mask_trs = (trs < 31)
trs_31, bags_trs_31 = trs[mask_trs], bags_trs[mask_trs]

bags_v1_struct = np.load(base_mri  / "bags_visit_1.npy")
bags_v2_struct = np.load(base_mri  / "bags_visit_2.npy")
base_fmri2     = base_fmri
bags_v1_func   = np.load(base_fmri2 / "bags_visit_1.npy")
bags_v2_func   = np.load(base_fmri2 / "bags_visit_2.npy")
bags_v1_eeg    = np.load(base_megeg / "bags_visit_1.npy")
bags_v2_eeg    = np.load(base_megeg / "bags_visit_2.npy")

assoc_specs = [
    dict(name="Amyloid-PET", x=centiloids, y=bags_cent,
         xlabel="Centiloids", ylabel="BAG (years)", identity=False),
    dict(name="Cognition", x=trs_31, y=bags_trs_31,
         xlabel="Total recall score", ylabel="BAG (years)", identity=False),
    dict(name="Longitudinal BAG (Structural)", x=bags_v1_struct, y=bags_v2_struct,
         xlabel="BAG visit 1 (years)", ylabel="BAG visit 2 (years)", identity=True),
    dict(name="Longitudinal BAG (Functional)", x=bags_v1_func, y=bags_v2_func,
         xlabel="BAG visit 1 (years)", ylabel="BAG visit 2 (years)", identity=True),
    dict(name="Longitudinal BAG (EEG)", x=bags_v1_eeg, y=bags_v2_eeg,
         xlabel="BAG visit 1 (years)", ylabel="BAG visit 2 (years)", identity=True),
]

assoc_results = []
for spec in assoc_specs:
    x_c, y_c, n_out, n0 = remove_outliers_pair(spec['x'], spec['y'])
    if "Longitudinal BAG" in spec['name']:
        keep = (y_c - x_c) >= -20
        x_c, y_c = x_c[keep], y_c[keep]
    r, p = pearson_safe(x_c, y_c)
    r2 = r**2 if np.isfinite(r) else np.nan
    f2 = (r2/(1-r2)) if (np.isfinite(r2) and r2 < 1) else (np.inf if np.isfinite(r2) and r2==1 else np.nan)
    assoc_results.append(dict(name=spec['name'], r=r, p_raw=p, f2=f2, N=len(x_c),
                              x=x_c, y=y_c, xlabel=spec['xlabel'], ylabel=spec['ylabel'], identity=spec['identity']))

pvals = [d['p_raw'] if np.isfinite(d['p_raw']) else 1.0 for d in assoc_results]
_, p_fdr, _, _ = multipletests(pvals, method='fdr_bh')
for d, pf in zip(assoc_results, p_fdr): d['p_fdr'] = pf

# Print summary and plot
assoc_summary = pd.DataFrame([
    dict(Measure=d['name'],
         r=(np.nan if not np.isfinite(d['r']) else round(d['r'],3)),
         p_raw=("nan" if not np.isfinite(d['p_raw']) else f"{d['p_raw']:.5f}".rstrip('0').rstrip('.')),
         p_FDR=("nan" if not np.isfinite(d['p_fdr']) else ("p<0.001" if d['p_fdr']<1e-3 else f"{d['p_fdr']:.5f}".rstrip('0').rstrip('.'))),
         f2=(np.nan if not np.isfinite(d['f2']) else (round(d['f2'],3) if d['f2']!=np.inf else "inf")),
         N=d['N'])
    for d in assoc_results
])
print("\nAssociations (no outliers):\n", assoc_summary.to_string(index=False))

for d in assoc_results:
    scatter_reg(d['x'], d['y'], d['xlabel'], d['ylabel'], d['name'],
                p_display=d['p_fdr'], identity=d['identity'])

# ============================================================
# D) PLASMA ASSOCIATIONS (OUTLIER-CLEANED) — p-tau217, NfL
#     FDR-BH ACROSS THIS BLOCK (2 tests)
# ============================================================
ptau217 = np.load(base_mri / "plasma_ptau217.npy")
nfl     = np.load(base_mri / "plasma_nfl.npy")
bags_ptau = np.load(base_mri / "bags_ptau217.npy")
bags_nfl  = np.load(base_mri / "bags_nfl.npy")

# Index trimming (as in your script)
idx_plasma = np.delete(np.arange(0,113,1), [94,102,110])
ptau217, nfl = ptau217[idx_plasma], nfl[idx_plasma]
bags_ptau, bags_nfl = bags_ptau[idx_plasma], bags_nfl[idx_plasma]

plasma_specs = [
    dict(name="Phosphorylated-tau", x=np.log(ptau217), y=bags_ptau,
         xlabel="Log([plasma p-tau217])", ylabel="BAG (years)"),
    dict(name="Neurofilament light chain", x=np.log(nfl), y=bags_nfl,
         xlabel="Log([plasma NfL])", ylabel="BAG (years)")
]

plasma_results = []
for spec in plasma_specs:
    x_c, y_c, _, _ = remove_outliers_pair(spec['x'], spec['y'])
    r, p = pearson_safe(x_c, y_c)
    r2 = r**2 if np.isfinite(r) else np.nan
    f2 = (r2/(1-r2)) if (np.isfinite(r2) and r2 < 1) else (np.inf if np.isfinite(r2) and r2==1 else np.nan)
    plasma_results.append(dict(name=spec['name'], r=r, p_raw=p, f2=f2, N=len(x_c),
                               x=x_c, y=y_c, xlabel=spec['xlabel'], ylabel=spec['ylabel']))

pvals = [d['p_raw'] if np.isfinite(d['p_raw']) else 1.0 for d in plasma_results]
_, p_fdr, _, _ = multipletests(pvals, method='fdr_bh')
for d, pf in zip(plasma_results, p_fdr): d['p_fdr'] = pf

plasma_summary = pd.DataFrame([
    dict(Measure=d['name'],
         r=(np.nan if not np.isfinite(d['r']) else round(d['r'],3)),
         p_raw=("nan" if not np.isfinite(d['p_raw']) else f"{d['p_raw']:.5f}".rstrip('0').rstrip('.')),
         p_FDR=("nan" if not np.isfinite(d['p_fdr']) else ("p<0.001" if d['p_fdr']<1e-3 else f"{d['p_fdr']:.5f}".rstrip('0').rstrip('.'))),
         f2=(np.nan if not np.isfinite(d['f2']) else (round(d['f2'],3) if d['f2']!=np.inf else "inf")),
         N=d['N'])
    for d in plasma_results
])
print("\nPlasma correlates (no outliers):\n", plasma_summary.to_string(index=False))

for d in plasma_results:
    scatter_reg(d['x'], d['y'], d['xlabel'], d['ylabel'], d['name'],
                p_display=d['p_fdr'])



