# ============================================================
# MRI-GMV Brain Age Pipeline
# - Train LinearSVR on HCs with repeated 5-fold CV
# - In each fold: fit age-bias on HC TRAIN, compute bias-corrected BAGs for ALL
# - Average BAGs across folds×reps
# - Restrict AD + HCs to Region == 'North' before GLM
# - GLM removes Age + Sex from bias-corrected BAGs
# - Outlier removal (±2 SD within group), ANOVA, post hocs, plots
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 0) Load GMV dataframe
# ----------------------------
DATA_CSV = r"MRI_data.csv"
df = pd.read_csv(DATA_CSV)

roi_cols = [c for c in df.columns if c.startswith('ROI_')]
if len(roi_cols) == 0:
    raise ValueError("No ROI_ columns found in MRI_data.csv.")

# Features & targets
X_all = np.nan_to_num(df[roi_cols].to_numpy(), nan=0.0)
Y_all = df['Age'].to_numpy()
labels_all = df['Diagnosis'].astype(str).to_numpy()
Region_all = df['Region'].astype(str).fillna('Unknown').to_numpy()

# Sex → female=1, male=0
def to_binary_female(series):
    s = series.copy()
    if s.dtype.kind in {'i','u','f'}:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[s == 2] = 1.0  # 1=male, 2=female
        out.loc[s == 1] = 0.0
        out = out.fillna(s.where(s.isin([0,1]), np.nan))
        return out.fillna(0.0)
    return s.astype(str).str.upper().map({'F':1.0,'FEMALE':1.0,'M':0.0,'MALE':0.0}).fillna(0.0)

Sex_all = to_binary_female(df['Sex']).to_numpy()

# ----------------------------
# 1) CV on HCs with in-fold bias correction
# ----------------------------
GROUPS_ORDER = ['HCs','AD','aDS','pDS','dDS']
is_hc = (labels_all == 'HCs')
X_hc = X_all[is_hc, :]
Y_hc = Y_all[is_hc]
if X_hc.shape[0] < 10:
    raise ValueError("Not enough HCs to train.")

reps = 15
n_splits = 5
S_all = len(Y_all)

Ypred_mat   = np.full((S_all, reps*n_splits), np.nan)
BAGbc_mat   = np.full((S_all, reps*n_splits), np.nan)
rreps = np.zeros(reps)
ereps = np.zeros(reps)

col_id = 0
for k in range(reps):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=k)
    rtemp_pool, error_pool = [], []

    for tr_idx, te_idx in cv.split(Y_hc, Y_hc):
        X_tr, Y_tr = X_hc[tr_idx, :], Y_hc[tr_idx]
        X_te, Y_te = X_hc[te_idx, :], Y_hc[te_idx]

        # Simple per-feature filter (keep all; raise threshold if you want pruning)
        r_vec = np.zeros(X_tr.shape[1])
        for i in range(X_tr.shape[1]):
            with np.errstate(invalid='ignore'):
                r, _ = stats.pearsonr(X_tr[:, i], Y_tr)
            r_vec[i] = 0 if np.isnan(r) else r
        keep = np.abs(r_vec) > 0.0

        X_tr_m = X_tr[:, keep]
        X_te_m = X_te[:, keep]
        X_all_m = X_all[:, keep]

        regr = svm.LinearSVR(max_iter=5000, C=1.0, fit_intercept=True, random_state=k)
        regr.fit(X_tr_m, Y_tr)

        # HC test metrics
        Y_pred_te = regr.predict(X_te_m)
        rtemp_pool.append(stats.pearsonr(Y_te, Y_pred_te)[0])
        error_pool.append(np.mean(np.abs(Y_te - Y_pred_te)))

        # ---- Age-bias from TRAIN fold only
        Y_pred_tr = regr.predict(X_tr_m)
        gap_train = Y_pred_tr - Y_tr
        X_bias = sm.add_constant(Y_tr)  # [1, age]
        mdl_bias = sm.OLS(gap_train, X_bias).fit()
        b_bias, a_bias = mdl_bias.params[0], mdl_bias.params[1]

        # ---- Predict EVERYONE with this fold model and apply this fold's bias
        Y_pred_all = regr.predict(X_all_m)
        BAG_raw = Y_pred_all - Y_all
        BAG_bc  = BAG_raw - (a_bias * Y_all + b_bias)

        Ypred_mat[:, col_id] = Y_pred_all
        BAGbc_mat[:, col_id] = BAG_bc
        col_id += 1

    rreps[k] = np.nanmean(rtemp_pool)
    ereps[k] = np.nanmean(error_pool)
    print(f"rep {k+1}/{reps}  r={rreps[k]:.3f}  MAE={ereps[k]:.3f}")

# Aggregate predictions and bias-corrected BAGs across folds×reps
Y_pred_mean = np.nanmean(Ypred_mat, axis=1)
BAG_bc_mean = np.nanmean(BAGbc_mat, axis=1)

# HC performance summary
hc_r = float(np.nanmean(rreps))
hc_mae = float(np.nanmean(ereps))
cohens_f2 = hc_r**2 / (1 - hc_r**2) if (1 - hc_r**2) > 0 else np.nan
print(f"\nHC aggregate performance: r={hc_r:.3f}, MAE={hc_mae:.3f}, Cohen's f²={cohens_f2:.3f}")

# ----------------------------
# 2) Restrict AD + HCs to Global North before GLM
# ----------------------------
keep_mask = np.ones(S_all, dtype=bool)
ad_or_hc = (labels_all == 'AD') | (labels_all == 'HCs')
keep_mask[ad_or_hc & (Region_all != 'North')] = False

res_all = pd.DataFrame({
    'Diagnosis': labels_all,
    'Region': Region_all,
    'Age': Y_all,
    'Sex_bin': Sex_all,
    'PredAge': Y_pred_mean,
    'BAG_biasCorr': BAG_bc_mean
})
res_filt = res_all.loc[keep_mask].reset_index(drop=True)

# ----------------------------
# 3) GLM to remove Age + Sex (on filtered set)
# ----------------------------
X_glm = sm.add_constant(np.column_stack([res_filt['Age'].to_numpy(),
                                         res_filt['Sex_bin'].to_numpy()]))
mdl_glm = sm.OLS(res_filt['BAG_biasCorr'].to_numpy(), X_glm).fit()
res_filt['BAG_adj'] = mdl_glm.resid

# ----------------------------
# 4) Plot: Real vs Predicted Age
# ----------------------------
plt.figure(figsize=(7.5, 6))
plt.scatter(Y_all, Y_pred_mean, s=40, alpha=0.7, edgecolor='black')
aa, bb, _ = stats.linregress(Y_all[np.isfinite(Y_pred_mean)],
                             Y_pred_mean[np.isfinite(Y_pred_mean)])[0:3]
plt.plot(Y_all, bb + aa * Y_all, color='crimson', lw=2, ls='dashed')
plt.xlabel('Chronological age (years)')
plt.ylabel('Predicted age (years)')
plt.title('GMV model: Real vs Predicted Age (HC-trained)')
annot = f"HC r = {hc_r:.3f}\nHC MAE = {hc_mae:.3f}\nCohen\'s f² = {cohens_f2:.3f}"
plt.text(np.nanmin(Y_all)+1, np.nanmax(Y_all)-5, annot, bbox=dict(facecolor='white', alpha=0.7))
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------
# 5) Outliers, ANOVA, post hocs
# ----------------------------
def remove_group_outliers(df_in, value_col, group_col, z=2.0):
    df_o = df_in.copy()
    keep = np.ones(len(df_o), dtype=bool)
    for g, sub in df_o.groupby(group_col):
        v = sub[value_col].values
        m = np.nanmean(v); s = np.nanstd(v, ddof=1)
        if np.isfinite(s) and s > 0:
            msk = np.abs(df_o.loc[df_o[group_col]==g, value_col] - m) <= z*s
        else:
            msk = pd.Series(True, index=df_o.loc[df_o[group_col]==g].index)
        keep[df_o[group_col]==g] = msk.astype(bool).values
    return df_o.loc[keep].reset_index(drop=True)

res_clean = remove_group_outliers(res_filt, 'BAG_adj', 'Diagnosis', z=2.0)
present_groups = [g for g in GROUPS_ORDER if g in res_clean['Diagnosis'].unique()]

# Welch's ANOVA
groups_vals = [res_clean.loc[res_clean['Diagnosis']==g, 'BAG_adj'].values for g in present_groups]
anova_res = anova_oneway(groups_vals, use_var='unequal', welch_correction=True)
print("\nWelch's ANOVA on adjusted BAGs (BAG_adj):")
print(anova_res)

# Pairwise Welch + FDR
def hedges_g(v1, v2):
    n1, n2 = len(v1), len(v2)
    s1, s2 = np.nanvar(v1, ddof=1), np.nanvar(v2, ddof=1)
    sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(n1+n2-2, 1))
    d = (np.nanmean(v1) - np.nanmean(v2)) / sp if sp > 0 else np.nan
    J = 1 - (3 / (4*(n1+n2) - 9)) if (n1+n2) > 2 else 1.0
    return d * J

pairs, tvals, pvals, diffs, gs, ns = [], [], [], [], [], []
for i in range(len(present_groups)):
    for j in range(i+1, len(present_groups)):
        g1, g2 = present_groups[i], present_groups[j]
        v1 = res_clean.loc[res_clean['Diagnosis']==g1, 'BAG_adj'].values
        v2 = res_clean.loc[res_clean['Diagnosis']==g2, 'BAG_adj'].values
        t, p = stats.ttest_ind(v1, v2, equal_var=False, nan_policy='omit')
        diff = np.nanmean(v1) - np.nanmean(v2)
        g = hedges_g(v1, v2)
        pairs.append((g1, g2)); tvals.append(t); pvals.append(p)
        diffs.append(diff); gs.append(g); ns.append((len(v1), len(v2)))

rej, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

print("\nPairwise Welch t-tests on adjusted BAGs (FDR corrected):")
for (g1, g2), t, p0, pf, rj, dff, g, (n1,n2) in sorted(
    zip(pairs, tvals, pvals, p_fdr, rej, diffs, gs, ns),
    key=lambda x: x[3]
):
    print(f"{g1} vs {g2}:  Δmean={dff:+.2f}  t={t:.2f}  g(Hedges)={g:.2f}  "
          f"p={p0:.4g}  p_FDR={pf:.4g}  reject={bool(rj)}  (n1={n1}, n2={n2})")

# ----------------------------
# 6) Violin plot (adjusted BAGs)
# ----------------------------
plt.figure(figsize=(9, 6))
sns.violinplot(data=res_clean, x='Diagnosis', y='BAG_adj', order=present_groups, inner=None, cut=0)
sns.pointplot(data=res_clean, x='Diagnosis', y='BAG_adj', order=present_groups,
              join=False, estimator=np.mean, errorbar=('ci', 95))
plt.axhline(0, ls='--', lw=1, color='k')
plt.title("GMV Brain Age Gaps (Adjusted for Age & Sex)\nOutliers removed (±2 SD)\n(AD & HCs restricted to North)")
plt.ylabel("BAG (years)")
plt.xlabel("")
plt.tight_layout()
plt.show()

# ----------------------------
# 7) Group means (summary)
# ----------------------------
print("\nGroup means of adjusted BAGs (outliers removed; AD & HCs = North only):")
for g in present_groups:
    v = res_clean.loc[res_clean['Diagnosis']==g, 'BAG_adj'].values
    print(f"{g}: mean={np.nanmean(v):+.2f}, sd={np.nanstd(v, ddof=1):.2f}, n={len(v)}")
