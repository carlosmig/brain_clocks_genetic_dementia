# ============================================================
# EEG-FC Brain Age Pipeline
# - Train Non-LinearSVR on ALL HCs with repeated 5-fold CV
# - Compute BAGs, age-bias correction (fit in HCs)
# - Restrict AD + HCs to Global South (Region == 'South')
# - GLM to remove Age + Sex from corrected BAGs
# - Remove outliers (±2 SD within group)
# - Plots: Real vs Predicted Age; Violin of adjusted BAGs
# - Welch's ANOVA + pairwise Welch t-tests with FDR
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
# 0) Load EEG & MEG FC tables
# ----------------------------
EEG_CSV = r"EEG_data.csv"
MEG_CSV = r"MEG_data.csv"

df_eeg = pd.read_csv(EEG_CSV)
df_meg = pd.read_csv(MEG_CSV)

# EEG: V1 only
if 'Visit' in df_eeg.columns:
    df_eeg = df_eeg[df_eeg['Visit'].astype(str).str.upper().eq('V1')].copy()

for req in ['Age','Sex','Diagnosis','Region']:
    if req not in df_eeg.columns: raise ValueError(f"EEG CSV missing column: {req}")
    if req not in df_meg.columns: raise ValueError(f"MEG CSV missing column: {req}")

fc_eeg_cols = [c for c in df_eeg.columns if c.startswith('FC_')]
fc_meg_cols = [c for c in df_meg.columns if c.startswith('FC_')]
if len(fc_eeg_cols)==0: raise ValueError("No FC_ columns in EEG CSV.")
if len(fc_meg_cols)==0: raise ValueError("No FC_ columns in MEG CSV.")

# Overlapping features (so the EEG model can score MEG)
fc_cols = [c for c in fc_eeg_cols if c in set(fc_meg_cols)]
if len(fc_cols)==0:
    raise ValueError("No overlapping FC_ features between EEG and MEG.")

# ----------------------------
# 1) Helpers
# ----------------------------
def to_binary_female(series):
    """Sex → female=1, male=0."""
    s = series.copy()
    if s.dtype.kind in {'i','u','f'}:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[s == 2] = 1.0   # 1=male, 2=female
        out.loc[s == 1] = 0.0
        out = out.fillna(s.where(s.isin([0,1]), np.nan))
        return out.fillna(0.0)
    return s.astype(str).str.upper().map({'F':1.0,'FEMALE':1.0,'M':0.0,'MALE':0.0}).fillna(0.0)

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

def hedges_g(v1, v2):
    n1, n2 = len(v1), len(v2)
    s1, s2 = np.nanvar(v1, ddof=1), np.nanvar(v2, ddof=1)
    sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(n1+n2-2, 1))
    d = (np.nanmean(v1) - np.nanmean(v2)) / sp if sp > 0 else np.nan
    J = 1 - (3 / (4*(n1+n2) - 9)) if (n1+n2) > 2 else 1.0
    return d * J

# ----------------------------
# 2) Prepare arrays
# ----------------------------
# EEG
df_eeg['Age'] = pd.to_numeric(df_eeg['Age'], errors='coerce')
df_eeg['Sex_bin'] = to_binary_female(df_eeg['Sex'])
df_eeg['Diagnosis'] = df_eeg['Diagnosis'].astype(str)
df_eeg['Region'] = df_eeg['Region'].astype(str)

X_eeg_all = np.nan_to_num(df_eeg[fc_cols].to_numpy(), nan=0.0)
Y_eeg_all = df_eeg['Age'].to_numpy()
L_eeg_all = df_eeg['Diagnosis'].to_numpy()
R_eeg_all = df_eeg['Region'].to_numpy()
S_eeg_all = df_eeg['Sex_bin'].to_numpy()

# MEG (will be scored by the EEG model each fold)
df_meg['Age'] = pd.to_numeric(df_meg['Age'], errors='coerce')
df_meg['Sex_bin'] = to_binary_female(df_meg['Sex'])
df_meg['Diagnosis'] = df_meg['Diagnosis'].astype(str)
df_meg['Region'] = df_meg['Region'].astype(str)

X_meg_all = np.nan_to_num(df_meg[fc_cols].to_numpy(), nan=0.0)
Y_meg_all = df_meg['Age'].to_numpy()
S_meg_all = df_meg['Sex_bin'].to_numpy()

# Train set = EEG HCs (North+South)
is_hc = (L_eeg_all == 'HCs')
X_hc = X_eeg_all[is_hc, :]
Y_hc = Y_eeg_all[is_hc]
if X_hc.shape[0] < 10:
    raise ValueError("Not enough EEG HCs to train the model.")

# ----------------------------
# 3) Repeated 5-fold CV on HCs with min_corr=0.3
#    Age-bias fit AND BAG computation happen inside each fold.
#    We store fold×rep predictions and BAGs, then average per subject.
# ----------------------------
min_corr = 0.3
reps = 15
n_splits = 5

S_eeg = len(Y_eeg_all)
S_meg = len(Y_meg_all)

Ypred_eeg_mat = np.full((S_eeg, reps*n_splits), np.nan)
Ypred_meg_mat = np.full((S_meg, reps*n_splits), np.nan)
BAGbc_eeg_mat = np.full((S_eeg, reps*n_splits), np.nan)
BAGbc_meg_mat = np.full((S_meg, reps*n_splits), np.nan)

rreps = np.zeros(reps)
ereps = np.zeros(reps)

col_id = 0
for k in range(reps):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=k)
    rtemp_pool, error_pool = [], []

    for tr_idx, te_idx in cv.split(Y_hc, Y_hc):
        X_tr, Y_tr = X_hc[tr_idx, :], Y_hc[tr_idx]
        X_te, Y_te = X_hc[te_idx, :], Y_hc[te_idx]

        # Feature filter (|r|>=0.3) computed on TRAIN only
        r_vec = np.zeros(X_tr.shape[1])
        for i in range(X_tr.shape[1]):
            with np.errstate(invalid='ignore'):
                r, _ = stats.pearsonr(X_tr[:, i], Y_tr)
            r_vec[i] = 0 if np.isnan(r) else r
        keep = np.abs(r_vec) >= min_corr

        X_tr_m = X_tr[:, keep]
        X_te_m = X_te[:, keep]
        X_eeg_m = X_eeg_all[:, keep]
        X_meg_m = X_meg_all[:, keep]

        # SVR (poly=2)
        regr = svm.SVR(max_iter=10000, C=2, kernel='poly', degree=2, epsilon=1e-4)
        regr.fit(X_tr_m, Y_tr)

        # HC test metrics (for logging)
        Y_pred_te = regr.predict(X_te_m)
        rtemp_pool.append(stats.pearsonr(Y_te, Y_pred_te)[0])
        error_pool.append(np.mean(np.abs(Y_te - Y_pred_te)))

        # -------- Age-bias correction parameters from TRAIN only
        Y_pred_tr = regr.predict(X_tr_m)
        gap_train = Y_pred_tr - Y_tr
        X_bias = sm.add_constant(Y_tr)           # [1, age]
        mdl_bias = sm.OLS(gap_train, X_bias).fit()
        b_bias, a_bias = mdl_bias.params[0], mdl_bias.params[1]

        # -------- EEG: predictions and bias-corrected BAGs
        Y_pred_all_eeg = regr.predict(X_eeg_m)
        BAG_raw_eeg = Y_pred_all_eeg - Y_eeg_all
        BAG_bc_eeg = BAG_raw_eeg - (a_bias * Y_eeg_all + b_bias)

        Ypred_eeg_mat[:, col_id] = Y_pred_all_eeg
        BAGbc_eeg_mat[:, col_id] = BAG_bc_eeg

        # -------- MEG: predictions and bias-corrected BAGs (using EEG fold bias)
        Y_pred_all_meg = regr.predict(X_meg_m)
        BAG_raw_meg = Y_pred_all_meg - Y_meg_all
        BAG_bc_meg = BAG_raw_meg - (a_bias * Y_meg_all + b_bias)

        Ypred_meg_mat[:, col_id] = Y_pred_all_meg
        BAGbc_meg_mat[:, col_id] = BAG_bc_meg

        col_id += 1

    rreps[k] = np.nanmean(rtemp_pool)
    ereps[k] = np.nanmean(error_pool)
    print(f"EEG rep {k+1}/{reps}: r={rreps[k]:.3f}, MAE={ereps[k]:.3f}")

# Aggregate over folds×reps (per subject)
Y_pred_eeg = np.nanmean(Ypred_eeg_mat, axis=1)
Y_pred_meg = np.nanmean(Ypred_meg_mat, axis=1)
BAG_bc_eeg = np.nanmean(BAGbc_eeg_mat, axis=1)
BAG_bc_meg = np.nanmean(BAGbc_meg_mat, axis=1)

# HC performance summary
hc_r = float(np.nanmean(rreps))
hc_mae = float(np.nanmean(ereps))
cohens_f2 = hc_r**2 / (1 - hc_r**2) if (1 - hc_r**2) > 0 else np.nan
print(f"\nEEG HC aggregate performance: r={hc_r:.3f}, MAE={hc_mae:.3f}, Cohen's f²={cohens_f2:.3f}")

# ----------------------------
# 4) GLM (Age, Sex) on combined EEG+MEG (using bias-corrected BAGs)
# ----------------------------
res_eeg = pd.DataFrame({
    'Modality': 'EEG',
    'Diagnosis': L_eeg_all,
    'Region': R_eeg_all,
    'Age': Y_eeg_all,
    'Sex_bin': S_eeg_all,
    'PredAge': Y_pred_eeg,
    'BAG_biasCorr': BAG_bc_eeg
})
res_meg = pd.DataFrame({
    'Modality': 'MEG',
    'Diagnosis': df_meg['Diagnosis'].to_numpy(),
    'Region': df_meg['Region'].to_numpy(),
    'Age': Y_meg_all,
    'Sex_bin': S_meg_all,
    'PredAge': Y_pred_meg,
    'BAG_biasCorr': BAG_bc_meg
})
res_all = pd.concat([res_eeg, res_meg], ignore_index=True)

X_glm = sm.add_constant(np.column_stack([res_all['Age'].to_numpy(),
                                         res_all['Sex_bin'].to_numpy()]))
mdl_glm = sm.OLS(res_all['BAG_biasCorr'].to_numpy(), X_glm).fit()
res_all['BAG_adj'] = mdl_glm.resid

# ----------------------------
# 5) Plot: EEG real vs predicted age
# ----------------------------
plt.figure(figsize=(7.5, 6))
plt.scatter(Y_eeg_all, Y_pred_eeg, s=40, alpha=0.7, edgecolor='black')
aa, bb, _ = stats.linregress(Y_eeg_all[np.isfinite(Y_pred_eeg)],
                             Y_pred_eeg[np.isfinite(Y_pred_eeg)])[0:3]
plt.plot(Y_eeg_all, bb + aa * Y_eeg_all, color='crimson', lw=2, ls='dashed')
plt.xlabel('Chronological age (years)'); plt.ylabel('Predicted age (years)')
plt.title('EEG FC model: Real vs Predicted Age (HC-trained, |r|≥0.3)')
annot = f"HC r = {hc_r:.3f}\nHC MAE = {hc_mae:.3f}\nCohen\'s f² = {cohens_f2:.3f}"
plt.text(np.nanmin(Y_eeg_all)+1, np.nanmax(Y_eeg_all)-5, annot, bbox=dict(facecolor='white', alpha=0.7))
plt.grid(True, ls='--', alpha=0.5); plt.tight_layout(); plt.show()

# ----------------------------
# 6) EEG stats/plots
# ----------------------------
res_eeg_only = res_all.query("Modality == 'EEG'").copy()

# Collapse DS subtypes → DS
res_eeg_only.loc[res_eeg_only['Diagnosis'].isin(['aDS','pDS','dDS']), 'Diagnosis'] = 'DS'

# Target EEG groups
mask_hc_south = (res_eeg_only['Diagnosis']=='HCs') & (res_eeg_only['Region']=='South')
mask_ad_south = (res_eeg_only['Diagnosis']=='AD') & (res_eeg_only['Region']=='South')
mask_ncar     = (res_eeg_only['Diagnosis']=='nCar')
mask_acar     = (res_eeg_only['Diagnosis']=='aCar')
mask_ds       = (res_eeg_only['Diagnosis']=='DS')

res_eeg_keep = res_eeg_only.loc[mask_hc_south | mask_ncar | mask_acar | mask_ad_south | mask_ds].copy()

# Label HCs South for plotting
res_eeg_keep.loc[mask_hc_south.loc[res_eeg_keep.index], 'Diagnosis'] = 'HCs_South'

# Outlier removal by group (±2 SD)
EEG_ORDER = ['HCs_South','nCar','aCar','AD','DS']
res_eeg_clean = remove_group_outliers(res_eeg_keep, 'BAG_adj', 'Diagnosis', z=2.0)
present_eeg = [g for g in EEG_ORDER if g in res_eeg_clean['Diagnosis'].unique()]

# Welch ANOVA
groups_vals = [res_eeg_clean.loc[res_eeg_clean['Diagnosis']==g, 'BAG_adj'].values for g in present_eeg]
anova_res = anova_oneway(groups_vals, use_var='unequal', welch_correction=True)
print("\nEEG — Welch's ANOVA on adjusted BAGs:")
print(anova_res)

# Pairwise Welch t-tests + FDR
def pairwise_welch(df, groups, col='BAG_adj'):
    pairs, tvals, pvals, diffs, gs, ns = [], [], [], [], [], []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            v1 = df.loc[df['Diagnosis']==g1, col].values
            v2 = df.loc[df['Diagnosis']==g2, col].values
            t, p = stats.ttest_ind(v1, v2, equal_var=False, nan_policy='omit')
            diff = np.nanmean(v1) - np.nanmean(v2)
            g = hedges_g(v1, v2)
            pairs.append((g1,g2)); tvals.append(t); pvals.append(p)
            diffs.append(diff); gs.append(g); ns.append((len(v1), len(v2)))
    rej, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    return pairs, tvals, pvals, rej, p_fdr, diffs, gs, ns

pairs, tvals, pvals, rej, p_fdr, diffs, gs, ns = pairwise_welch(res_eeg_clean, present_eeg)
print("\nEEG — Pairwise Welch t-tests on adjusted BAGs (FDR):")
for (g1,g2), t, p0, pf, rj, dff, g, (n1,n2) in sorted(
    zip(pairs,tvals,pvals,p_fdr,rej,diffs,gs,ns), key=lambda x: x[3]
):
    print(f"{g1} vs {g2}:  Δmean={dff:+.2f}  t={t:.2f}  g={g:.2f}  "
          f"p={p0:.4g}  p_FDR={pf:.4g}  reject={bool(rj)}  (n1={n1}, n2={n2})")

# Violin (EEG)
plt.figure(figsize=(9, 6))
sns.violinplot(data=res_eeg_clean, x='Diagnosis', y='BAG_adj', order=present_eeg, inner=None, cut=0)
sns.pointplot(data=res_eeg_clean, x='Diagnosis', y='BAG_adj', order=present_eeg,
              join=False, estimator=np.mean, errorbar=('ci',95))
plt.axhline(0, ls='--', lw=1, color='k')
plt.title("EEG Brain Age Gaps (Adjusted for Age & Sex)\nOutliers removed (±2 SD)")
plt.ylabel("BAG (years)"); plt.xlabel("")
plt.tight_layout(); plt.show()

print("\nEEG — Group means (adjusted BAGs, outliers removed):")
for g in present_eeg:
    v = res_eeg_clean.loc[res_eeg_clean['Diagnosis']==g, 'BAG_adj'].values
    print(f"{g}: mean={np.nanmean(v):+.2f}, sd={np.nanstd(v, ddof=1):.2f}, n={len(v)}")

# ----------------------------
# 7) MEG: HCs vs DS
# ----------------------------
res_meg_only = res_all.query("Modality == 'MEG'").copy()
res_meg_only = res_meg_only[res_meg_only['Diagnosis'].isin(['HCs','DS'])].copy()
res_meg_clean = remove_group_outliers(res_meg_only, 'BAG_adj', 'Diagnosis', z=2.0)

present_meg = [g for g in ['HCs','DS'] if g in res_meg_clean['Diagnosis'].unique()]

if all(g in present_meg for g in ['HCs','DS']):
    v_hc = res_meg_clean.loc[res_meg_clean['Diagnosis']=='HCs', 'BAG_adj'].values
    v_ds = res_meg_clean.loc[res_meg_clean['Diagnosis']=='DS', 'BAG_adj'].values
    t_meg, p_meg = stats.ttest_ind(v_hc, v_ds, equal_var=False, nan_policy='omit')
    g_meg = hedges_g(v_hc, v_ds)
    dmean = np.nanmean(v_hc) - np.nanmean(v_ds)
    print("\nMEG — Welch t-test (HCs vs DS) on adjusted BAGs:")
    print(f"Δmean={dmean:+.2f}  t={t_meg:.2f}  g={g_meg:.2f}  p={p_meg:.4g}  "
          f"(n_HC={len(v_hc)}, n_DS={len(v_ds)})")
else:
    print("\nMEG — Not enough groups for HCs vs DS comparison.")

plt.figure(figsize=(6.5, 5.5))
sns.violinplot(data=res_meg_clean, x='Diagnosis', y='BAG_adj', order=present_meg, inner=None, cut=0)
sns.pointplot(data=res_meg_clean, x='Diagnosis', y='BAG_adj', order=present_meg,
              join=False, estimator=np.mean, errorbar=('ci',95))
plt.axhline(0, ls='--', lw=1, color='k')
plt.title("MEG Brain Age Gaps (Adjusted for Age & Sex)\nOutliers removed (±2 SD)")
plt.ylabel("BAG (years)"); plt.xlabel("")
plt.tight_layout(); plt.show()

print("\nMEG — Group means (adjusted BAGs, outliers removed):")
for g in present_meg:
    v = res_meg_clean.loc[res_meg_clean['Diagnosis']==g, 'BAG_adj'].values
    print(f"{g}: mean={np.nanmean(v):+.2f}, sd={np.nanstd(v, ddof=1):.2f}, n={len(v)}")
