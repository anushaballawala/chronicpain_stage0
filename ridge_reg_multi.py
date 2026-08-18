import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import sys
sys.path.append("/home/jiahuang/test-code-hazel/")
import gen_fxns
from plot_pca import plot_pca_loadings
import h5py
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sp
import sklearn
from datetime import date
import pandas as pd 
from datetime import timedelta
from scipy.stats import zscore
import seaborn as sns 
import pickle 
import os 
from datetime import datetime

from scipy.signal import welch
import pandas as pd
from fooof import FOOOF, FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fm, get_band_peak_fg

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

bandref = dict({'delta':[3,4], 
                 'theta':[4,8], 
                 'alpha':[8,12], 
                 'beta':[12,30], 
                 'low_gamma':[30,50], 
                 'high_gamma':[70,110]})
ptIDs = ['RCS02', 'RCS03', 'RCS04', 'RCS05', 'RCS06', 'RCS07','RCS08','RCS09']
def build_long_df_zscored(pt, bandref, r2_threshold=0.9):
    record_path = f"/userdata/jiahuang/pain-data/Stage1-test/{pt}/records_df_fooof_{pt}.pkl"
    records_df = pd.read_pickle(record_path)
    records_df['roi'] = records_df['ch_label'].str.replace(r'_\d+$', '', regex=True)
    records_df = records_df[records_df['r2'] >= r2_threshold]

    freqs_welch = records_df['fooof'].iloc[0].freqs
    bands = list(bandref.keys())
    ch_labels = sorted(records_df['ch_label'].unique())
    rows = []
    thresh_low_aff = np.percentile(records_df['affective'], 30)
    thresh_high_aff = np.percentile(records_df['affective'], 70)

    thresh_low_sens = np.percentile(records_df['sensory'], 30)
    thresh_high_sens = np.percentile(records_df['sensory'], 70)


    for ch in ch_labels:
        sub = records_df[records_df['ch_label'] == ch].sort_values('trial_idx').reset_index(drop=True)
        
        corrected_stack = np.stack([r['fooof'].fooofed_spectrum_ - r['fooof']._ap_fit for _, r in sub.iterrows()])
        raw_stack       = np.stack([r['fooof'].power_spectrum for _, r in sub.iterrows()])  # trials x freq
        corrected_z = (corrected_stack - corrected_stack.mean(axis=0)) / corrected_stack.std(axis=0)
        raw_z       = (raw_stack       - raw_stack.mean(axis=0))       / raw_stack.std(axis=0)

        for i, row in sub.iterrows():
            for band in bands:
                fmin, fmax = bandref[band]
                idxf = np.where((freqs_welch >= fmin) & (freqs_welch <= fmax))[0]
                rows.append({
                    'ptID':             pt,
                    'trial_idx':        row['trial_idx'],
                    'ch_label':         ch,
                    'roi':              row['roi'],
                    'band':             band,
                    'band_power':       np.nanmean(corrected_stack[i, idxf]),
                    'band_power_z':     np.nanmean(corrected_z[i, idxf]),
                    'band_power_raw':   np.nanmean(raw_stack[i, idxf]),
                    'band_power_raw_z': np.nanmean(raw_z[i, idxf]),
                    'aff_score':        row['affective'],
                    'sens_score':       row['sensory'],
                    'stratified_group_aff':int(row['affective']>=thresh_high_aff) + int(row['affective']>=thresh_low_aff),
                    'stratified_group_sens':int(row['sensory']>=thresh_high_sens) + int(row['sensory']>=thresh_low_sens)
                })

    df = pd.DataFrame(rows)
    return df

long_df = pd.concat([build_long_df_zscored(pt, bandref) for pt in ptIDs], ignore_index=True)

long_df['feat_col'] = long_df['band'] + '__' + long_df['roi']

scores = long_df.drop_duplicates(['ptID', 'trial_idx'])[['ptID', 'trial_idx', 'aff_score', 'sens_score','stratified_group_aff','stratified_group_sens']]

# ---------------- df1: ROI-level (mean over channels within roi) ----------------
agg_roi = (
    long_df.groupby(['ptID', 'trial_idx', 'feat_col'])[['band_power_z', 'band_power_raw_z']]
    .mean()
    .reset_index()
)

roi_corrected = agg_roi.pivot_table(index=['ptID', 'trial_idx'], columns='feat_col', values='band_power_z')
roi_raw       = agg_roi.pivot_table(index=['ptID', 'trial_idx'], columns='feat_col', values='band_power_raw_z')

roi_corrected.columns = [f"{c}_corrected_z" for c in roi_corrected.columns]
roi_raw.columns       = [f"{c}_raw_z" for c in roi_raw.columns]

feat_df_roi = pd.concat([roi_corrected, roi_raw], axis=1).reset_index()
feat_df_roi = feat_df_roi.merge(scores, on=['ptID', 'trial_idx'])

# ---------------- df2: channel-level (no aggregation, keep all channels) ----------------
ch_corrected = long_df.pivot_table(
    index=['ptID', 'trial_idx', 'ch_label'], columns='feat_col', values='band_power_z', aggfunc='mean'
)
ch_raw = long_df.pivot_table(
    index=['ptID', 'trial_idx', 'ch_label'], columns='feat_col', values='band_power_raw_z', aggfunc='mean'
)

ch_corrected.columns = [f"{c}_corrected_z" for c in ch_corrected.columns]
ch_raw.columns       = [f"{c}_raw_z" for c in ch_raw.columns]

feat_df_channel = pd.concat([ch_corrected, ch_raw], axis=1).reset_index()
feat_df_channel = feat_df_channel.merge(scores, on=['ptID', 'trial_idx'])

coverage = feat_df_roi[['ptID']+[c for c in feat_df_roi.columns if c.endswith('_corrected_z')]].groupby("ptID").apply(lambda g: g.iloc[:,1:].notna().any())
# coverage.loc['pct_true'] = coverage.mean() * 100
coverage_frac = coverage.mean()
keep_rois = coverage_frac[coverage_frac >= 0.625].index.tolist()
rois_only = sorted(set(
    c.split('__', 1)[1].replace('_corrected_z', '').replace('_raw_z', '')
    for c in keep_rois
))

def run_stratified_cv(X, y_aff, y_sens, n_splits, q, alphas_grid, random_state=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_alphas_aff, fold_alphas_sens = [], []
    y_pred_aff = np.full(len(y_aff), np.nan)
    y_pred_sens = np.full(len(y_sens), np.nan)

    y_bins_aff = pd.qcut(y_aff, q=q, labels=False, duplicates='drop')
    y_bins_sens = pd.qcut(y_sens, q=q, labels=False, duplicates='drop')

    for train_idx, test_idx in skf.split(X, y_bins_aff):
        imputer = SimpleImputer(strategy='mean').fit(X[train_idx])
        X_train, X_test = imputer.transform(X[train_idx]), imputer.transform(X[test_idx])
        m = RidgeCV(alphas=alphas_grid).fit(X_train, y_aff[train_idx])
        y_pred_aff[test_idx] = m.predict(X_test)
        fold_alphas_aff.append(m.alpha_)

    for train_idx, test_idx in skf.split(X, y_bins_sens):
        imputer = SimpleImputer(strategy='mean').fit(X[train_idx])
        X_train, X_test = imputer.transform(X[train_idx]), imputer.transform(X[test_idx])
        m = RidgeCV(alphas=alphas_grid).fit(X_train, y_sens[train_idx])
        y_pred_sens[test_idx] = m.predict(X_test)
        fold_alphas_sens.append(m.alpha_)

    return {
        'n_splits': n_splits,
        'q': q,
        'r2_aff': r2_score(y_aff, y_pred_aff),
        'r2_sens': r2_score(y_sens, y_pred_sens),
        'alphas_aff': fold_alphas_aff,
        'alphas_sens': fold_alphas_sens,
        'alpha_aff_mode': pd.Series(fold_alphas_aff).mode()[0],
        'alpha_sens_mode': pd.Series(fold_alphas_sens).mode()[0],
    }
keep_cols_final = [c for c in feat_df_roi.columns if any(roi in c for roi in rois_only)]
X_df = feat_df_roi[keep_cols_final]
feat_cols = [c for c in X_df.columns if c.endswith('_corrected_z')]

X_df = X_df[feat_cols].copy()

groups = feat_df_roi['ptID'].values

X_df = X_df.fillna(X_df.mean())

X = X_df.values

alphas_grid = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 75, 100, 200, 500, 1000]
n_splits_options = [3, 5, 8, 10]
q_options = [2, 3, 4, 5]


y_aff = scores['aff_score'].values
y_sens = scores['sens_score'].values

sweep_results = []
for n_splits in n_splits_options:
    for q in q_options:
        res = run_stratified_cv(X, y_aff, y_sens, n_splits, q, alphas_grid)
        sweep_results.append(res)

sweep_df = pd.DataFrame(sweep_results)
display(sweep_df[['n_splits','q','r2_aff','r2_sens','alpha_aff_mode','alpha_sens_mode']])