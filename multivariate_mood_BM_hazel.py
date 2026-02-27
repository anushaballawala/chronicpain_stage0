# %%
## Overview 

# INPUT: data that has been preprocessed (i.e. wavelet extracted and averaged) to run partial correlation and regular spearman correlation 
# INPUT: .csv files saved in jsaal folder (now copied over to /userdata/aallawala/pain_data/stage0/redcap) 
# not structured yet to run as python script bc still working on adding other analysis. 

#For Hazel: the variables to work with for logistic regression are: 
# Issues that we typically run into that would be good to fix: 
# 
# 1. some trials are missing for surveys (i.e. nan values for pain scores but not for mood scores or the other way around
# -- we should keep all the data we can unless we're doing a 1-1 comparison like for pairwise t-stats)
# 2. some channels are missing across days, so we need a way to handle this. might be easier to talk about this over zoom 


# X = clean_psd_z.T trials x number_frequencies x number_channels (e.g. 200 x 40 x 150)
# Y = vasd_z_clean_r number_trials 
# Z = vasp_z_clean 
# (you'll find these later down in this script)



# %%
import sys
sys.path.append("/home/jiahuang/test-code-hazel/")
import gen_fxns
import h5py
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sp
from datetime import date
import pandas as pd 
from datetime import timedelta
from scipy.stats import zscore
import seaborn as sns 
import pickle 
import os 

# %%
ptID = "RCS06" # patient ID
path_string = f"/userdata/jiahuang/pain-data/Stage1-test/{ptID}/biomarker/preproc_data/all_channels/"
pt_path = f"/userdata/rvatsyayan/AnushaData/HDF5 Pain Data/{ptID}"
data_root = Path(path_string)
file_keyword = '_meanpsd'
dataset_name = "mean_psd" # files with mean power spectral density 

# Function to extract numeric part from filename
def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def load_h5_files(path_string, pt_path, file_keyword, dataset_name):
    import re
    h5_arrays = []
    fileids = [] 
    # Get list of files in directory and sort them based on numeric part
    files = sorted(os.listdir(path_string), key=extract_number)
    
    for filename in files:
        if filename.endswith('.h5') and file_keyword in filename:
            # Construct the full file path
            filepath = os.path.join(path_string, filename)
            print(filepath)
            # Load the .h5 file
            with h5py.File(filepath, 'r') as hf:
                # Assuming you want to load the first dataset from each file
                # Load the dataset as float32
                dataset = np.array(hf[dataset_name], dtype=np.float32)
                # Append the dataset to the list
                h5_arrays.append(dataset)

    files = sorted(os.listdir(pt_path), key=extract_number)
    for filename in files:
        if filename.endswith('.h5'):
            with h5py.File(os.path.join(pt_path, filename), 'r') as hf:
                # get record_id 
                fileid = int(pd.DataFrame(hf['pain_info']).iloc[0,0])
                fileids.append(fileid)
    # print(fileids)

    return h5_arrays, fileids



# try:
#     loaded_datasets = load_h5_files(directory_path, keyword, dataset_name)
#     # Now you can work with loaded_datasets
# except FileNotFoundError:
#     print(f"Directory '{directory_path}' not found.")

h5_arrays,fileids = load_h5_files(path_string, pt_path, file_keyword, dataset_name)



# %%

# concatenate data and zscore 
all_data  = []
all_data = np.stack(h5_arrays, axis=2)
print("freqs x channels x trials:", all_data.shape) 
del h5_arrays
from scipy.stats import zscore
psd_z = zscore(all_data,axis = 2) #freqs x channels x trials: e.g. (40, 124, 241)



# %%
file = f"/userdata/rvatsyayan/AnushaData/Pain_Scores_{ptID}.xlsx"
raw_surveys = pd.read_excel(file)
print(raw_surveys.shape) # e.g.  (225, 22)
n_trials = raw_surveys.shape[0]
raw_surveys

# %%
# clean up data 

missing_bm_data = np.setdiff1d(raw_surveys.record_id, fileids) # Return the unique values in ar1 that are not in # unique vals in survey ids that dont have neural data files.

missing_surveys = np.setdiff1d( fileids, raw_surveys.record_id) # unique vals in neuraldata file ids that dont exist in survey data. 

missing_channels = [i for i in range(all_data.shape[1]) if np.any(all_data[:,i,:])!=0]
print((missing_channels))
removed = [i for i in range(all_data.shape[1]) if np.any(all_data[:,i,:])==0]
print((removed))

# remove missing survey record ids from neural data. 
idx_missing_surveys = ~np.isin(fileids, missing_surveys)

new_alldata = psd_z[:,:,idx_missing_surveys]
new_alldata = new_alldata[:,missing_channels,:]
print(new_alldata.shape)

idx_missing_neuraldata = ~np.isin(raw_surveys.record_id, missing_bm_data)
new_surveys = raw_surveys.iloc[idx_missing_neuraldata]
print(new_surveys.shape)

assert(new_surveys.shape[0] == new_alldata.shape[2])

# Create a new list with only the elements where the boolean mask is True
filtered_list = [item for item, keep in zip(fileids, idx_missing_surveys) if keep]
print(filtered_list)

# %%
remaining_diff = np.setdiff1d( new_surveys.record_id, filtered_list)
if len(remaining_diff) == 0:
    print('continue')
elif len(remaining_diff) >0:
    print('Error')

print(new_surveys.shape)
print(len(filtered_list))

# %%
## reshape and zscore surveys. 
# psd_z_vec = new_alldata.reshape(new_alldata.shape[2], -1)
n_freq = new_alldata.shape[0]
n_ch = new_alldata.shape[1]
n_feats = n_freq * n_ch  # num of neural features
n_trials = new_alldata.shape[2]
psd_z_vec = np.reshape(new_alldata, (n_feats, n_trials))

vasd = new_surveys['mood_vas_s0'].to_numpy() # depression/mood survey scores 
vasp = new_surveys['intensity_vas_s0'].to_numpy() #pain survey scores
mpq_affective = new_surveys[['tiring_exhausting_s0', 'sickening_s0', 'fearful_s0','punishing_cruel_s0']]
sum_affective = np.sum(mpq_affective,axis=1).to_numpy()

vasd_vec = vasd.reshape(-1)
vasp_vec = vasp.reshape(-1)
sum_affective_vec = sum_affective.reshape(-1)

def man_z_score(array):
    array_mean = np.nanmean(array)
    array_std = np.nanstd(array)
    zscore_array = (array - array_mean)/array_std
    return zscore_array

vasd_z = man_z_score(vasd_vec)
vasp_z = man_z_score(vasp_vec)
affective_z = man_z_score(sum_affective_vec)

# %%
## remove nans from filtered data.  

def find_nans(array):
    # if nan_idx.any():
    nan_idx = np.argwhere(np.isnan(array))
    if nan_idx.any():
        print("The array contains NaN values.")
    else:
        print("The array does not contain any NaN values.") 
    return nan_idx

nan_idx_vasd = find_nans(vasd_z)
nan_idx_vasp = find_nans(vasp_z)
nan_idx_aff = find_nans(affective_z)
nan_idx_all  = np.unique(np.concatenate([nan_idx_vasd, nan_idx_vasp, nan_idx_aff]))
print(nan_idx_all)
# clean up surveys from nans. 
vasd_z_clean = np.delete(vasd_z, [nan_idx_all], axis = 0)
vasp_z_clean = np.delete(vasp_z, [nan_idx_all], axis = 0)
affective_z = np.delete(affective_z, [nan_idx_all], axis = 0)

# remove nans. 
new_surveys = new_surveys.drop(new_surveys.index[nan_idx_all], axis=0) # remove by positional index clean_psd_z= np.delete(psd_z_vec, nan_idx_all, axis=1)
clean_psd_z= np.delete(psd_z_vec, nan_idx_all, axis=1) #**** ?????
print(clean_psd_z.shape)



# %%
new_surveys.shape

# %%
# flip depression scores (so that worse mood = higher mood score, to match worse pain = higher pain score)

# Reverse the scores
max_score = max(vasd_z_clean)
min_score = min(vasd_z_clean)

# Reverse the depression scores such that higher original scores correspond to lower new scores
vasd_z_clean_r = max_score + min_score - vasd_z_clean


# %%
# Dimension reduction of all_data
clean_alldata = all_data[:,:,idx_missing_surveys]
clean_alldata= np.delete(clean_alldata, nan_idx_all, axis=2)
clean_alldata = clean_alldata[:,missing_channels,:]
print(clean_alldata.shape)
n_trials = clean_alldata.shape[2]

band_data = np.zeros((6, n_ch, n_trials))
bandref = {'delta':(1,4), 'theta':(5,8), 'alpha':(9,12), 'beta':(13,30), 'low gamma':(31,70), 'high gamma':(71,150)}
bands = list(bandref.keys())

import re
ch_labels = []
freqs = []
with h5py.File(f"{path_string}/1_meanpsd.h5", 'r') as hf:
    for i in missing_channels:
        ch = re.findall(r"\d+([A-Za-z]+\d+)",str(hf.attrs['ch_labels'][i]))[0]
        ch_labels.append(ch)
    freqs = hf.attrs['freqs']
canonical_freq = [bandref[bands[i]][0] for i in range(len(bands))]


for b in range(len(bands)):
    band = bands[b]
    (fmin, fmax) = bandref[band]
    idxf = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    band_data[b, :, :] = np.mean(clean_alldata[idxf, :, :], axis=0)

large_ROI = np.unique([re.findall(r"[A-Za-z]+", s)[0] for s in ch_labels])
ch_data = np.zeros((len(bands), len(large_ROI), n_trials))

for i in range(len(large_ROI)):
	ch = large_ROI[i]
	idxc = [ch in ch_name for ch_name in ch_labels]
	ch_data[:,i,:] = np.mean(band_data[:, idxc, :], axis = 1)

ch_data_new = np.zeros((len(freqs), len(large_ROI), n_trials))

for i in range(len(large_ROI)):
	ch = large_ROI[i]
	idxc = [ch in ch_name for ch_name in ch_labels]
	ch_data_new[:,i,:] = np.mean(clean_alldata[:, idxc, :], axis = 1)
	
all_data_clean_freq_roi = zscore(ch_data, axis = 2)
all_data_clean_freq_roi_z = all_data_clean_freq_roi.reshape(len(bands)*len(large_ROI), n_trials).T
all_data_clean_freq_z = zscore(band_data, axis=2).reshape(len(bands)*len(ch_labels), n_trials).T
all_data_clean_roi_z = zscore(ch_data_new, axis=1).reshape(len(freqs)*len(large_ROI), n_trials).T

print(ch_labels)
print(large_ROI)
print(freqs)
print(canonical_freq)

# %% [markdown]
# 

# %%
sys.path.append("/home/jiahuang/test-code-hazel/gen_fxns/")
savepath = f"/userdata/jiahuang/pain-data/figures/Stage1-test/{ptID}/"
import os
os.makedirs(savepath, exist_ok=True)

def basic_heatmap(ax, array, ch_labels, freqs, cbar_title, title):
    fig_params = [8, 12]
    caxis_lim = [-0.1, 0.2]
    
    sns.heatmap((array.T), cmap="RdBu_r", 
                vmax = np.nanmax(np.abs(array)), vmin = -np.nanmax(np.abs(array)), center = 0,cbar=True,
                yticklabels=ch_labels, xticklabels=np.round(freqs), 
                cbar_kws={'label': cbar_title}, ax=ax)
    ax.set_xlabel("Frequency(Hz)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)


def plot_heatmap_subplots(data_list, ch_labels, freqs, cbar_titles, titles, savetitle, nrows, ncols):
    """
    Generate subplots using the custom heatmap function.

    Parameters:
    data_list (list of ndarray): List of 2D arrays to be plotted as heatmaps.
    cbar_titles (list of str): List of colorbar titles for each heatmap.
    titles (list of str): List of titles for each heatmap.
    nrows (int): Number of rows in the subplot grid.
    ncols (int): Number of columns in the subplot grid.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 10 * nrows))
    axes = axes.flatten() 

    for i, (array, cbar_title, title) in enumerate(zip(data_list, cbar_titles, titles)):
        basic_heatmap(axes[i], array, ch_labels, freqs, cbar_title, title)

    plt.tight_layout()
    save_dir = f"/userdata/jiahuang/pain-data/figures/Stage1-test/{ptID}/{savetitle}"

    plt.savefig(save_dir, dpi=300, edgecolor='k', facecolor="white")
    plt.show()

# %%
# OLS Tstats
import statsmodels.api as sm
from patsy import dmatrices

df = pd.DataFrame({'VASP': vasp_z_clean.flatten(), 'VASD': vasd_z_clean_r.flatten()})
clean_psd_z_news = [clean_psd_z.T, all_data_clean_freq_z, all_data_clean_freq_roi_z]
title = [ "All chanel all frequency OLS T-stats", "All chanel canonical frequency OLS T-stats","Large roi canonical frequency OLS T-stats"]
ch = [ch_labels, ch_labels, large_ROI]
freq = [freqs, canonical_freq, canonical_freq]

for j in range(3):
    clean_psd_z_new = clean_psd_z_news[j]
    for i in range(clean_psd_z_new.shape[1]):
        df[f'PSD{i}'] = clean_psd_z_new[:, i]

    design_matrices = {}

    for i in range(clean_psd_z_new.shape[1]):
        formula = f'PSD{i} ~ VASD + VASP'
        y, X = dmatrices(formula, data=df, return_type='dataframe')
        design_matrices[f'PSD{i}'] = {'y': y, 'X': X}

    results = {}

    for key in design_matrices.keys(): #keys are psd0, psd1, etc. 
        y = design_matrices[key]['y']
        X = design_matrices[key]['X']
        
        mod = sm.OLS(y, X)  # Create model
        res = mod.fit()     # Fit model
        
        results[key] = {
            'summary': res.summary(),  # Summary of the regression
            'tvalues': res.tvalues,    # T-values of the coefficients
            'pvalues': res.pvalues     # P-values of the coefficients
        }

    #put t-values in an array. 
    #put t-values in an array. 
    t_vals_d= []
    t_vals_p = [] 
    p_vals_p =[] 
    p_vals_d = []
    for i in range(len(results)):
        tmp_tval = results[f'PSD{i}']['tvalues']['VASD']
        t_vals_d.append(tmp_tval)

        tmp_pval = results[f'PSD{i}']['pvalues']['VASD']
        p_vals_d.append(tmp_tval)

        tmp_tval = results[f'PSD{i}']['tvalues']['VASP']
        t_vals_p.append(tmp_tval)

        tmp_pval = results[f'PSD{i}']['pvalues']['VASP']
        p_vals_p.append(tmp_tval)

    tval_stack_d = np.vstack(t_vals_d)
    tval_reshape_d = np.reshape(tval_stack_d,[len(freq[j]), len(ch[j])])

    pval_stack_d = np.vstack(p_vals_d)
    pval_reshape_d = np.reshape(pval_stack_d,[len(freq[j]), len(ch[j])])

    tval_stack_p = np.vstack(t_vals_p)
    tval_reshape_p = np.reshape(tval_stack_p,[len(freq[j]), len(ch[j])])

    pval_stack_p = np.vstack(p_vals_p)
    pval_reshape_p = np.reshape(pval_stack_p,[len(freq[j]), len(ch[j])])

    mask_pvalues = pval_reshape_p > 0.05
    # Apply mask: Set values where mask is True to NaN
    masked_t_vals_p = np.where(mask_pvalues, np.nan, tval_reshape_p)

    mask_pvalues = pval_reshape_d > 0.05
    # Apply mask: Set values where mask is True to NaN
    masked_t_vals_d = np.where(mask_pvalues, np.nan, tval_reshape_d)

    plot_heatmap_subplots([masked_t_vals_p, masked_t_vals_d], ch[j], freq[j], ['Tstats','Tstats'],['OLS PSD vs Pain %s' % (ptID), 'OLS PSD vs Mood %s' % (ptID)], title[j], nrows=1, ncols=2)

# %%
# Correlation analysis

def run_corr(x,y, n_freq, n_ch):
    from scipy.stats import spearmanr
    corr,p_value = spearmanr(x,y)
    correlations = corr[:-1,-1]
    p = p_value[:-1,-1].reshape(n_freq,n_ch)
    new_corr = correlations.reshape(n_freq,n_ch)
    return new_corr, p

corr_vasd, p = run_corr(clean_psd_z.T, vasd_z_clean_r, n_freq, n_ch)
corr_vasd_mask = np.where(p>0.05, np.nan, corr_vasd)
corr_vasd2, p = run_corr(all_data_clean_freq_roi_z, vasd_z_clean_r, len(bands), len(large_ROI))
corr_vasd2_mask = np.where(p>0.05, np.nan, corr_vasd2)
corr_vasd3, p = run_corr(all_data_clean_freq_z, vasd_z_clean_r, len(bands), len(ch_labels))
corr_vasd3_mask = np.where(p>0.05, np.nan, corr_vasd3)
corr_vasd4, p = run_corr(all_data_clean_roi_z, vasd_z_clean_r, len(freqs), len(large_ROI))
corr_vasd4_mask = np.where(p>0.05, np.nan, corr_vasd4)

corr_vasp, p2 = run_corr(clean_psd_z.T, vasp_z_clean, n_freq, n_ch)
corr_vasp_mask = np.where(p2>0.05, np.nan, corr_vasp)
corr_vasp2, p2 = run_corr(all_data_clean_freq_roi_z, vasp_z_clean, len(bands), len(large_ROI))
corr_vasp2_mask = np.where(p2>0.05, np.nan, corr_vasp2)
corr_vasp3, p2 = run_corr(all_data_clean_freq_z, vasp_z_clean, len(bands), len(ch_labels))
corr_vasp3_mask = np.where(p2>0.05, np.nan, corr_vasp3)
corr_vasp4, p2 = run_corr(all_data_clean_roi_z, vasp_z_clean, len(freqs), len(large_ROI))
corr_vasp4_mask = np.where(p2>0.05, np.nan, corr_vasp4)

corr_aff, p3 = run_corr(clean_psd_z.T, affective_z, n_freq, n_ch)
corr_aff_mask = np.where(p3>0.05, np.nan, corr_aff)
corr_aff2, p3 = run_corr(all_data_clean_freq_roi_z, affective_z, len(bands), len(large_ROI))
corr_aff2_mask = np.where(p3>0.05, np.nan, corr_aff2)
corr_aff3, p3 = run_corr(all_data_clean_freq_z, affective_z, len(bands), len(ch_labels))
corr_aff3_mask = np.where(p3>0.05, np.nan, corr_aff3)
corr_aff4, p3 = run_corr(all_data_clean_roi_z, affective_z, n_freq, len(large_ROI))
corr_aff4_mask = np.where(p3>0.05, np.nan, corr_aff4)


# Plot the heatmaps in a 2x2 grid
plot_heatmap_subplots([corr_vasp, corr_vasd, corr_aff], ch_labels, freqs, ['Corr', 'Corr','Corr'], ['PSD vs Pain %s' % (ptID), 'PSD vs Mood %s' % (ptID),'PSD vs Affective %s' % (ptID)], "All chanel roi correlation - with affective comp", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp2, corr_vasd2, corr_aff2], large_ROI, canonical_freq, ['Corr', 'Corr','Corr'], ['Band+ROI Concatenated PSD vs Pain %s' % (ptID), 'Band+ROI Concatenated PSD vs Mood %s' % (ptID),'Band+ROI Concatenated PSD vs Affective %s' % (ptID)], "Canonical chanel roi correlation - with affective comp", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp3, corr_vasd3, corr_aff3], ch_labels, canonical_freq, ['Corr', 'Corr','Corr'], ['Only Band Concatenated PSD vs Pain %s' % (ptID), 'Only Band Concatenated PSD vs Mood %s' % (ptID),'Only Band Concatenated PSD vs Affective %s' % (ptID)], "Canonical chanel correlation - with affective comp ", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp4, corr_vasd4, corr_aff4], large_ROI, freqs, ['Corr', 'Corr','Corr'], ['Only ROI Concatenated PSD vs Pain %s' % (ptID), 'Only ROI Concatenated PSD vs Mood %s' % (ptID), 'Only ROI Concatenated PSD vs Affective %s' % (ptID)], "Brain roi correlation - with affective comp ", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp_mask, corr_vasd_mask, corr_aff_mask], ch_labels, freqs, ['Corr', 'Corr','Corr'], ['PSD vs Pain %s' % (ptID), 'PSD vs Mood %s' % (ptID),'PSD vs Affective %s' % (ptID)], "Masked All chanel roi correlation - with affective comp", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp2_mask, corr_vasd2_mask, corr_aff2_mask], large_ROI, canonical_freq, ['Corr', 'Corr','Corr'], ['Band+ROI Concatenated PSD vs Pain %s' % (ptID), 'Band+ROI Concatenated PSD vs Mood %s' % (ptID), 'Band+ROI Concatenated PSD vs Affective %s' % (ptID)], "Masked Canonical chanel roi correlation - with affective comp ", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp3_mask, corr_vasd3_mask, corr_aff3_mask], ch_labels, canonical_freq, ['Corr', 'Corr','Corr'], ['Only Band Concatenated PSD vs Pain %s' % (ptID), 'Only Band Concatenated PSD vs Mood %s' % (ptID),'Only Band Concatenated PSD vs Affective %s' % (ptID)], "Masked Canonical chanel correlation - with affective comp ", nrows=1, ncols=3)

plot_heatmap_subplots([corr_vasp4_mask, corr_vasd4_mask, corr_aff4_mask], large_ROI, freqs, ['Corr', 'Corr','Corr'], ['Only ROI Concatenated PSD vs Pain %s' % (ptID), 'Only ROI Concatenated PSD vs Mood %s' % (ptID),'Only ROI Concatenated PSD vs Affective %s' % (ptID)], "Masked Brain roi correlation - with affective comp ", nrows=1, ncols=3)


# %%
# Partial Correlation controlled for pain

from scipy import stats, linalg

# Step 1: Fit regression models and obtain residuals
# Regression of Y on Z
n_feats = [len(ch_labels)*len(freqs), len(large_ROI)*len(bands), len(ch_labels)*len(bands)]
X1 = clean_psd_z.T
X2 = all_data_clean_freq_roi_z
X3 = all_data_clean_freq_z
Xs = [X1, X2, X3]
y_sticks = [ch_labels, large_ROI, ch_labels]
x_sticks = [freqs, canonical_freq, canonical_freq]
Yd = vasd_z_clean_r
Yp = vasp_z_clean

slope_YY, intercept_YY, _, _, _ = stats.linregress(Yd, Yp)
Yp_pred = slope_YY * Yd + intercept_YY
Y_resid = Yp - Yp_pred
for i in range(len(Xs)):
    X = Xs[i]
    # Regression of each feature in X on Z
    X_resid = np.zeros_like(X)
    for j in range(n_feats[i]):
        slope_XZ, intercept_XZ, _, _, _ = stats.linregress(Yd, X[:, j])
        X_pred = slope_XZ * Yd + intercept_XZ
        X_resid[:, j] = X[:, j] - X_pred

    # Step 2: Compute partial correlation coefficients
    partial_corr_XY_Z_reg = np.zeros(n_feats[i])
    partial_corr_XY_Z_reg_p = np.zeros(n_feats[i])
    for j in range(n_feats[i]):
        partial_corr_XY_Z_reg[j], partial_corr_XY_Z_reg_p[j] = stats.pearsonr(X_resid[:, j], Y_resid)
    mask = partial_corr_XY_Z_reg_p>0.05
    partial_corr_XY_Z_reg_mask = np.where(mask, np.nan, partial_corr_XY_Z_reg)
    # print("Partial correlation coefficients between X and Y controlling for Z using regression: ")
    pcorr_reshaped = partial_corr_XY_Z_reg.reshape(len(x_sticks[i]), len(y_sticks[i]))
    pcorr_reshaped_mask = partial_corr_XY_Z_reg_mask.reshape(len(x_sticks[i]), len(y_sticks[i]))

    print("Partial correlation coefficients between X and Y controlling for Z using regression:")
    # pcorr = pd.DataFrame(pcorr_reshaped, index=freqs, columns=ch_labels)

    fig, ax = plt.subplots(figsize=[8,10])
    ax = sns.heatmap((pcorr_reshaped.T), cmap="RdBu_r", cbar=True, 
        yticklabels=y_sticks[i], xticklabels=np.round(x_sticks[i]), cbar_kws={'label': 'Partial Correlation'})
    yticklabels='auto'
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Channel")
    plt.title('Partial Correlation of PSD vs mood (controlling for pain) - %s' % (ptID))
    # xticklabels = 'auto'
    ytick_labels = y_sticks[i]
    if ptID=="RCS05":
        y_ticks = [0,8,21,32, 40, 50, 58, 71, 84, 91]
    elif ptID=="RCS04":
        y_ticks = [0,8,15,26,31,41,47,54,65, 66]
    elif ptID == "RCS02":
        y_ticks = [0,9,24, 40, 55,70,79,91,99]
    elif ptID == "RCS07":
        print('get it')

    y_ticks = np.arange(0,len(y_sticks[i]),1)
    # ytick_labels = ytick_labels[y_ticks]
    if i == 0: 
        x_ticks = [0, 11, 15, 18, 23, 26, 29, 32, 35, 37, 39]
        xtick_labels = freqs[x_ticks]
    else: 
        x_ticks = range(6)
        xtick_labels = canonical_freq  # Use the correct subset of ylabels
    ax.set(xticks = x_ticks, xticklabels = np.round(xtick_labels), yticks = y_ticks, yticklabels = ytick_labels)
    plt.tight_layout()
    # plt.savefig(save_dir, dpi=300, edgecolor='k', facecolor="white")

    plt.show()

    fig, ax = plt.subplots(figsize=[8,10])
    ax = sns.heatmap((pcorr_reshaped_mask.T), cmap="RdBu_r", cbar=True, 
        yticklabels=y_sticks[i], xticklabels=np.round(x_sticks[i]), cbar_kws={'label': 'Partial Correlation'})
    yticklabels='auto'
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Channel")
    plt.title('Partial Correlation of PSD vs mood (controlling for pain) - %s' % (ptID))
    # xticklabels = 'auto'
    ytick_labels = y_sticks[i]
    if ptID=="RCS05":
        y_ticks = [0,8,21,32, 40, 50, 58, 71, 84, 91]
    elif ptID=="RCS04":
        y_ticks = [0,8,15,26,31,41,47,54,65, 66]
    elif ptID == "RCS02":
        y_ticks = [0,9,24, 40, 55,70,79,91,99]
    elif ptID == "RCS07":
        print('get it')

    y_ticks = np.arange(0,len(y_sticks[i]),1)
    # ytick_labels = ytick_labels[y_ticks]
    if i == 0: 
        x_ticks = [0, 11, 15, 18, 23, 26, 29, 32, 35, 37, 39]
        xtick_labels = freqs[x_ticks]
    else: 
        x_ticks = range(6)
        xtick_labels = canonical_freq  # Use the correct subset of ylabels
    ax.set(xticks = x_ticks, xticklabels = np.round(xtick_labels), yticks = y_ticks, yticklabels = ytick_labels)
    plt.tight_layout()
    # plt.savefig(save_dir, dpi=300, edgecolor='k', facecolor="white")

    plt.show()
# pcorr

# %%
# Regularized partial correlation network (tabled)

from sklearn.covariance import GraphicalLassoCV

partial_corr_re_X = np.mean(all_data_clean_freq_roi,axis=0).T

model = GraphicalLassoCV(alphas=10,cv=5,max_iter=2000)

model.fit(partial_corr_re_X)
model.cv_results_

fig, ax = plt.subplots(figsize=(7,6))
map = ax.imshow(model.covariance_, aspect='auto')
plt.title(f"{ptID} regularized partial correlation network")
plt.xlabel("Features")

ax.set_xticks(np.arange(len(large_ROI)))
ax.set_yticks(np.arange(len(large_ROI)))

ax.set_xticklabels(large_ROI, fontsize=6)
ax.set_yticklabels(large_ROI, fontsize=6)
cb = fig.colorbar(map,ax=ax, shrink=1)
cb.ax.tick_params(labelsize=7, size=1, pad=1)
cb.ax.set_ylabel('covariance', size=10)
plt.show()


# %%
# Also regularized partial correlation network code from https://pmc.ncbi.nlm.nih.gov/articles/PMC12461088/#IMAG.a.162-S1
# This will have package dependencies issue
import gglasso.problem
from gglasso.problem import glasso_problem
from sklearn.covariance import log_likelihood,empirical_covariance
partial_corr_re_X = all_data_clean_freq_roi_z.T

def graphicalLassoCV(data,L1s=None,kFolds=10,optMethod='loglikelihood',foldsScheme='blocked'):
    if L1s is None:
        # Test log-scaled range of L1s (from 0.316 to 0.001)
        L1s = np.arange(-.5,-3.1,-.1) 
        L1s = 10**L1s

    nTRs = data.shape[1]
    kFoldsTRs = np.full((kFolds,nTRs),False)

    if foldsScheme=='blocked':
        TRsPerFold = nTRs/kFolds
        t1 = 0
        for k in range(kFolds):
            t2 = int(np.round((k+1)*TRsPerFold))
            kFoldsTRs[k,t1:t2] = True
            t1 = t2
    elif foldsScheme=='interleaving':
        k = 0
        for t in range(nTRs):
            kFoldsTRs[k,t] = True
            k += 1
            if k >= kFolds:
                k = 0
    
    scores = np.zeros((len(L1s),kFolds))
    for l,L1 in enumerate(L1s):
        # Loop through folds
        for k in range(kFolds):
            # Estimate the regularized partial correlation and precision (intermediate) matrices
            parCorr,prec = graphicalLasso(data[:,~kFoldsTRs[k]],L1)

            if optMethod == 'loglikelihood':
                # Calculate negative loglikelihood
                empCov_test = np.cov(stats.zscore(data[:,kFoldsTRs[k]],axis=1),rowvar=True)
                scores[l,k] = -log_likelihood(empCov_test,prec)

            # elif optMethod == 'R2':
            #     # Calculate R^2
            #     scores[l,k],r = activityPrediction(stats.zscore(data[:,kFoldsTRs[k]],axis=1),parCorr)

    # Find the best param according to each performance metric
    meanScores = np.mean(scores,axis=1)
    if optMethod == 'loglikelihood':
        bestParam = L1s[meanScores==np.amin(meanScores)]
    elif optMethod == 'R2':
        bestParam = L1s[meanScores==np.amax(meanScores)]

    # Estimate the regularized partial correlation using all data and the optimal hyperparameters
    parCorr,prec = graphicalLasso(data,bestParam)
    return parCorr, prec

def graphicalLasso(data,L1):
    '''
    Calculates the L1-regularized partial correlation matrix of a dataset. Runs GGLasso's graphical lasso function (glasso_problem.solve()) and several other necessary steps.
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1 : L1 (lambda1) hyperparameter value
    OUTPUT:
        glassoParCorr : regularized partial correlation coefficients (i.e., FC matrix)
        prec : precision matrix, where entries are not yet transformed into partial correlations (used to compute loglikelihood)
    '''

    nNodes = data.shape[0]

    # Z-score the data
    data_scaled = stats.zscore(data,axis=1)

    # Estimate the empirical covariance
    empCov = np.cov(data_scaled,rowvar=True)

    # Number of timepoints in data
    nTRs = data.shape[1]

    # Run glasso
    glasso = glasso_problem(empCov,nTRs,reg_params={'lambda1':L1},latent=False,do_scaling=False)
    # #Output to null device to suppress verbose output
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    #     result = glasso.solve(verbose=False)
    prec = np.squeeze(glasso.solution.precision_)

    # Transform precision matrix into regularized partial correlation matrix
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
    glassoParCorr = -prec * denom * denom.T
    np.fill_diagonal(glassoParCorr,0)

    return glassoParCorr,prec

parCorr, prec = graphicalLassoCV(partial_corr_re_X)
parCorr

# %%
# Representation Similarity Analysis

# Input:
#   X = clean_psd_z (trials x number_frequencies x number_channels, 211*40*124)
#   Yd = vasd_z_clean_r (number_trials)
#   Yp = vasp_z_clean 

# Calculate RDM:
#    RDMeeg[i, j] = 1 - pearsoncorr(X[i,:,:], X[j,:,:])
#    RDMpain[i, j] = Euclidean(Yd[i],Yd[j])

# Upper triangle:
#    eeg_vec = upper(RDMeeg)
#    pain_vec = upper(RDMpain)

# Compute RSA:
#    rsa = SpearmanCorr(eeg_vec, pain_vec)

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

def upper_triangular(df):
    idx = np.triu_indices_from(df, k=1)
    return df[idx]

RDMpain = squareform(pdist(vasp_z_clean.reshape(-1,1), metric = 'euclidean'))
RDMdep = squareform(pdist(vasd_z_clean_r.reshape(-1,1), metric = 'euclidean'))

# raw score of vas is 0-100, zscore is -1 to 1
RDMf = squareform(pdist(all_data_clean_freq_roi_z, metric='correlation'))

# dissimilarity 1-correlation

pain_vec = upper_triangular(RDMpain)
dep_vec = upper_triangular(RDMdep)
f_vec = upper_triangular(RDMf)

corr_freqvd, p_value_freqvd = spearmanr(f_vec, dep_vec)
corr_freqvp, p_value_freqvp = spearmanr(f_vec, pain_vec)
corr_dvp, p_value_dvp = spearmanr(pain_vec, dep_vec)

# Visualize RDM
fig, ax = plt.subplots(figsize=[6, 2], nrows=1, ncols=3, dpi=300)
im = [0,0,0]
im[0] = ax[0].imshow(RDMf, cmap='viridis')
ax[0].set_title('RDM EEG', size=7, pad=3)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('trials', size=7, labelpad=1)
ax[0].set_ylabel('trials', size=7, labelpad=1)

im[1] = ax[1].imshow(RDMpain, cmap='viridis')
ax[1].set_title('RDM Pain', size=7, pad=3)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlabel('trials', size=7, labelpad=1)
ax[1].set_ylabel('trials', size=7, labelpad=1)

im[2] = ax[2].imshow(RDMdep, cmap='viridis')
ax[2].set_title('RDM Depression', size=7, pad=3)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_xlabel('trials', size=7, labelpad=1)
ax[2].set_ylabel('trials', size=7, labelpad=1)

cb = fig.colorbar(im[0], ax=ax, shrink=0.8)
cb.ax.tick_params(labelsize=5, size=1, pad=1)
cb.ax.set_ylabel('Correlation distance', size=7)
plt.savefig(f"{savepath}RDM.png")
plt.show()

print(corr_dvp, corr_freqvd, corr_freqvp)
print(p_value_dvp, p_value_freqvd, p_value_freqvp)

# %%
# Canonical Correlation Analysis
# Step 0: Input
# X is concatenated neural data, Y is behavioral data (could be high dimensional, multiple mood metric or multiple pain metric)
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

X = all_data_clean_freq_roi_z # n_trial * (len(bands)*len(large_ch)
Y = np.stack([vasd_z_clean_r, vasp_z_clean],axis=1)
# Or X = band_data_z.reshape((len(bands)*n_ch, n_trial).T -- channels not concatenated

# Step 1: Standardizing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)

# Step 2: fitting CCA
cca = CCA(n_components=2) # could have more components
cca.fit(X_scaled, Y_scaled)
X_c, Y_c = cca.transform(X_scaled, Y_scaled)
corrs = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=X_c.shape[1])

# Step 3: storing all weights
for i in range(2):
	print(cca.x_weights_[:, i])
	print(cca.y_weights_[:, i])


# %%
fig, ax = plt.subplots(figsize=(8,6))
plt.imshow(cca.x_weights_[:,0].reshape(len(large_ROI), len(bands)), aspect='auto')
plt.colorbar(label='Weight')
x_ticks, y_ticks = np.arange(len(bands)), np.arange(len(large_ROI))
ax.set(xticks = x_ticks, xticklabels = bands, yticks = y_ticks, yticklabels = large_ROI)
plt.title(f"{ptID} CCA1: y_depression = {np.round(cca.y_weights_[0, 0],3)}, y_pain = {np.round(cca.y_weights_[1, 0],3)}")
ax.set_xlabel("Canonical frequency")
ax.set_ylabel("ROI")
plt.show()

# %%
# Regularized CCA analysis

import rcca
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = all_data_clean_freq_roi_z # n_trial * (len(bands)*len(large_ch)
Y = np.stack([vasd_z_clean_r, vasp_z_clean],axis=1)
iterations = 200
all_numCC = []
all_numreg = []


scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

for i in range(iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

    # from sklearn.decomposition import PCA

    # pca = PCA(n_components=0.999)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca  = pca.transform(X_test)

    regs = np.logspace(-3,3,10)
    ccaCV = rcca.CCACrossValidate(kernelcca=False, numCCs = [1,2],regs = regs)

    # Use the train() and validate() methods to run the analysis and perform cross-dataset prediction.
    ccaCV.train([X_train, Y_train])
    ccaCV.validate([X_test, Y_test])
    all_numCC.append(ccaCV.best_numCC)
    all_numreg.append(ccaCV.best_reg)
    
occur = [(all_numreg == i).sum() for i in regs]
best_reg = [regs[i] for i in range(len(regs)) if occur[i]==np.max(occur)][0]
print(occur, best_reg)
updated_cca = rcca.CCA(kernelcca=False, numCC = 1, reg = best_reg)
updated_cca.train([X,Y])
updated_cca.cancorrs, updated_cca.ws[1]

# %%

fig, ax = plt.subplots(figsize=(8,6))
plt.imshow(updated_cca.ws[0][:,0].reshape(len(large_ROI), len(bands)), aspect='auto')
plt.colorbar(label='Weight')
x_ticks, y_ticks = np.arange(len(bands)), np.arange(len(large_ROI))
ax.set(xticks = x_ticks, xticklabels = bands, yticks = y_ticks, yticklabels = large_ROI)
plt.title(f"{ptID} regularized CCA1: y_depression = {np.round(updated_cca.ws[1][0,0],3)}, y_pain = {np.round(updated_cca.ws[1][1, 0],3)}, corr = {np.round(updated_cca.cancorrs[0],3)}")
ax.set_xlabel("Canonical frequency")
ax.set_ylabel("ROI")
plt.show()

# %%
# Regularized cca from cca-zoo - unifinished, they also have sparse cca model
from cca_zoo.linear import rCCA
from sklearn.model_selection import train_test_split
X = all_data_clean_freq_roi_z # n_trial * (len(bands)*len(large_ch)
Y = np.stack([vasd_z_clean_r, vasp_z_clean],axis=1)
scaler = StandardScaler()
Y = scaler.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
cs = np.logspace(-3,3,10)
for c in cs:
    model = rCCA(c = c)

    model.fit([X_train, Y_train])
    model.canonical_loadings_

# %%
# saving metadata
import json
save_metadata = f'/userdata/jiahuang/pain-data/Stage1-test/{ptID}'
removed = [i for i in range(all_data.shape[1]) if np.any(all_data[:,i,:])==0]

# date, person did the analysis, short text description, code version, .py, ptID, ch_label, ch_removed

metadata = {
    "Date": date.today(),
    "Analyzer": 'Hazel Huang',
    "Code version": ,
    "ptID":ptID,
    "Channel labels": ch_labels,
    "Removed channels":removed,
    "description": "missing survey/ieeg channel removed, normalized, 3 types of concatenation, run correlation/partial correlation/cca/rsa/ols tstats."

}

with open(f"{save_metadata}/metadata.json", "w") as f:
    json.dump(metadata, f)

# %%
from fooof import FOOOF, FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fm, get_band_peak_fg

# Initialize FOOOF group object for each trial
fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.1, max_n_peaks=6)

# Define frequency range across which to model the spectrum
freq_range = [3, 40] # Min frequency suggested to be twice the lowest recorded frequency to filter out noise
new_freq = np.arange(1,130,130/40) # minf = 1, maxf = 130; all_data_nolog is recalculated based on new_freq and without log

fg.fit(new_freq, np.array(np.transpose(all_data_nolog[:,:,0])), freq_range) # new_freq and spectra data are required to be linearly spaced, spectra requires psd (no log)

# fg.save('FG_results', save_settings=True, save_results=True)

# Extract aperiodic parameters
aps = fg.get_params('aperiodic_params')

# Extract peak parameters
peaks = fg.get_params('peak_params')

# Extract goodness-of-fit metrics
errors = fg.get_params('error') # Need an error threshold?
r2s = fg.get_params('r_squared')

# Save raw data of periodic and aperiodic features
pdf = pd.DataFrame(peaks, columns=['CF','PW','BW','Channel']) # Num peaks of all group/channel * 4
pdf['Channel'] = pdf['Channel'].astype('int')
apdf = pd.DataFrame(aps, columns=['Offset','Exp']) # Num group/channel * 2
pdf.to_csv("dir")
apdf.to_csv("dir")


