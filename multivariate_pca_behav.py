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
import sklearn
from datetime import date
import pandas as pd 
from datetime import timedelta
from scipy.stats import zscore
import seaborn as sns 
import pickle 
import os 
from datetime import datetime

# %%
behav_score_allpt = {}
all_data_clean_roi_z_allpt = {}
all_data_clean_freq_roi_z_allpt = {}
ch_allpt = {}
ptIDs = ['RCS02','RCS03','RCS04','RCS05','RCS06','RCS08','RCS09']

# %%
for ptID in ptIDs:
    path_string = f"/userdata/jiahuang/pain-data/Stage1-test/{ptID}/biomarker/preproc_data/all_channels/"
    fig_save = f"/userdata/jiahuang/pain-data/Stage1-test/{ptID}/"
    pt_path = f"/userdata/rvatsyayan/AnushaData/HDF5 Pain Data/{ptID}"
    # pt_path = f"/home/rvatsyayan/AnushaData"
    electrode_df = pd.read_csv(f"/home/jiahuang/test-code-hazel/{ptID}_new_electrode_property_df.csv")
    data_root = Path(path_string)
    file_keyword = '_meanpsd'
    dataset_name = "mean_psd" # files with mean power spectral density 

    # hamd_path_string = Path(f"/userdata/rvatsyayan/AnushaData/HDF5 HAMD Data/{ptID}") 
    # hamd_data_root = Path(hamd_path_string)

    # Function to extract numeric part from filename
    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)))

    def load_h5_files(path_string, pt_path, 
                    #   hamd_path, 
                    file_keyword, dataset_name):
        import re
        h5_arrays = []
        hamd_arrays = {}
        fileids = [] 
        # Get list of files in directory and sort them based on numeric part
        files = sorted(os.listdir(path_string), key=extract_number)
        # hamd_files = sorted(os.listdir(hamd_path), key=extract_number)
        
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

        # for filename in hamd_files:
        #     if filename.endswith('.h5'):
        #         # Construct the full file path
        #         filepath = os.path.join(hamd_path_string, filename)
        #         # Load the .h5 file
        #         with h5py.File(filepath, 'r') as hf:
        #             # Assuming you want to load the first dataset from each file
        #             # Load the dataset as float32
        #             sum_hamd = int(hf['hamd_info'][8][0])
        #             # Append the dataset to the list
        #             hamd_arrays[int(hf['hamd_info'][0][0])]=sum_hamd

        files = sorted(os.listdir(pt_path), key=extract_number)
        for filename in files:
            if filename.endswith('.h5'):
                with h5py.File(os.path.join(pt_path, filename), 'r') as hf:
                    # get record_id 
                    fileid_pre = (pd.DataFrame(hf['pain_info']).iloc[1,0]).decode('utf-8')
                    fileid = datetime.strptime(fileid_pre, '%d-%b-%Y %H:%M:%S')
                    fileids.append(fileid)
        fileids = pd.to_datetime(fileids)
        print(fileids)

        return h5_arrays, fileids, hamd_arrays



    # try:
    #     loaded_datasets = load_h5_files(directory_path, keyword, dataset_name)
    #     # Now you can work with loaded_datasets
    # except FileNotFoundError:
    #     print(f"Directory '{directory_path}' not found.")

    h5_arrays,fileids, hamd_arrays = load_h5_files(path_string, pt_path, file_keyword, dataset_name)



    # %%

    # concatenate data and zscore 
    all_data  = []
    all_data = np.stack(h5_arrays, axis=2)
    print("freqs x channels x trials:", all_data.shape) 
    del h5_arrays
    from scipy.stats import zscore
    psd_z = zscore(all_data,axis = 2) #freqs x channels x trials: e.g. (40, 124, 241)



    # %%
    file = f"/home/rvatsyayan/AnushaData/Pain_Scores_{ptID}.xlsx"
    raw_surveys = pd.read_excel(file)
    print(raw_surveys.shape) # e.g.  (225, 22)
    n_trials = raw_surveys.shape[0]
    raw_surveys

    # %%
    # clean up data 

    # missing_bm_data = np.setdiff1d(raw_surveys.record_id, fileids) # Return the unique values in ar1 that are not in # unique vals in survey ids that dont have neural data files.

    # missing_surveys = np.setdiff1d( fileids, raw_surveys.record_id) # unique vals in neuraldata file ids that dont exist in survey data. 
    missing_bm_data = np.setdiff1d(raw_surveys.Timestamp, fileids)
    missing_surveys = np.setdiff1d(fileids, raw_surveys.Timestamp)
    missing_channels = [i for i in range(all_data.shape[1]) if np.any(all_data[:,i,:])!=0]
    removed = [i for i in range(all_data.shape[1]) if np.any(all_data[:,i,:])==0]
    print((removed))

    electrode_df = pd.read_csv(f"/home/jiahuang/test-code-hazel/{ptID}_new_electrode_property_df.csv")

    electrode_df["edf_ch_idx"] = (
        electrode_df["EDF Channel Number"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .astype(int)
        - 1
    )
    electrode_df_idx = [i for i in range(electrode_df.shape[0]) if str(electrode_df['New ROI Label Hazel'][i])!='nan']
    sorted_electrode_df = electrode_df.iloc[electrode_df_idx].sort_values('New ROI Label Hazel')
    save_idx_sorted = sorted_electrode_df['edf_ch_idx'].to_numpy()
    save_idx_sorted_clean = np.array(
        [i for i in save_idx_sorted if i in missing_channels]
    )
    electrode_df_clean = electrode_df[
        electrode_df["edf_ch_idx"].isin(save_idx_sorted_clean)
    ]
    electrode_df_clean = (
        electrode_df_clean
        .set_index("edf_ch_idx")
        .loc[save_idx_sorted_clean]
        .reset_index()
    )
    electrode_indices_by_ROI = (
        electrode_df_clean
        .groupby("New ROI Label Hazel")["edf_ch_idx"]
        .apply(list)
        .to_dict()
    )

    print(electrode_indices_by_ROI)

    # remove missing survey record ids from neural data. 
    idx_missing_surveys = ~np.isin(fileids, missing_surveys)

    new_alldata = psd_z[:,:,idx_missing_surveys]
    # new_alldata = new_alldata[:,missing_channels,:]
    print(new_alldata.shape)

    # idx_missing_neuraldata = ~np.isin(raw_surveys.record_id, missing_bm_data)
    idx_missing_neuraldata = ~np.isin(raw_surveys.Timestamp, missing_bm_data)
    new_surveys = raw_surveys.iloc[idx_missing_neuraldata]
    print(new_surveys.shape)

    assert(new_surveys.shape[0] == new_alldata.shape[2])

    # %%
    ## reshape and zscore surveys. 
    # psd_z_vec = new_alldata.reshape(new_alldata.shape[2], -1)
    n_freq = new_alldata.shape[0]
    n_ch = len(save_idx_sorted_clean)
    n_feats = n_freq * n_ch  # num of neural features
    n_trials = new_alldata.shape[2]
    # psd_z_vec = np.reshape(new_alldata, (n_feats, n_trials))

    def man_z_score(array):
        array_mean = np.nanmean(array)
        array_std = np.nanstd(array)
        zscore_array = (array - array_mean)/array_std
        return zscore_array
    # new_surveys.fillna(0, inplace=True)
    vasp = new_surveys['intensity_vas_s0'].to_numpy()
    vasd = new_surveys['mood_vas_s0'].to_numpy() # depression/mood survey scores 
    nrs = new_surveys['nrs_s0'].to_numpy()
    unpleasantness = new_surveys['unpleasantness_vas_s0'].to_numpy()
    mpq_somatic = new_surveys.iloc[:,7:18]
    mpq_affective = new_surveys[['tiring_exhausting_s0', 'sickening_s0', 'fearful_s0','punishing_cruel_s0']]
    sum_affective = np.sum(mpq_affective,axis=1).to_numpy()
    sum_somatic = np.sum(mpq_somatic,axis=1).to_numpy()

    vasp_vec = vasp.reshape(-1)
    vasd_vec = vasd.reshape(-1)
    nrs_vec = nrs.reshape(-1)
    sum_affective_vec = sum_affective.reshape(-1)
    sum_somatic_vec = sum_somatic.reshape(-1)
    unpleasantness_vec = unpleasantness.reshape(-1)

    if ptID == 'RCS08':
        mask_valid = (
            ~np.isnan(nrs_vec) &
            ~np.isnan(sum_affective_vec) &
            ~np.isnan(sum_somatic_vec) &
            ~np.isnan(unpleasantness_vec) &
            ~np.isnan(vasp_vec) &
            ((sum_affective_vec + sum_somatic_vec) != 0)
        )
    elif ptID == 'RCS09':
        mask_valid = (
            ~np.isnan(nrs_vec) &
            ~np.isnan(sum_somatic_vec) &
            ~np.isnan(unpleasantness_vec) &
            ~np.isnan(vasp_vec) &
            ((sum_affective_vec + sum_somatic_vec) != 0)
        )
    else:
        mask_valid = (
            ~np.isnan(vasd_vec) &
            ~np.isnan(nrs_vec) &
            ~np.isnan(sum_affective_vec) &
            ~np.isnan(sum_somatic_vec) &
            ~np.isnan(unpleasantness_vec) &
            ~np.isnan(vasp_vec) &
            ((sum_affective_vec + sum_somatic_vec) != 0)
        )

    vasp_clean = vasp_vec[mask_valid]
    vasd_clean = vasd_vec[mask_valid]
    nrs_clean = nrs_vec[mask_valid]
    sum_affective_clean = sum_affective_vec[mask_valid]
    sum_somatic_clean = sum_somatic_vec[mask_valid]
    unpleasantness_clean = unpleasantness_vec[mask_valid]

    vasd_z_clean = man_z_score(vasd_clean)
    nrs_z_clean = man_z_score(nrs_clean)
    affective_z_clean = man_z_score(sum_affective_clean)
    somatic_z_clean = man_z_score(sum_somatic_clean)
    unpleasant_z_clean = man_z_score(unpleasantness_clean)
    vasp_z_clean = man_z_score(vasp_clean)

    new_surveys = new_surveys.loc[mask_valid,:]
    print(new_surveys.shape)
    
    # nan_idx_all  = np.unique(np.concatenate([nan_idx_vasd, nan_idx_nrs,nan_idx_aff]))
    # rcs08: 
    # rcs09: 
    # print(nan_idx_all, len(nan_idx_all))

    # # clean up surveys from nans. 
    # vasd_z_clean = np.delete(vasd_z, [nan_idx_all], axis = 0)
    # vasp_z_clean = np.delete(vasp_z, [nan_idx_all], axis = 0)
    # nrs_z_clean = np.delete(nrs_z, [nan_idx_all], axis = 0)
    # affective_z_clean = np.delete(affective_z, [nan_idx_all], axis = 0)
    # somatic_z_clean = np.delete(somatic_z, [nan_idx_all], axis = 0)
    # unpleasant_z_clean = np.delete(unpleasantness_z,[nan_idx_all], axis=0)

    # remove nans. 
    # new_surveys = new_surveys.drop(new_surveys.index[nan_idx_all], axis=0) # remove by positional index clean_psd_z= np.delete(psd_z_vec, nan_idx_all, axis=1)
    # print(new_surveys.shape)

    # # PCA behavioral composite score
    from sklearn.decomposition import PCA
    # rcs08: X_score = np.stack([nrs_z_clean, affective_z_clean], axis=1)
    # rcs09: X_score = np.stack([nrs_z_clean], axis=1)
    if ptID == 'RCS08':
        X_score = np.stack([unpleasant_z_clean, affective_z_clean,vasp_z_clean, nrs_z_clean, somatic_z_clean], axis=1)
        var_labels = ['unpleasantness','mpq-affective','vasp','nrs','mpq-somatic']
    elif ptID == 'RCS09':
        X_score = np.stack([unpleasant_z_clean,vasp_z_clean, nrs_z_clean, somatic_z_clean], axis=1)
        var_labels = ['unpleasantness','vasp','nrs','mpq-somatic']
    else:
        X_score = np.stack([-vasd_z_clean, unpleasant_z_clean, affective_z_clean,vasp_z_clean, nrs_z_clean, somatic_z_clean], axis=1)
        var_labels = ['vasd-r','unpleasantness','mpq-affective','vasp','nrs','mpq-somatic']

    model = PCA(n_components=3)
    model.fit(X_score)
    pca_stack_score = model.fit_transform(X_score)

    # behav_score_allpt[ptID] = model.fit_transform(X_score)

    # %%
    # import matplotlib.gridspec as gridspec
    # variance_explained = model.explained_variance_ratio_*100
    # cumulative_variance = [variance_explained[0], variance_explained[0]+variance_explained[1], variance_explained[0]+variance_explained[1]+variance_explained[2]]
    # loadings = (model.components_)
    # pc_labels  = [f'PC1 ({np.round(variance_explained[0],2)}%)', f'PC2 ({np.round(variance_explained[1],2)}%)', f'PC3 ({np.round(variance_explained[2],2)}%)']

    # fig = plt.figure(figsize=(12, 4))
    # gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2],wspace=0.5)

    # ax1 = fig.add_subplot(gs[0])
    # x = np.arange(1, len(variance_explained) + 1)
    
    # ax1.bar(x, variance_explained, color='#4c7fc4', width=0.5, zorder=2)
    # ax1.plot(x, cumulative_variance,
    #         color='#e8917a', marker='o', linewidth=2,
    #         markersize=6, zorder=3)
    
    # ax1.set_xlabel('Principal Component', fontsize=11)
    # ax1.set_ylabel('Variance Explained (%)', fontsize=11)
    # ax1.set_title(f'{ptID} Scree Plot', fontsize=13, pad=10)
    # ax1.set_xticks(x)
    # ax1.set_xlim(0.4, len(x) + 0.6)
    # ax1.set_ylim(0, 110)
    # ax1.spines[['top', 'right']].set_visible(False)
    # ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # ax2 = fig.add_subplot(gs[1])
    
    # sns.heatmap(
    #     loadings,
    #     ax=ax2,
    #     annot=True,
    #     fmt='.2f',
    #     cmap='RdBu_r',
    #     vmin=-1, vmax=1,
    #     xticklabels=var_labels,
    #     yticklabels=pc_labels,
    #     linewidths=0.5,
    #     linecolor='white',
    #     cbar_kws={'label': 'Loading', 'shrink': 0.85}
    # )
    
    # ax2.set_title(f'{ptID} PCA Loadings', fontsize=13, pad=10)
    # ax2.tick_params(axis='x', rotation=90, labelsize=10)
    # ax2.tick_params(axis='y', rotation=0,  labelsize=10)
    # plt.savefig(f"{fig_save}PCA_loadings_6var.png")
    # plt.show()

    # %%
    new_labels = sorted(['IC', 'Caudate', 'ACC', 'MCC','OFC', 'AINS', 'PINS','dmPFC','dlPFC','THAL'])
    new_labels_bi = ['L '+nl for nl in new_labels] + ['R '+ nl for nl in new_labels]

    # Dimension reduction of all_data
    clean_alldata = all_data[:,:,idx_missing_surveys]
    clean_alldata= clean_alldata[:,:,mask_valid]
    # clean_alldata = clean_alldata[:,missing_channels,:]
    print(clean_alldata.shape)
    n_ch_all = clean_alldata.shape[1]
    n_trials = clean_alldata.shape[2]

    band_data = np.zeros((6, n_ch_all, n_trials))
    bandref = {'delta':(1,4), 'theta':(5,8), 'alpha':(9,12), 'beta':(13,30), 'low gamma':(31,70), 'high gamma':(71,150)}
    bands = list(bandref.keys())

    import re
    ch_labels = []
    freqs = []
    if ptID == 'RCS08':
        with h5py.File(f"{path_string}/5_meanpsd.h5", 'r') as hf:
            freqs = hf.attrs['freqs']
    else:
        with h5py.File(f"{path_string}/1_meanpsd.h5", 'r') as hf:
            freqs = hf.attrs['freqs']
    canonical_freq = [bandref[bands[i]][0] for i in range(len(bands))]


    for b in range(len(bands)):
        band = bands[b]
        (fmin, fmax) = bandref[band]
        idxf = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_data[b, :, :] = np.mean(clean_alldata[idxf, :, :], axis=0)

    large_ROI = list(electrode_indices_by_ROI.keys())
    ch_data = np.zeros((len(bands), len(large_ROI), n_trials))

    for i in range(len(large_ROI)):
        ch = large_ROI[i]
        idxc = electrode_indices_by_ROI[ch]
        ch_data[:,i,:] = np.mean(band_data[:, idxc, :], axis = 1)

    ch_data_new = np.zeros((len(freqs), len(large_ROI), n_trials))

    for i in range(len(large_ROI)):
        ch = large_ROI[i]
        idxc = electrode_indices_by_ROI[ch]
        ch_data_new[:,i,:] = np.mean(clean_alldata[:, idxc, :], axis = 1)
        
    all_data_clean_freq_roi = zscore(ch_data, axis = 2)
    all_data_clean_freq_roi_z = all_data_clean_freq_roi.reshape(len(bands)*len(large_ROI), n_trials).T
    all_data_clean_roi_z = zscore(ch_data_new, axis=1).reshape(len(freqs)*len(large_ROI), n_trials).T

    print(large_ROI)
    print(all_data_clean_freq_roi_z.shape)

    # %%
    all_data_clean_roi_z_allpt[ptID] = all_data_clean_roi_z
    all_data_clean_freq_roi_z_allpt[ptID] = all_data_clean_freq_roi_z
    ch_allpt[ptID] = large_ROI
    behav_score_allpt[ptID] = pca_stack_score
    # behav_score_allpt[ptID] = sum_score
    # behav_score_allpt[ptID] = pca_stack_score
    # pca_var[ptID] = model.explained_variance_ratio_

    # %%
    # Behavior score correlation
    from scipy.stats import linregress

    def annotate_r2(x, y, **kws):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            return
        
        slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
        r2 = r_value**2
        
        ax = plt.gca()
        ax.text(0.05, 0.9, f"$R^2$ = {r2:.2f}",
                transform=ax.transAxes,
                fontsize=10)
    
    if ptID == 'RCS08':
        score_df = pd.DataFrame({
            'Unpleasantness-z': unpleasant_z_clean,
            'MPQ-affective-z':affective_z_clean,
            'VASP-z': vasp_z_clean,
            'NRS-z': nrs_z_clean,
            'MPQ-somatic-z': somatic_z_clean
        })

    elif ptID == 'RCS09':
        score_df = pd.DataFrame({
            'Unpleasantness-z': unpleasant_z_clean,
            'VASP-z': vasp_z_clean,
            'NRS-z': nrs_z_clean,
            'MPQ-somatic-z': somatic_z_clean
        })

    else: 
        score_df = pd.DataFrame({
            'VASD-z-r': -vasd_z_clean,
            'Unpleasantness-z': unpleasant_z_clean,
            'MPQ-affective-z':affective_z_clean,
            'VASP-z': vasp_z_clean,
            'NRS-z': nrs_z_clean,
            'MPQ-somatic-z': somatic_z_clean
        })
    # sns.set_theme('paper')
    g = sns.PairGrid(score_df)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.regplot,scatter_kws={'s': 10},robust=False)
    g.map_offdiag(annotate_r2)
    g.fig.suptitle(f"{ptID} Behvaior Score Correlation",y=1.02)
    g.fig.savefig(f"{fig_save}{ptID}_behavior_score_correlation.png", dpi=300, bbox_inches='tight')


sys.path.append("/home/jiahuang/test-code-hazel/gen_fxns/")
savepath = f"/userdata/jiahuang/pain-data/figures/Stage1-test/"
import os
os.makedirs(savepath, exist_ok=True)

def basic_heatmap(ax, array, ch_labels, freqs, cbar_title, title, save=False):
    fig_params = [8, 12]
    caxis_lim = [-0.1, 0.2]
    
    sns.heatmap((array.T), cmap="RdBu_r", 
                vmax = np.nanmax(np.abs(array)), vmin = -np.nanmax(np.abs(array)), center = 0,cbar=True,
                yticklabels=new_labels_bi, xticklabels=np.round(freqs), 
                cbar_kws={'label': cbar_title}, ax=ax)
    missing_cols = [j for j, roi in enumerate(new_labels_bi) if roi not in list(ch_labels)]
    print(missing_cols)
    for col in missing_cols:
        ax.add_patch(plt.Rectangle((0, col), array.shape[0], 1,
            fill=True, facecolor='lightgray', alpha=0.6,
            hatch='////', edgecolor='gray', linewidth=0,
            zorder=3
        ))
    ax.set_xlabel("Frequency(Hz)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    if save: plt.savefig(f"{savepath}{title}.png")

def basic_heatmap_onlyone(array, ch_labels, freqs, cbar_title, title, save=False, savepath=""):
    plt.figure(figsize=(8, 12))

    vmax = np.nanmax(np.abs(array))

    ax = sns.heatmap(array.T, cmap="RdBu_r", vmax=vmax, vmin=-vmax, center=0, cbar=True,
        yticklabels=new_labels_bi,
        xticklabels=np.round(freqs),
        cbar_kws={'label': cbar_title}
    )

    missing_cols = [j for j, roi in enumerate(new_labels_bi) if roi not in list(ch_labels)]

    for col in missing_cols:
        ax.add_patch(plt.Rectangle((0, col), array.shape[0],1,
            facecolor='lightgray',alpha=0.6,hatch='////',edgecolor='gray',
            linewidth=0,zorder=3
        ))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Channel")
    plt.title(title)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()

    if save:
        plt.savefig(f"{savepath}{title}.png", dpi=300, bbox_inches='tight')

    plt.close()

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

    for i, (array, ch_label, cbar_title, title) in enumerate(zip(data_list, ch_labels, cbar_titles, titles)):
        basic_heatmap(axes[i], array, ch_label, freqs, cbar_title, title)

    plt.tight_layout()
    save_dir = f"/userdata/jiahuang/pain-data/figures/Stage1-test/{savetitle}"

    plt.savefig(save_dir, dpi=300, edgecolor='k', facecolor="white")
    plt.show()

# %%
# Correlation with permutation test
def run_corr(x,y, n_freq, n_ch):
    from scipy.stats import spearmanr
    corr,p_value = spearmanr(x,y)
    correlations = corr[:-1,-1]
    p = p_value[:-1,-1].reshape(n_freq,n_ch)
    new_corr = correlations.reshape(n_freq,n_ch)
    return new_corr, p

def corr_permutation(X, Y, n_freq, n_ch, n_perm=1000):
    corr_null = np.zeros((n_perm, n_freq,n_ch))
    corr_real, __ = run_corr(X,Y,n_freq,n_ch) #return n_freq*n_ch

    for i in range(n_perm):
        Y_perm  = np.random.permutation(Y)
        null_stats, __ = run_corr(X,Y_perm, n_freq, n_ch)
        corr_null[i,:,:] = null_stats
        
    p = np.mean(np.abs(corr_null) >= np.abs(corr_real),axis=0).reshape(n_freq,n_ch)
    return corr_real, p

def make_no_data_ch_nan(data, large_ROI):
    full = np.full((data.shape[0], 20), np.nan)
    for j, roi in enumerate(new_labels_bi):
        if roi in large_ROI:
            idx = list(large_ROI).index(roi)
            full[:, j] = data[:, idx]
    return full

corr_pca_allpt = {}
corr_pca2_allpt = {}
for pt in ptIDs:
    corr_pca, perm_p = corr_permutation(all_data_clean_freq_roi_z_allpt[pt], behav_score_allpt[pt][:,0], len(bands), len(ch_allpt[pt]))
    corr_pca_mask = np.where(perm_p>0.05, np.nan, corr_pca)
    corr_pca_allpt[pt] = make_no_data_ch_nan(corr_pca_mask, ch_allpt[pt])
    corr_pca2, perm_p = corr_permutation(all_data_clean_freq_roi_z_allpt[pt], behav_score_allpt[pt][:,1], len(bands), len(ch_allpt[pt]))
    corr_pca2_mask = np.where(perm_p>0.05, np.nan, corr_pca2)
    corr_pca2_allpt[pt] = make_no_data_ch_nan(corr_pca2_mask, ch_allpt[pt])
    basic_heatmap_onlyone(corr_pca_allpt[pt],ch_allpt[pt],canonical_freq,'Corr',f'PSD vs PCA comp1 behavioral {pt} FINAL LABEL',True,savepath)
    basic_heatmap_onlyone(corr_pca_allpt[pt],ch_allpt[pt],canonical_freq,'Corr',f'PSD vs PCA comp2 behavioral {pt} FINAL LABEL',True,savepath)
plot_heatmap_subplots([corr_pca_allpt['RCS02'], corr_pca_allpt['RCS03'], corr_pca_allpt['RCS04'], corr_pca_allpt['RCS05'], corr_pca_allpt['RCS06'],corr_pca_allpt['RCS08'],corr_pca_allpt['RCS09']], list(dict(sorted(ch_allpt.items())).values()), canonical_freq, ['Corr', 'Corr', 'Corr','Corr','Corr', 'Corr','Corr'], ['PSD vs PCA1 behavioral RCS02', 'PSD vs PCA1 behavioral RCS03', 'PSD vs PCA1 behavioral RCS04', 'PSD vs PCA1 behavioral RCS05', 'PSD vs PCA1 behavioral RCS06','PSD vs PCA1 behavioral RCS08','PSD vs PCA1 behavioral RCS09'], "04282026 FINAL Masked Canonical chanel roi correlation till rcs09 - PCA behavioral 6 var comp1- new label - permutation ", nrows=2, ncols=4)
plot_heatmap_subplots([corr_pca2_allpt['RCS02'], corr_pca2_allpt['RCS03'], corr_pca2_allpt['RCS04'], corr_pca2_allpt['RCS05'], corr_pca2_allpt['RCS06'],corr_pca2_allpt['RCS08'], corr_pca2_allpt['RCS09']], list(dict(sorted(ch_allpt.items())).values()), canonical_freq, ['Corr', 'Corr', 'Corr','Corr','Corr', 'Corr','Corr'], ['PSD vs PCA2 behavioral RCS02', 'PSD vs PCA2 behavioral RCS03', 'PSD vs PCA2 behavioral RCS04', 'PSD vs PCA2 behavioral RCS05', 'PSD vs PCA2 behavioral RCS06','PSD vs PCA2 behavioral RCS08','PSD vs PCA2 behavioral RCS09'], "04282026 FINAL Masked all freq roi correlation till rcs09 - PCA behavioral 6 var comp2 - new label - permutation ", nrows=2, ncols=4)


