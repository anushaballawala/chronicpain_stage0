# %%

# %%
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

def mad_artifact_removal(psd_data, threshold_factor):
    """Use Median Absolute Deviation for robust outlier detection."""
    cleaned_psd = psd_data.copy()
    num_freqs, num_timepoints, num_ch = psd_data.shape
    
    stats = {'total_removed': 0, 'total_points': 0}
    
    for ch in range(num_ch):
        for freq in range(num_freqs):
            data = psd_data[freq, :, ch]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                if mad < 1e-8: mad = 1e-8
                
                mod_z = 0.6745 * (data - median) / mad
                outliers = np.abs(mod_z) > threshold_factor
                cleaned_psd[freq, outliers, ch] = np.nan
                
                stats['total_removed'] += np.sum(outliers)
                stats['total_points'] += len(data)
    
    stats['overall_percentage'] = (stats['total_removed'] / stats['total_points']) * 100
    return cleaned_psd, stats

def process_file(filepath, threshold_factor):
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    output_psd = os.path.join(base_dir, filename.replace("_wavelet.h5", "_meanpsd_clean.h5"))
    output_stats = os.path.join(base_dir, filename.replace("_wavelet.h5", "_artifact_stats.pkl"))
    
    print(f"  Processing: {filename}")
    with h5py.File(filepath, "r") as hf_in:
        raw_psd = np.array(hf_in["decomp_signal"], dtype=np.float32)
        psd_clean, stats = mad_artifact_removal(raw_psd, threshold_factor=threshold_factor)
        mean_psd = np.nanmean(psd_clean, axis=1)
        
        with h5py.File(output_psd, "w") as hf_out:
            hf_out.create_dataset("mean_psd", data=mean_psd)
            for attr in hf_in.attrs:
                hf_out.attrs[attr] = hf_in.attrs[attr]
        
        with open(output_stats, "wb") as f:
            pickle.dump(stats, f)

    return mean_psd, stats
# %%
behav_score_allpt = {}
all_data_clean_freq_z_allpt = {}
all_channel_labels_allpt = {}
ch_allpt = {}
ptIDs = [
        'RCS02'
         ,'RCS03','RCS04','RCS05','RCS06','RCS08'
         ,'RCS09'
        ]

# %%
for ptID in ptIDs:
    path_string = f"/userdata/jiahuang/pain-data/Stage1-test/{ptID}/biomarker/preproc_data/202605_newpreproc_all_channels/"
    fig_save = f"/userdata/jiahuang/pain-data/Stage1-test/{ptID}/meanpsd_mad_4point5"
    pt_path = f"/userdata/rvatsyayan/AnushaData/HDF5 Pain Data/{ptID}"

    savepath = f"/userdata/jiahuang/pain-data/figures/Stage1-test/062026-newpreproc/meanpsd_mad_4point5"
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(fig_save, exist_ok=True)

    # pt_path = f"/home/rvatsyayan/AnushaData"
    electrode_df = pd.read_csv(f"/home/jiahuang/test-code-hazel/{ptID}_new_electrode_property_df.csv")
    data_root = Path(path_string)
    # file_keyword = '_meanpsd_clean'
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
        removed_mad = []
        fileids = [] 
        # Get list of files in directory and sort them based on numeric part
        files = sorted(os.listdir(path_string), key=extract_number)
        # hamd_files = sorted(os.listdir(hamd_path), key=extract_number)
        
        for filename in files:
            if filename.endswith('meanpsd_clean.h5'):
            # and file_keyword in filename:
                # Construct the full file path
                filepath = os.path.join(path_string, filename)
                # mean_psd, stats = process_file(filepath, threshold_factor=4.5)

                # print(filepath)
                # Load the .h5 file
                with h5py.File(filepath, 'r') as hf:
                    # Assuming you want to load the first dataset from each file
                    # Load the dataset as float32
                    dataset = np.array(hf[dataset_name], dtype=np.float32)
                    
                #     raw_psd = np.array(hf["decomp_signal"], dtype=np.float32)
                #     psd_clean, stats = mad_artifact_removal(raw_psd, threshold_factor=4.5)
                #     mean_psd = np.nanmean(psd_clean, axis=1)

                    # Append the dataset to the list
                    h5_arrays.append(dataset)
                # h5_arrays.append(mean_psd)
                # removed_mad.append(stats['overall_percentage'])

                    

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

        return h5_arrays, fileids, removed_mad



    # try:
    #     loaded_datasets = load_h5_files(directory_path, keyword, dataset_name)
    #     # Now you can work with loaded_datasets
    # except FileNotFoundError:
    #     print(f"Directory '{directory_path}' not found.")

    h5_arrays,fileids, removed_mad = load_h5_files(path_string, pt_path, file_keyword, dataset_name)
    # pd.DataFrame(removed_mad).to_csv(rf"{fig_save}/{ptID}_mad_stats_all_trial.csv")


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

    from sklearn.decomposition import PCA
    import numpy as np

    if ptID == 'RCS08':

        X_sensory = np.stack([
            vasp_z_clean,
            nrs_z_clean,
            somatic_z_clean
        ], axis=1)

        sensory_labels = [
            'vasp',
            'nrs',
            'mpq-somatic'
        ]

        sensory_pca = PCA(n_components=3)
        sensory_latent = sensory_pca.fit_transform(X_sensory)

        X_affective = np.stack([
            unpleasant_z_clean,
            affective_z_clean
        ], axis=1)

        affective_labels = [
            'unpleasantness',
            'mpq-affective'
        ]

        affective_pca = PCA(n_components=2)
        affective_latent = affective_pca.fit_transform(X_affective)

    elif ptID == 'RCS09':
        X_sensory = np.stack([
            vasp_z_clean,
            nrs_z_clean,
            somatic_z_clean
        ], axis=1)

        sensory_labels = [
            'vasp',
            'nrs',
            'mpq-somatic'
        ]

        sensory_pca = PCA(n_components=3)
        sensory_latent = sensory_pca.fit_transform(X_sensory)

        X_affective = np.stack([
            unpleasant_z_clean
        ], axis=1)

        affective_labels = [
            'unpleasantness'
        ]

        affective_latent = unpleasant_z_clean.copy()


    else:
        X_sensory = np.stack([
            vasp_z_clean,
            nrs_z_clean,
            somatic_z_clean
        ], axis=1)

        sensory_labels = [
            'vasp',
            'nrs',
            'mpq-somatic'
        ]

        sensory_pca = PCA(n_components=3)
        sensory_latent = sensory_pca.fit_transform(X_sensory)

        X_affective = np.stack([
            -vasd_z_clean,
            unpleasant_z_clean,
            affective_z_clean
        ], axis=1)

        affective_labels = [
            'vasd-r',
            'unpleasantness',
            'mpq-affective'
        ]

        affective_pca = PCA(n_components=3)
        affective_latent = affective_pca.fit_transform(X_affective)

    # ============================================
    # STORE PCA RESULTS
    # ============================================

    pca_results = {}
    pca_results['sensory'] = {
        'model': sensory_pca,
        'scores': sensory_latent,
        'loadings': sensory_pca.components_,
        'variance_explained': sensory_pca.explained_variance_ratio_,
        'labels': sensory_labels
    }

    if ptID != 'RCS09':
        pca_results['affective'] = {
            'model': affective_pca,
            'scores': affective_latent,
            'loadings': affective_pca.components_,
            'variance_explained': affective_pca.explained_variance_ratio_,
            'labels': affective_labels
        }
    else:
        pca_results['affective'] = {
            'scores': affective_latent,
            'labels': affective_labels
        }

    # %%
    new_labels = sorted(['IC', 'Caudate', 'ACC', 'MCC','OFC', 'AINS', 'PINS','dmPFC','dlPFC','THAL'])
    new_labels_bi = ['L '+nl for nl in new_labels] + ['R '+ nl for nl in new_labels]

    hypoth_labels = sorted(['ACC', 'MCC', 'AINS', 'PINS','THAL'])
    hypoth_labels_bi = ['L '+nl for nl in hypoth_labels] + ['R '+ nl for nl in hypoth_labels]

    # Dimension reduction of all_data
    clean_alldata = psd_z[:,:,idx_missing_surveys]
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
        band_data[b, :, :] = np.nanmean(clean_alldata[idxf, :, :], axis=0)

    large_ROI = list(electrode_indices_by_ROI.keys())
    ch_data = np.zeros((len(bands), len(large_ROI), n_trials))

    all_channel_labels = (
        electrode_df_clean["New ROI Label Hazel"].astype(str) + "_" + electrode_df_clean.groupby("New ROI Label Hazel").cumcount().add(1).astype(str)
    ).tolist()    

    all_channel_labels_allpt[ptID] = all_channel_labels
    band_data_temp = band_data[:,save_idx_sorted_clean,:]
    
    
 
    
    all_data_clean_freq_z = zscore(band_data_temp, axis = 2).reshape(len(bands)*len(all_channel_labels), n_trials).T
    # all_data_clean_freq_z = zscore(band_data_temp, axis = 2).reshape(len(bands)*len(all_cha), n_trials).T

    # %%
    all_data_clean_freq_z_allpt[ptID] = all_data_clean_freq_z
    # ch_allpt[ptID] = all_channel_labels
    behav_score_allpt[ptID] = pca_results


# %%
sys.path.append("/home/jiahuang/test-code-hazel/gen_fxns/")


def basic_heatmap(ax, array, ch_labels, freqs, cbar_title, title, save=False):
    fig_params = [16, 8]
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
    if save: plt.savefig(f"{savepath}/{title}.png")

def basic_heatmap_onlyone(array, ch_labels, freqs, cbar_title, title, save=False, savepath=""):
    plt.figure(figsize=(16, 5))

    vmax = np.nanmax(np.abs(array))

    ax = sns.heatmap(array, cmap="RdBu_r", vmax=0.7, vmin=-vmax, center=0, cbar=True,
        xticklabels=ch_labels,
        yticklabels=np.round(freqs),
        cbar_kws={'label': cbar_title}
    )

    # missing_cols = [j for j, roi in enumerate(new_labels_bi) if roi not in list(ch_labels)]

    # for col in missing_cols:
    #     ax.add_patch(plt.Rectangle((0, col), array.shape[0],1,
    #         facecolor='lightgray',alpha=0.6,hatch='////',edgecolor='gray',
    #         linewidth=0,zorder=3
    #     ))

    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Channel")
    plt.title(title)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()

    if save:
        plt.savefig(f"{savepath}/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()
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
    save_dir = f"{savepath}/{savetitle}"

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

from statsmodels.stats.multitest import fdrcorrection

def fdr_mask(corr, pvals, alpha=0.05):

    p_flat = pvals.flatten()

    reject, p_fdr = fdrcorrection(
        p_flat,
        alpha=alpha
    )

    reject = reject.reshape(pvals.shape)

    corr_fdr = np.where(
        reject,
        corr,
        np.nan
    )

    return corr_fdr, reject


def fdr_mask_roiwise(corr, pvals, alpha=0.05):

    n_freq, n_roi = pvals.shape

    reject_all = np.zeros_like(pvals, dtype=bool)

    for roi in range(n_roi):

        reject_roi, _ = fdrcorrection(
            pvals[:, roi],
            alpha=alpha
        )

        reject_all[:, roi] = reject_roi

    corr_fdr = np.where(
        reject_all,
        corr,
        np.nan
    )

    return corr_fdr, reject_all

def make_no_data_ch_nan(data, large_ROI):
    full = np.full((data.shape[0], 20), np.nan)
    for j, roi in enumerate(new_labels_bi):
        if roi in large_ROI:
            idx = list(large_ROI).index(roi)
            full[:, j] = data[:, idx]
    return full

def get_behavior_score(pt_dict, behavior_type):

    if behavior_type == 'all':
        return pt_dict['all']['scores'][:,0]

    elif behavior_type == 'sensory':
        return pt_dict['sensory']['scores'][:,0]

    elif behavior_type == 'affective':

        if len(pt_dict['affective']['scores'].shape) == 1:
            return pt_dict['affective']['scores']

        return pt_dict['affective']['scores'][:,0]

corr_band_allpt = {
    'all': {},
    'sensory': {},
    'affective': {}
}

corr_raw_allpt = {
    'all': {},
    'sensory': {},
    'affective': {}
}


behavior_types = ['sensory', 'affective']

            
for pt in ptIDs:
    for behavior in behavior_types:
        y = get_behavior_score(
            behav_score_allpt[pt],
            behavior
        )

        # =====================================
        # CANONICAL BAND PSD
        # =====================================

        corr_band, perm_p = corr_permutation(
            all_data_clean_freq_z_allpt[pt],
            y,
            len(bands),
            len(all_channel_labels_allpt[pt])
        )

        # =====================================
        # uncorrected
        # =====================================

        corr_band_uncorrected = np.where(
            perm_p > 0.05,
            np.nan,
            corr_band
        )
        
        
        # =====================================
        # FDR corrected
        # =====================================

        corr_band_fdr, reject_fdr = fdr_mask_roiwise(
            corr_band,
            perm_p
        )

        corr_band_allpt[behavior][pt] = {
            'uncorrected': corr_band_uncorrected,
            'fdr': corr_band_fdr,
            'raw': corr_band
        }

        basic_heatmap_onlyone(corr_band_allpt[behavior][pt]['uncorrected'],all_channel_labels_allpt[pt],canonical_freq,
                            'Corr', f'{pt} Canonical zPSD vs {behavior} PC1 Uncorrected', True, savepath)
        basic_heatmap_onlyone(corr_band_allpt[behavior][pt]['fdr'],all_channel_labels_allpt[pt],canonical_freq,
                            'Corr', f'{pt} Canonical zPSD vs {behavior} PC1 ROI FDR', True, savepath)
        basic_heatmap_onlyone(corr_band_allpt[behavior][pt]['raw'],all_channel_labels_allpt[pt],canonical_freq,
                            'Corr', f'{pt} Canonical zPSD vs {behavior} PC1 Raw', True, savepath)
        


# %%



