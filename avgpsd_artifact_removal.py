import h5py
import numpy as np
import os
import pickle
import sys
from scipy.stats import zscore

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

def main(filepath):
    # Output file names
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    output_psd = os.path.join(base_dir, filename.replace("_wavelet.h5", "_meanpsd_clean.h5"))
    output_stats = os.path.join(base_dir, filename.replace("_wavelet.h5", "_artifact_stats.pkl"))

    print(f"Processing: {filename}")

    with h5py.File(filepath, "r") as hf_in:
        # Load [Freq x Time x Channel]
        raw_psd = np.array(hf_in["decomp_signal"], dtype=np.float32)
        psd_z = zscore(raw_psd, axis=1)
        # Artifact removal
        cleaned_psd, stats = mad_artifact_removal(psd_z, threshold_factor=3.5)
        
        # Collapse Time via nanmean to get Total Power
        mean_psd = np.nanmean(cleaned_psd, axis=1) 
        
        # Save Mean PSD
        with h5py.File(output_psd, "w") as hf_out:
            hf_out.create_dataset("mean_psd", data=mean_psd)
            for attr in hf_in.attrs:
                hf_out.attrs[attr] = hf_in.attrs[attr]

        # Save stats
        with open(output_stats, "wb") as f:
            pickle.dump(stats, f)
    
    print(f"Done. Removed {stats['overall_percentage']:.2f}% artifacts.")

if __name__ == "__main__":
    main(sys.argv[1])

    