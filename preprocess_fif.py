#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sp
import sys
from datetime import date
import mne 
sys.path.append("/home/aallawala/iEEG-GenFxns/Signal_proc")

filepath = sys.argv[1] 
# filepath = "/datastore_spirit/human/ChronicPain_NK/js_decoding/all_bipolar/RCS04/during_surv/fif/RCS04_951_eeg_raw.fif"
ptID = 'RCS05'
# filepath = '/datastore_spirit/human/ChronicPain_NK/js_decoding/all_bipolar/%s/during_surv/fif/' % (ptID)
# filename = '%s_869_eeg_raw.fif' % (ptID)
# fullfile = filepath + filename

# filepath = '/datastore_spirit/human/ChronicPain_NK/js_decoding/all_bipolar/RCS04/during_surv/fif/RCS04_900_eeg_raw.fif'
print(filepath)

raw = mne.io.read_raw_fif(filepath)
data,times = raw[:]
rec_timestamp = raw.info['meas_date']
sfreq = raw.info['sfreq']
raw_ch_labels = raw.info['ch_names'][:]
n_ch = raw.info['nchan']
import re
tmp = re.findall(r'\d+', filepath)
res = list(map(int, tmp))
fileid = res[2]

remove_pol = lambda s: s.replace('POL ', '').replace('-Ref', '')

modified_list_for_loop = []
for s in raw_ch_labels:
    modified_list_for_loop.append(remove_pol(s))

print(modified_list_for_loop)

# Apply lambda function to each string in the list using list comprehension
ch_label = [remove_pol(s) for s in raw_ch_labels]

from preproc_fxns import butterworth_notch_filter

notch_freqs = [60, 120, 180, 240]
order = 4       # Example value, replace with the actual value

filt_data = butterworth_notch_filter(data.T,order, notch_freqs[0], sfreq, n_ch)
filt_data = butterworth_notch_filter(filt_data,order, notch_freqs[1], sfreq, n_ch)
filt_data = butterworth_notch_filter(filt_data,order, notch_freqs[2], sfreq, n_ch)
filt_data = butterworth_notch_filter(filt_data,order, notch_freqs[3], sfreq, n_ch)

num_frequencies = 40 #updated to capture more frequencies 03/22/24 
min_freq = 1  # Hz
max_freq = 130  # Hz

freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frequencies)


wave_num = [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 10, 10, 10, 10, 11, 11, 12, 13, 14, 16, 16, 17, 17, 18, 18, 18, 18]
len(wave_num)

# # wavelet analysis 

def decomp_wavelet(data,freqs,srate,wave_num):
    from scipy.signal import convolve

    tmp_amplitude = np.zeros((len(freqs), len(data)), dtype=float)
    tmp_phase = np.zeros((len(freqs), len(data)), dtype=float)

    # loop through frequencies, build new wavelet, convolve
    for fi in range(len(freqs)):
        # initialize wave_num related variables for each fi
        
        # wavelet cycles
        wavelet_cycles = wave_num[fi]

        # set wavelet window size, using lowest freq, wave number, and sample rate
        # high-freqs will have greater zero padding
        lowest_freq = freqs[fi]
        max_win_size = (1 / lowest_freq) * (wavelet_cycles / 2)
        max_win_size = max_win_size * 2.5  # add 150% length to ensure zero is reached
        
        # wavelet window
        #wavelet_win = np.arange(-max_win_size, max_win_size + 1 / srate, 1 / srate)
        wavelet_win = np.arange(-max_win_size, max_win_size, 1/srate)

        # Ensure the length of wavelet_win matches the length of data
        if len(wavelet_win) > len(data):
            wavelet_win = wavelet_win[:len(data)]
            
        # initialize variables
        tmp_freq_analytic = np.zeros_like(data, dtype=complex)
        tmp_sine = np.zeros_like(wavelet_win, dtype=complex)
        tmp_gaus_win = np.zeros_like(wavelet_win)
        tmp_wavelet = np.zeros_like(wavelet_win, dtype=complex)

        # create sine wave at center frequency
        tmp_sine = np.exp(2j * np.pi * freqs[fi] * wavelet_win)

        # make Gaussian window, with a width/sd = cycles
        #tmp_gaus_win = np.exp(-wavelet_win**2 / (2 * (wavelet_cycles / (2 * np.pi * freqs[fi]))**2))
        tmp_gaus_win = np.exp(-np.power(wavelet_win, 2) / (2 * np.power(wavelet_cycles / (2 * np.pi * freqs[fi]), 2)))

        # make wavelet as the dot-product of the sine wave and Gaussian window
        tmp_wavelet = tmp_sine * tmp_gaus_win

        # convolve data with wavelet - remove zero padding ('same' length as input)
        # pre-flip kernel, to deal with flip in conv, keeps phase ok?
        tmp_freq_analytic = convolve(data, tmp_wavelet[::-1], mode='same')

        # extract amplitude and phase data
        # apply amplitude normalization in function?
        tmp_amplitude[fi, :] = np.abs(tmp_freq_analytic)
        # tmp_phase[fi, :] = np.angle(tmp_freq_analytic)  # phase

    # collect data
    decomp_signal_amplitude = tmp_amplitude.astype(np.float32)
    decomp_signal_phase = tmp_phase

    
    return decomp_signal_amplitude, decomp_signal_phase


import timeit

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is" + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


tic()
decomp_signal_array = np.zeros(shape=(len(freqs),len(filt_data),n_ch), dtype=np.float32)
#decomp_signal_phase = np.zeros(shape=(len(freqs),len(reref_data),num_ch))

for ichan in range(n_ch):
    tmp_amplitude, tmp_phase = decomp_wavelet(filt_data[:,ichan].T,freqs,sfreq,wave_num)
    decomp_signal_array[:,:,ichan] = tmp_amplitude
    #decomp_signal_phase[:,:,ichan] = tmp_phase

print('wavelet decomposition completed')
(decomp_signal_array.nbytes)/1024**3
decomp_signal_array.dtype
toc()

# sample every 7th sample 

resampled_psd = decomp_signal_array[:,::7,:]

mean_psd = np.mean(a=resampled_psd,axis = 1)

# save as h5 file 
import os
directory, filename = os.path.split(filepath)
filename_no_ext, ext = os.path.splitext(filename)

str1 = str(fileid)
str2 = '_ieeg_wavelet.h5'
h5_filename = ptID + '_' + str1 + str2
h5_filepath = '/userdata/aallawala/pain_data/stage0/%s/mood_biomarker/preproc_data/' % (ptID)
full_filename = h5_filepath + h5_filename
version = '1.0'
print(full_filename)

hf = h5py.File(full_filename, 'w')

hf.create_dataset('resampled_psd', data=resampled_psd)
hf.attrs['description'] = 'data preprocessed: notch filter, PSD computed, resampled timesamples in PSD'
hf.attrs['filename'] = filename
hf.attrs['srate_new'] = sfreq
hf.attrs['notch_freqs'] = notch_freqs
hf.attrs['freqs'] = freqs
hf.attrs['version'] = version
hf.attrs['ch_labels'] = ch_label
hf.attrs['PatientID'] = ptID
hf.attrs['num_ch'] = n_ch
hf.attrs['rec_timestamp'] = str(rec_timestamp.isoformat)
# get date
today_date = date.today()
date_string = today_date.strftime("%Y-%m-%d")
hf.attrs['date preprocessed'] = date_string

print('wavelet data saved')

hf.close()

# save mean data. 

str1 = str(fileid)
str2 = '_ieeg_wavelet_mean.h5'
h5_filename = ptID + '_' + str1 + str2
h5_filepath = '/userdata/aallawala/pain_data/stage0/%s/mood_biomarker/preproc_data/' % (ptID)
full_filename = h5_filepath + h5_filename
version = '1.0'
print(full_filename)

hf_n = h5py.File(full_filename, 'w')
hf_n.create_dataset(name='mean_psd', data=mean_psd)

hf_n.attrs['filename'] = filename
hf_n.attrs['srate_new'] = sfreq
hf_n.attrs['freqs'] = freqs
hf_n.attrs['version'] = "v1.0"
hf_n.attrs['ch_labels'] = ch_label
hf_n.attrs['PatientID'] = ptID
hf_n.attrs['n_ch'] = n_ch
hf_n.attrs['fileid'] = fileid
hf_n.attrs['rec_timestamp'] = str(rec_timestamp.isoformat)
# get date
today_date = date.today()
date_string = today_date.strftime("%Y-%m-%d")
hf_n.attrs['date data processed'] = date_string
hf_n.close()
print('mean psd saved')

