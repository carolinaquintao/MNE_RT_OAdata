import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import os.path as op
from scipy.signal import freqz
from mne.filter import create_filter
from mne.preprocessing import ICA

raw = mne.io.read_raw_edf('C:/Users/ufpa/OneDrive - Universidade Federal do Pará - UFPA/Naynara/BaseDeDadosITV/suj1_36_B/sampleData.edf', preload=True)
print(raw)

electrodes = ['EEG P3 - Pz', 'EEG C3 - Pz', 'EEG F3 - Pz', 'EEG Fz - Pz', 'EEG F4 - Pz', 'EEG C4 - Pz', 'EEG P4 - Pz',
              'EEG Cz - Pz', 'EEG CM - Pz', 'EEG A1 - Pz', 'EEG Fp1 - Pz', 'EEG Fp2 - Pz', 'EEG T3 - Pz', 'EEG T5 - Pz',
              'EEG O1 - Pz', 'EEG O2 - Pz', 'EEG F7 - Pz', 'EEG F8 - Pz', 'EEG A2 - Pz', 'EEG T6 - Pz', 'EEG T4 - Pz']

montage = mne.channels.make_standard_montage('standard_1020')
#montage = mne.channels.make_dig_montage(kind='standard_1020', ch_names=electrodes, unit='m', transform=False)
print(montage)
raw.set_montage(montage, on_missing='ignore')
# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw, picks=0)

orig_raw = raw.copy()
raw.load_data()
mne.time_frequency.psd_welch(raw)
ica.apply(raw)

# show some frontal channels to clearly illustrate the artifact removal
chs = ['EEG Fz - Pz', 'EEG F4 - Pz', 'EEG F3 - Pz', 'EEG Fp1 - Pz', 'EEG Fp2 - Pz']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=12, duration=4)
raw.plot(order=chan_idxs, start=12, duration=4)

print('Hi, PyCharm')
