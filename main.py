import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal
import os.path as op
from scipy.signal import freqz
from mne.filter import create_filter
from mne.preprocessing import ICA

raw = mne.io.read_raw_edf('C:/Users/ufpa/OneDrive - Universidade Federal do Pará - UFPA/Naynara/BaseDeDadosITV/suj1_36_B/suj1_36_28012019_RT_OApos_raw.edf', preload=True)
raw_data = raw.get_data()

ica = mne.preprocessing.ICA(n_components=20, random_state=0)
bad_idx, scores = ica.find_bads_ecg(raw, 'EEG Fp1 - Pz', threshold=1.5)
print(bad_idx)
ica.fit(raw.copy().filter(8,30))
ica.plot_components(outlines='skirt')

fig, axes = plt.subplots(1, 2)
ax = axes[0]
raw.plot_psd(
    average=False, line_alpha=0.6, fmin=0, fmax=100, xscale='log',
    spatial_colors=True, show=False, ax=[ax])
ax.set(xlabel='Frequency (Hz)', title='Raw')
plt.show()


epochs = mne.make_fixed_length_epochs(raw, duration=15, preload=False)
event_related_plot = epochs.plot_image(picks=['EEG C3 - Pz'])

print('oi')

"""
=========================
Complex Morlet Wavelet
=========================
"""

fs = 300
w = .5
freq = np.linspace(1, fs/4, 30)
widths = w*fs / (2*freq*np.pi)
cwtm = signal.cwt(raw_data[0], signal.morlet2, widths, w=w)
plt.pcolormesh(np.abs(cwtm), cmap='viridis', shading='gouraud')
plt.show()
#Aplicar uma transformada wavelet (usando a complex Morlet wavelet) para obtenção de time-frequency plots
"""
=========================
STFT - Short Time Fourier Transform
=========================
"""
f, t, Zxx = signal.stft(raw_data[0], fs= 300, nperseg=300)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()

# from scipy.signal import spectrogram
#
# ff, tt, Sxx = spectrogram(raw_data[10], fs=300, nperseg=512)
# plt.pcolormesh(tt, ff, Sxx, shading='gouraud')
# plt.title('title')
# plt.xlabel('t (sec)')
# plt.ylabel('Frequency (Hz)')
# plt.grid()
# plt.colorbar(format="%+2.0f dB")
# plt.show()
print(raw)
print(raw.info)
raw_data = raw.get_data()
raw.plot(duration=10, n_channels=25, scalings='auto')
# Some console prints to make sure the data is correctly loaded
print("Channels names:", raw.ch_names)
print("Raw_data' shape: ", str(raw_data.shape))
print("Number of channels: ", str(len(raw_data)))
print("Number of samples: ", str(len(raw_data[0])))

electrodes = ['EEG P3 - Pz', 'EEG C3 - Pz', 'EEG F3 - Pz', 'EEG Fz - Pz', 'EEG F4 - Pz', 'EEG C4 - Pz', 'EEG P4 - Pz',
              'EEG Cz - Pz', 'EEG CM - Pz', 'EEG A1 - Pz', 'EEG Fp1 - Pz', 'EEG Fp2 - Pz', 'EEG T3 - Pz', 'EEG T5 - Pz',
              'EEG O1 - Pz', 'EEG O2 - Pz', 'EEG F7 - Pz', 'EEG F8 - Pz', 'EEG A2 - Pz', 'EEG T6 - Pz', 'EEG T4 - Pz']

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')

# raw_obj.plot(duration=5, n_channels=1)
#filtro notch para suavisar ruido da rede eletrica e suas harmonicas
raw.notch_filter(np.arange(60, 120, 60), filter_length='auto',
                 phase='zero')
raw.plot_psd(area_mode='range', tmax=10.0, average=False)

# filtro pass-banda entre 0.1 e 100Hz
raw.filter(.1, 100., fir_design='firwin')
raw.plot_psd(area_mode='range', tmax=10.0, average=False)# ***********

# raw.plot()
raw.info['bads'].extend(['EEG CM - Pz', 'EEG X3 - Pz', 'EEG X2 - Pz', 'EEG X1 - Pz', 'Trigger'])  # add a list of channels
ica = ICA(n_components=20, method='fastica', random_state=23, max_iter=800).fit(raw)
ica.exclude = [1]
raw_clean = ica.apply(raw.copy())

"""
=========================
PSD (linear vs log scale)
=========================
The Power Spectral Density (PSD) plot shows different information for
linear vs. log scale.
"""
fig, axes = plt.subplots(1, 2)
ax = axes[0]
raw.plot_psd(
    average=False, line_alpha=0.6, fmin=0, fmax=100, xscale='log',
    spatial_colors=True, show=False, ax=[ax])
ax.set(xlabel='Frequency (Hz)', title='Raw')

ax = axes[1]
raw_clean.plot_psd(
    average=False, line_alpha=0.6, fmin=0, fmax=100, xscale='log',
    spatial_colors=False, show=False, ax=[ax])
ax.set(xlabel='Frequency (Hz)', title='Clean')
plt.show()
# """
# Using a linear frequency-axis scaling,we can convince ourselves easily that the data is unfiltered,
# as it contains clear peak from power line at 60 Hz
# """
fig, axes = plt.subplots(1, 2)
ax = axes[0]
raw.plot_psd(average=False, line_alpha=0.6, n_fft=2048, n_overlap=1024, fmin=0,
    fmax=100, xscale='linear', spatial_colors=False, show=False, ax=[ax])
ax.set(xlabel='Frequency (Hz)', ylabel='', title='Raw')
ax.axvline(60., linestyle='--', alpha=0.25, linewidth=2)
ax.axvline(60., linestyle='--', alpha=0.25, linewidth=2)

ax = axes[1]
raw_clean.plot_psd(average=False, line_alpha=0.6, n_fft=2048, n_overlap=1024, fmin=0,
    fmax=100, xscale='linear', spatial_colors=False, show=False, ax=[ax])
ax.set(xlabel='Frequency (Hz)', ylabel='', title='Clean')
ax.axvline(60., linestyle='--', alpha=0.25, linewidth=2)
ax.axvline(60., linestyle='--', alpha=0.25, linewidth=2)

plt.show()


print('Hi, PyCharm')
#
