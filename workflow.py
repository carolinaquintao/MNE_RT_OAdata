import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_connectivity import envelope_correlation


# """""""""""""
# Carrega o dado
# """"""""""""""""
raw = mne.io.read_raw_edf(
    'C:/Users/ufpa/OneDrive - Universidade Federal do Pará - '
    'UFPA/Naynara/BaseDeDadosITV/suj1_36_B/suj1_36_28012019_RT_OA_raw.edf',
    preload=True)
# raw_data = raw.get_data() #se quiser pegar só a matriz de valores

# """""""""""""
# Para renomear os canais e poder usar a montagem do sistema 10-20
# """"""""""""""""
print('Renaming channels')
electrodes = ['EEG P3 - Pz', 'EEG C3 - Pz', 'EEG F3 - Pz', 'EEG Fz - Pz', 'EEG F4 - Pz', 'EEG C4 - Pz', 'EEG P4 - Pz',
              'EEG Cz - Pz', 'EEG CM - Pz', 'EEG A1 - Pz', 'EEG Fp1 - Pz', 'EEG Fp2 - Pz', 'EEG T3 - Pz', 'EEG T5 - Pz',
              'EEG O1 - Pz', 'EEG O2 - Pz', 'EEG X3 - Pz', 'EEG X2 - Pz', 'EEG F7 - Pz', 'EEG F8 - Pz', 'EEG X1 - Pz',
              'EEG A2 - Pz', 'EEG T6 - Pz', 'EEG T4 - Pz', 'Trigger']
values = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'X3',
          'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'Trigger']
electrodesN = dict(zip(electrodes, values))
mne.rename_channels(raw.info, electrodesN)

# """""""""""""
# Adiciona a montagem do sistema 10-20
# """"""""""""""""
print('Setting montage 10-20')
montage = mne.channels.make_standard_montage('standard_1020')
raw.info.set_montage(montage, on_missing='ignore')

# """""""""""""
# Remove canais inúteis
# """"""""""""""""
print('Removing channels: CM, X3, X2, X1, Trigger')
raw.drop_channels(['CM', 'X3', 'X2', 'X1', 'Trigger'])

# """""""""""""
#  filtro pass-banda entre 0.1 e 100Hz
# """"""""""""""""
raw.filter(.2, 100., fir_design='firwin')
# raw.plot_psd(area_mode='range', tmax=10.0, average=False)# ***********

raw.notch_filter(np.arange(60, 121, 60), filter_length='auto',
                 phase='zero')
# raw.plot_psd(area_mode='range', tmax=10.0, average=False)

# """""""""""""
# Procede o ICA com referência ao canal 'Fp1' e 'Fp2' (exemplo). Aqui poderia ser feito com relação a um canal de
# EOG (find_bads_eog) ou ECG find_bads_ecg
# """"""""""""""""
print('Proceeding ICA')
ica = ICA(n_components=20, method='fastica', random_state=23).fit(raw)
bad_idx, scores = ica.find_bads_eog(raw, ['Fp1', 'Fp2'], threshold=2)
# barplot of ICA component "EOG match" scores
# ica.plot_sources(raw, picks=bad_idx[0])
# # plot diagnostics
# ica.plot_properties(raw, picks=bad_idx)# plot ICs applied to raw data, with EOG matches highlighted
# ica.plot_sources(raw)
# ica.plot_components()
raw_clean = ica.apply(raw.copy())

# """""""""""""
#  Use average of mastoid channels as reference
# """"""""""""""""
raw_clean_avg_ref = raw_clean.copy().set_eeg_reference(ref_channels=['A1', 'A2'])

# """""""""""""
#  Divide em épocas de 1 segundo
# """"""""""""""""
epochs = mne.make_fixed_length_epochs(raw_clean_avg_ref, duration=1, preload=False)

# """""""""""""
#  Filtra nas bandas de interesse
# """"""""""""""""
# para ver as bandas seta i=1 ou tira do if ^^
i = 0
if i:
    # DELTA está relacionada ao sono profundo. Entrando, está associada a patologias e é encontrada normalmente no
    # córtex temporal
    delta_data = epochs.load_data().filter(l_freq=0.5, h_freq=3).get_data()

    # TETA é observada principalmente no momento de emoções intensas e meditação profunda, normalmente associadas às
    # atividades do hipocampo
    theta_data = epochs.load_data().filter(l_freq=4, h_freq=7).get_data()
    lowTheta_data = epochs.load_data().filter(l_freq=4, h_freq=5.5).get_data()
    highTheta_data = epochs.load_data().filter(l_freq=5.5, h_freq=7).get_data()

    # ALFA ocorre quando estamos com os olhos fechados relaxando ou meditando. Pode ser registrada no córtex visual.
    alpha_data = epochs.load_data().filter(l_freq=8, h_freq=12).get_data()

    # BETA são registradas no córtex sensoriomotor e está associada principalmente ao movimento de membros inferiores.
    beta_data = epochs.load_data().filter(l_freq=12, h_freq=30).get_data()
    lowBeta_data = epochs.load_data().filter(l_freq=12, h_freq=18).get_data()
    highBeta_data = epochs.load_data().filter(l_freq=18, h_freq=30).get_data()

    # GAMMA está relacionada a concentração
    gamma_data = epochs.load_data().filter(l_freq=31, h_freq=100).get_data()

    # MU é um espectro da onda alfa que se localiza no córtex sensoriomotor e está relacionado ao movimento de
    # membros superiores
    mu_data = epochs.load_data().filter(l_freq=7, h_freq=12).get_data(picks=['Cz', 'C3', 'C4'])
    smr_data = epochs.load_data().filter(l_freq=13, h_freq=15).get_data(picks=['Cz', 'C3', 'C4'])  # centrais


# """""""""""""
#  Se quiser plotar a média dos valores em cada banda, seta i=1 ou tira do if ^^
# """"""""""""""""
i = 0
if i:
    values = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'F7', 'F8', 'X1',
              'T6', 'T4']
    delta_data = epochs.load_data().filter(l_freq=0.5, h_freq=3).average().plot(spatial_colors=True, picks=values)


# """""""""""""
#  Conectividade em uma banda específica no primeiro e no último segundo
# """"""""""""""""
i = 0
if i:
    delta_data = epochs.load_data().filter(l_freq=0.5, h_freq=3).get_data()

    corr_matrix = envelope_correlation(delta_data).get_data()
    print(corr_matrix.shape)

    first_sec = corr_matrix[0]
    last_sec = corr_matrix[-1]
    corr_matrices = [first_sec, last_sec]
    color_lims = np.percentile(np.array(corr_matrices), [5, 95])
    titles = ['First Second', 'Last Second']

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Correlation Matrices from First 30 Seconds and Last 30 Seconds')
    for ci, corr_matrix in enumerate(corr_matrices):
        ax = axes[ci]
        mpbl = ax.imshow(corr_matrix, clim=color_lims)
        ax.set_xlabel(titles[ci])
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.2, 0.025, 0.6])
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cbar.set_label('Correlation Coefficient')
    plt.show()

# """""""""""""
#  PSD em uma banda específica
# """"""""""""""""
from mne.time_frequency import psd_multitaper

psds, freqs = psd_multitaper(epochs.load_data(), low_bias=True,
                             fmin=0.5, fmax=30, proj=True,
                             n_jobs=1)


print('----FIM----')
