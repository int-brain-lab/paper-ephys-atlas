
import numpy as np
from scipy import signal


def compute_inst_phase_amp(x, events, fs, nperseg=None, return_stft=False):
    '''
    Compute the STFT of signal x -> which returns f and t vectors and Zxx matrix. Then:
    Compute the instantaneous phase and amplitude at each time point in events,
    at each frequency contained in f, using the sampling window length in t. Then:
    Compute vector strength
    :param: x: mx1 array of signal (a.u.) [typically LFP signal, units: Volts]
    :param events: nx1 array of time points (seconds) [typically, spike times]
    :param fs: sampling frequency of the signal (Hz)
    :return:

    '''

    ## Compute STFT
    if nperseg is None:
        nperseg = fs / 2  # fs * 2 -> t window = 1 sec ; fs / 2 -> t window = 1/4 sec, f = 0-2-4-6-8...Hz
    f, t, Zxx = signal.stft(
        x, fs=fs, nperseg=nperseg,
        window='hann', noverlap=None,
        nfft=None, detrend=False, return_onesided=True,
        boundary='zeros', padded=True, axis=- 1)

    amp_z = np.abs(Zxx)
    phase_z = np.angle(Zxx) + np.pi/2  # TODO I do not know why but the phase returned is offset

    # Align events

    T_t = t[1]-t[0]  # size window of STFT (seconds)
    indx_phase = (events - (events % T_t))/T_t
    indx_phase = indx_phase.astype(int)
    phase_p = phase_z[:, indx_phase]
    amp_inst = amp_z[:, indx_phase]

    events = np.atleast_2d(events)
    f2d = np.atleast_2d(f)

    phase_lag = np.dot(2*np.pi*f2d.T, (events % T_t))
    phase_inst = np.unwrap(phase_lag+phase_p)

    # Comptue vector strength, cf signal.vectorstrength
    # convert to vectors
    vectors = np.exp(1j * phase_inst)

    # the vector strength is just the magnitude of the mean of the vectors
    # the vector phase is the angle of the mean of the vectors
    vectormean = np.mean(vectors, axis=1)
    v_strength = abs(vectormean)
    v_phase = np.angle(vectormean)

    if return_stft:
        return [phase_inst, amp_inst, v_strength, v_phase, f], [t, Zxx, amp_z, phase_z]
    else:
        return [amp_inst, phase_inst, v_strength, v_phase, f]

## TEST

"""
From OW 
https://github.com/int-brain-lab/ibldevtools/blob/210fdc2591d684f7e3aa95335cd74ae7597dda42/olivier/2022-12-09_PrecessionTests.py#L35-L34
Create a phase precession test by generating a sine wave of f_sin Hz and spike trains
locked to the phase p_x
"""
# create a sine wave of f_sin  Hz with p_x phase delay
fs = 2500
rl = 30
f_sin = 4
p_x = 0  # - np.pi / 2
amp_x = 6

tx = np.arange(0, rl, 1 / fs)
x = amp_x * np.sin(2 * np.pi * f_sin * tx + p_x)

# create a spike train locked to the p_x (if p_x = - np.pi / 2 ; trough)
spikes = {}
spikes['times'] = np.arange(0, rl, 1 / f_sin)

# Compute phase locking etc
[phase_inst, amp_inst, v_strength, v_phase, f], [t, Zxx, amp_z, phase_z] = \
    compute_inst_phase_amp(x=x, events=spikes['times'], fs=fs, nperseg=None,  return_stft=True)

# Find freq of interest for test
# Note: rounding errors ; first sample effect -> remove
indx_f = np.where(f == f_sin)
phase_f = phase_inst[indx_f, 1:].flatten()
assert np.all(np.logical_or(np.unwrap(phase_f) < 1e-10 + p_x,  # unwrap does not work properly
                            np.unwrap(phase_f)-2*np.pi < 1e-10 + p_x))  # Should be =p_x
assert v_strength[indx_f] > 0.99  # Should be =1
assert p_x-0.002 < v_phase[indx_f] < p_x+0.002  # Should be =p_x
amp_f = amp_inst[indx_f, 1:].flatten()
assert np.all(np.logical_and(float(amp_x/2)-0.00001 < amp_f, amp_f < float(amp_x/2)+0.00001))  # Should be =amp_x/2

if False:
    import matplotlib.pyplot as plt
    ## Plots amplitude / phase
    plt.pcolormesh(t, f, amp_z, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    plt.pcolormesh(t, f, phase_z, vmin=-np.pi, vmax=np.pi, shading='gouraud')
    plt.title('STFT Phase')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.plot(t, np.zeros_like(t), '*')
    plt.show()

    plt.figure()
    plt.plot(tx, x)
    plt.plot(spikes['times'], np.zeros_like(spikes['times']), 'o')
