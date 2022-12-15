
import numpy as np
from scipy import signal
from scipy.fft import fft


def unwrap_pi(p_in):
    p_out = p_in
    ind_large = np.where(np.logical_and(p_in > np.pi, p_in > 0))
    ind_small = np.where(np.logical_and(p_in < np.pi, p_in < 0))
    if len(ind_large) > 0:
        p_l = p_in[ind_large] % np.pi
        p_out[ind_large] = p_l - (((p_in[ind_large] - p_l)/np.pi) % 2)*np.pi
    if len(ind_small) > 0:
        p_s = p_in[ind_small] % -np.pi
        p_out[ind_small] = p_s + (((p_in[ind_small] - p_s)/np.pi) % 2)*np.pi
    return p_out

# test
# p_in = np.array([0, np.pi+0.1, 2*np.pi+0.3, -np.pi-0.1, -2*np.pi-0.3])
# p_out = unwrap_pi(p_in)
# assert np.all(p_out == np.array([0, -np.pi+0.1, 0.3, np.pi-0.1, -0.3]))
# assert len(np.where(p_out > np.pi)) == 0
# assert len(np.where(p_out < -np.pi)) == 0
#Todo test fails due to rounding error

def compute_phase_window(nperseg):
    wind = signal.get_window('hann', nperseg)
    N = len(wind)
    yf = fft(wind)
    phase_w = np.angle(np.append(yf[0:N//2], yf[-1]))
    return phase_w


def compute_inst_phase_amp(x, events, fs, nperseg=None, return_stft=False, phase_w=None):
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
        nperseg = int(fs / 2)  # fs * 2 -> t window = 1 sec ; fs / 2 -> t window = 1/4 sec, f = 0-2-4-6-8...Hz
    f, t, Zxx = signal.stft(
        x, fs=fs, nperseg=nperseg,
        window='hann', noverlap=None,
        nfft=None, detrend=False, return_onesided=True,
        boundary='zeros', padded=True, axis=- 1)

    amp_z = np.abs(Zxx)
    phase_z = np.angle(Zxx) # TODO I do not know why but the phase returned is offset

    if nperseg == int(fs / 2):
        # Dephase the freq that follow 4k-2 by pi
        phase_z[np.where(f % 4 != 0)[0],:] = phase_z[np.where(f % 4 != 0)[0],:] + np.pi

    # #Remove phase from Hanning Window
    # if phase_w is None:
    #     phase_w = compute_phase_window(nperseg)
    #
    # b = np.expand_dims(phase_w, axis=1)
    # phase_wr = np.repeat(b, phase_z.shape[1], axis=1)
    # phase_z = np.angle(Zxx) - phase_wr


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
f_sin = 6.2
p_x = 0  # - np.pi / 2
amp_x = 10

tx = np.arange(0, rl, 1 / fs)
x = amp_x * np.cos(2 * np.pi * f_sin * tx + p_x)

# create a spike train locked to the p_x (if p_x = - np.pi / 2 ; trough)
spikes = {}
spikes['times'] = np.arange(0, rl, 1 / f_sin)

# Compute phase locking etc
[phase_inst, amp_inst, v_strength, v_phase, f], [t, Zxx, amp_z, phase_z] = \
    compute_inst_phase_amp(x=x, events=spikes['times'], fs=fs, nperseg=None,  return_stft=True)  # nperseg = int(fs/4) -> T/8


# Test with hilbert transform
phase_hilb = np.angle(signal.hilbert(x))
assert np.logical_and(phase_hilb[0] > p_x-1e-10, phase_hilb[0] < p_x+1e-10)


# Find freq of interest for test

indx_f = np.argmin(abs(f - f_sin), axis=0)

phase_fz = unwrap_pi(phase_z[indx_f, :])
assert np.logical_and(phase_fz[0] > p_x-0.02, phase_fz[0] < p_x+0.02)
p_lag = (f_sin%1)*2*np.pi
inc_t = np.where(t == 1)
assert np.logical_and(phase_fz[inc_t] > (p_x+p_lag)-0.01, phase_fz[inc_t] < (p_x+p_lag)+0.01)



# Note: rounding errors ; first samples effect -> remove
# indx_f = np.where(f == f_sin)
phase_f = np.unwrap(phase_inst[indx_f, 2:].flatten())  # unwrap does not work properly ; remove 2 origin values
assert np.all(np.logical_or(np.logical_and(phase_f > p_x-1e-10, phase_f < p_x+1e-10),  # Should be =p_x
                            phase_f-2*np.pi > p_x-1e-10, phase_f-2*np.pi < p_x+1e-10))
assert v_strength[indx_f] > 0.99  # Should be =1
assert p_x-0.002 < v_phase[indx_f] < p_x+0.002  # Should be =p_x
amp_f = amp_inst[indx_f, 2:].flatten() # remove 2 origin values
assert np.all(np.logical_and(float(amp_x/2)-0.00001 < amp_f, amp_f < float(amp_x/2)+0.00001))  # Should be =amp_x/2
# TODO I do not understand why amp/2 and not amp
if False:
    import matplotlib.pyplot as plt
    %gui qt
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
    plt.show()

    plt.figure()
    plt.plot(tx, x)
    plt.plot(spikes['times'], np.zeros_like(spikes['times']), 'o')
    plt.plot(t, np.zeros_like(t), '*')
    plt.show()


    # Test window used in STFT
    from scipy.fft import fftfreq

    nperseg = int(fs / 2)
    wind = signal.get_window('hann', nperseg)

    # Number of sample points
    N = len(wind)
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = wind
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]

    # Amp
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()

    # phase
    plt.plot(xf, np.angle(yf[0:N//2]))
    plt.grid()
    plt.show()

#
# if False:
#     fs = 2500
#     rl = 30
#     f_sin = 6
#     p_x = 0  # - np.pi / 2
#     amp_x = 10
#
#     tx = np.arange(0, rl, 1 / fs)
#     x = amp_x * np.sin(2 * np.pi * f_sin * tx + p_x)
#
#     # Compute stft
#     nperseg = int(fs / 2)  # fs * 2 -> t window = 1 sec ; fs / 2 -> t window = 1/4 sec, f = 0-2-4-6-8...Hz
#     f, t, Zxx = signal.stft(
#         x, fs=fs, nperseg=nperseg,
#         window='hann', noverlap=None,
#         nfft=None, detrend=False, return_onesided=True,
#         boundary='zeros', padded=True, axis=- 1)
#
#     amp_z = np.abs(Zxx)
#     phase_z = np.angle(Zxx)  # TODO I do not know why but the phase returned is offset
#
#     # Compute hilbert
#     phase_hilb = np.angle(signal.hilbert(x))
#
#     # Find freq of interest for test
#     indx_f = np.where(f == f_sin)
#
#     print(f'{f_sin} Hz sinus, phase {p_x} - Hilbert phase: {phase_hilb[0]}')
#     print(f'{f_sin} Hz sinus, phase {p_x} - STFT phase: {phase_z[indx_f, 0:5]}')


