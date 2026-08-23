import numpy as np
import pywt
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def denoise_bandpass(noisy, fs, lowcut=300, highcut=3000, order=4):
    """Per-channel Butterworth bandpass, zero-phase (filtfilt)."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, noisy, axis=1)


def denoise_cmr(noisy, fs=None):
    """Common Median Reference: subtract the across-channel median at each time point."""
    reference = np.median(noisy, axis=0)
    return noisy - reference


def denoise_car(noisy, fs=None):
    """Common Average Reference: subtract the across-channel mean. (Not robust to spikes.)"""
    reference = np.mean(noisy, axis=0)
    return noisy - reference


def denoise_wavelet_channel(x, wavelet='db4', level=4, thresh_scale=1.0):
    """DWT denoise one channel: soft-threshold detail coeffs, reconstruct."""
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = thresh_scale * sigma * np.sqrt(2 * np.log(n))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(new_coeffs, wavelet)[:n]


def denoise_wavelet(noisy, fs=None, wavelet='db4', level=4, thresh_scale=1.0):
    """DWT denoise all channels (per-channel)."""
    out = np.zeros_like(noisy)
    for ch in range(noisy.shape[0]):
        out[ch] = denoise_wavelet_channel(noisy[ch], wavelet, level, thresh_scale)
    return out