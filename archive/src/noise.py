import numpy as np


def make_pink_noise(n_samples, fs):
    """Pink (1/f) noise via FFT: white noise shaped so power falls as 1/f."""
    white = np.random.normal(0, 1, n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1 / np.sqrt(freqs[1:])
    spectrum_pink = spectrum * scaling
    pink = np.fft.irfft(spectrum_pink, n=n_samples)
    return pink / np.std(pink)


def make_mains(n_samples, fs, mains_amp=0.05, base_freq=50, n_harmonics=4):
    """50 Hz mains + harmonics (100, 150, 200 Hz), amplitude 1/n per harmonic."""
    t = np.arange(n_samples) / fs
    mains = np.zeros(n_samples)
    for n in range(1, n_harmonics + 1):
        freq = base_freq * n
        amp = mains_amp / n
        mains += amp * np.sin(2 * np.pi * freq * t)
    return mains


def make_noise(n_channels, n_samples, fs, white_std=0.1, pink_std=0.1,
               mains_amp=0.05, common_frac=0.5):
    """Layered noise: white + pink + mains, with common_frac shared across channels."""
    noise = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        w = white_std * np.random.normal(0, 1, n_samples)
        p = pink_std * make_pink_noise(n_samples, fs)
        noise[ch] = (1 - common_frac) * (w + p)
    w_common = white_std * np.random.normal(0, 1, n_samples)
    p_common = pink_std * make_pink_noise(n_samples, fs)
    mains = make_mains(n_samples, fs, mains_amp)
    shared = w_common + p_common + mains
    for ch in range(n_channels):
        noise[ch] += common_frac * shared
    return noise