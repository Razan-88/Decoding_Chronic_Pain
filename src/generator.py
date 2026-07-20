import numpy as np


def make_spike_template(fs=20000, duration_ms=2.0,
                        trough_pos=0.4, trough_width=0.12,
                        peak_pos=0.6, peak_width=0.25, peak_ratio=0.4):
    """Asymmetric extracellular AP template (difference of gaussians). Trough normalised to -1.0."""
    n = int(fs * duration_ms / 1000)
    t = np.linspace(0, 1, n)
    trough = -np.exp(-((t - trough_pos)**2) / (2 * trough_width**2))
    peak = peak_ratio * np.exp(-((t - peak_pos)**2) / (2 * peak_width**2))
    wave = trough + peak
    wave = wave / np.abs(wave.min())
    return wave


def make_channel_map(n_channels=32):
    """Amplitude of the good unit on each channel (index = channel - 1)."""
    amp = np.zeros(n_channels)
    amp[18] = 1.0                                   # ch19
    amp[8], amp[13], amp[24], amp[29] = 0.6, 0.45, 0.7, 0.5   # ch9,14,25,30 (strong)
    amp[2], amp[9], amp[11] = 0.15, 0.1, 0.25       # ch3,10,12 (weak)
    amp[15], amp[16], amp[31] = 0.16, 0.22, 0.24    # ch16,17,32 (weak)
    return amp


def make_recording(fs=20000, n_channels=32, stim_period=4.0, n_pulses=221,
                   latency=0.003, n_dev_spikes=10):
    """
    Build a clean synthetic recording.
    Returns: rec (n_channels x n_samples), amp (channel map), dev_spikes (spike times, s).
    """
    wave = make_spike_template(fs=fs)
    amp = make_channel_map(n_channels)

    ttl_times = np.arange(n_pulses) * stim_period
    spike_times = ttl_times + latency
    dev_spikes = spike_times[:n_dev_spikes]

    duration_s = dev_spikes[-1] + 0.01
    n_samples = int(duration_s * fs)
    rec = np.zeros((n_channels, n_samples))

    n_template = len(wave)
    for t_spike in dev_spikes:
        start = int(t_spike * fs)
        end = start + n_template
        for ch in range(n_channels):
            rec[ch, start:end] += wave * amp[ch]

    return rec, amp, dev_spikes