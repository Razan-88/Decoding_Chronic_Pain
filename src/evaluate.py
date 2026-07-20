import numpy as np


def rmse(denoised, clean):
    return np.sqrt(np.mean((denoised - clean) ** 2))


def spike_amplitude(data, spike_times, fs, n_template=40):
    troughs = []
    for t_spike in spike_times:
        start = int(t_spike * fs)
        end = start + n_template
        troughs.append(data[start:end].min())
    return np.mean(troughs)


def channel_profile(data, spike_times, fs, n_channels, n_template=40):
    return np.array([spike_amplitude(data[ch], spike_times, fs, n_template)
                     for ch in range(n_channels)])


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def fidelity(denoised, clean, spike_times, fs, n_channels):
    p_clean = channel_profile(clean, spike_times, fs, n_channels)
    p_denoised = channel_profile(denoised, spike_times, fs, n_channels)
    return cosine_similarity(p_clean, p_denoised)


def snr_db(signal_amp, noise_std):
    return 20 * np.log10(np.abs(signal_amp) / noise_std)


def detect_spikes(data, fs, k=4, noise_std=None):
    if noise_std is None:
        noise_std = data.std()
    threshold = -k * noise_std
    below = data < threshold
    crossings = np.where(np.diff(below.astype(int)) == 1)[0] + 1
    return crossings / fs


def score_detections(detected, true_times, tolerance=0.001):
    matched_true = set()
    tp = 0
    for d in detected:
        close = np.where(np.abs(true_times - d) <= tolerance)[0]
        unmatched = [i for i in close if i not in matched_true]
        if len(unmatched) > 0:
            tp += 1
            matched_true.add(unmatched[0])
    fp = len(detected) - tp
    fn = len(true_times) - len(matched_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1