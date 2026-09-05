import numpy as np
import h5py
import torch
import torch.nn as nn
import pywt
from scipy.signal import find_peaks
from scipy.signal import find_peaks, correlate
from scipy.ndimage import uniform_filter1d
from sklearn.svm import OneClassSVM, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def build_template_from_annotations(signal, spike_times, reference_start, half_width=45, align_window=30, sample_frequency=30000):
    segments = []
    for spike_time in spike_times:
        centre_index = int(round((spike_time - reference_start) * sample_frequency))
        lo = centre_index - half_width - align_window
        hi = centre_index + half_width + align_window
        if lo < 0 or hi > len(signal):
            continue
        wide = signal[lo:hi]
        centre = half_width + align_window
        local = wide[centre-align_window:centre+align_window]
        trough_index = np.argmin(local) + (centre-align_window)
        segment = wide[trough_index-half_width:trough_index+half_width]
        if len(segment) == 2 * half_width:
            segments.append(segment)
    if not segments:
        raise ValueError("No complete annotated waveforms available.")
    return np.mean(np.asarray(segments), axis=0)

def post_stimulus_latencies(spike_times, stimuli):
    index = np.searchsorted(stimuli, spike_times, side='right') - 1
    valid = index >= 0
    return (spike_times[valid] - stimuli[index[valid]]) * 1000

def label_survivors(survivor_times, annotations, tolerance=0.002):
    return np.array([np.any(np.abs(annotations - t) <= tolerance) for t in survivor_times])



def extract_waveforms(survivor_times, signal, window_start, half_width=45, align_window=30, sample_frequency=30000):
    waveforms = []
    keep = []
    for t in survivor_times:
        centre_index = int(round((t - window_start) * sample_frequency))
        lo, hi = centre_index - half_width - align_window, centre_index + half_width + align_window
        if lo < 0 or hi > len(signal):
            keep.append(False)
            continue
        wide = signal[lo:hi]
        centre = half_width + align_window
        trough = np.argmin(wide[centre-align_window:centre+align_window]) + (centre-align_window)
        segment = wide[trough-half_width:trough+half_width]
        if len(segment) == 2*half_width:
            waveforms.append(segment)
            keep.append(True)
        else:
            keep.append(False)
    return np.array(waveforms), np.array(keep)
    
def oneclass_dev_eval(dev_waveforms, dev_labels, eval_waveforms, eval_labels, nu=0.1):
    dev_targets = dev_waveforms[dev_labels]
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(dev_targets)
    eval_scores = svm.decision_function(eval_waveforms)
    eval_preds = (eval_scores > 0).astype(int)
    return eval_labels.astype(int), eval_scores, eval_preds


def twoclass_dev_eval(dev_waveforms, dev_labels, eval_waveforms, eval_labels, C=1.0):
    scaler = StandardScaler().fit(dev_waveforms)
    svm = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced")
    svm.fit(scaler.transform(dev_waveforms), dev_labels.astype(int))
    eval_scores = svm.decision_function(scaler.transform(eval_waveforms))
    eval_preds = svm.predict(scaler.transform(eval_waveforms))
    return eval_labels.astype(int), eval_scores, eval_preds
    
def bootstrap_ci_stratified(labels, scores, preds=None, n_boot=2000, seed=0, alpha=0.05):
    labels = np.asarray(labels); scores = np.asarray(scores)
    preds = (scores > 0).astype(int) if preds is None else np.asarray(preds)
    target_index = np.where(labels == 1)[0]
    nontarget_index = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    keys = ["precision", "recall", "f1", "roc_auc", "pr_ap"]
    bootstrap_values = {metric_name: [] for metric_name in keys}
    for _ in range(n_boot):
        target_resample = rng.choice(target_index, len(target_index), replace=True)
        nontarget_resample = rng.choice(nontarget_index, len(nontarget_index), replace=True)
        index = np.concatenate([target_resample, nontarget_resample])
        if len(np.unique(labels[index])) < 2:
            continue
        m = _metrics(labels[index], preds[index], scores[index])
        for metric_name in keys:
            bootstrap_values[metric_name].append(m[metric_name])
    point = _metrics(labels, preds, scores)
    out = {}
    for metric_name in keys:
        array = np.asarray(bootstrap_values[metric_name], dtype=float)
        lo, hi = np.percentile(array, [100*alpha/2, 100*(1-alpha/2)])
        out[metric_name] = (point[metric_name], float(lo), float(hi))
    return out
    
def _metrics(labels, preds, scores):
    """Compute precision, recall, F1, ROC-AUC, PR-AP on one set of candidates."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    scores = np.asarray(scores)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    both_classes = len(np.unique(labels)) == 2
    roc = roc_auc_score(labels, scores) if both_classes else np.nan
    pr_ap = average_precision_score(labels, scores) if both_classes else np.nan
    return dict(precision=precision, recall=recall, f1=f1, roc_auc=roc, pr_ap=pr_ap)


def make_training_pairs_real(template, n=40000, window=60, jitter=8, spike_fraction=0.5, seed=0):
    """Generate (noisy, clean) training pairs for the denoisers from a spike template."""
    random_gen = np.random.default_rng(seed)
    spike_shape = template.copy()
    original_position = np.linspace(0, 1, len(spike_shape))
    new_position = np.linspace(0, 1, window)
    spike_shape = np.interp(new_position, original_position, spike_shape)
    spike_shape = spike_shape / np.abs(spike_shape.min())
    clean = np.zeros((n, window))
    for i in range(n):
        if random_gen.random() < spike_fraction:
            spike_shift = random_gen.integers(-jitter, jitter + 1)
            spike_amplitude = random_gen.uniform(0.5, 3.0)
            clean[i] = np.roll(spike_shape, spike_shift) * spike_amplitude
    noise_std = random_gen.uniform(0.1, 0.6, size=(n, 1))
    noisy = clean + random_gen.normal(0, 1, size=clean.shape) * noise_std
    return (torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32))


def ae_denoise_signal(channel, denoiser, window=60, step=20):
    """Apply a trained dense autoencoder across a long signal with overlapping windows."""
    denoiser.eval()
    denoised_output = np.zeros(len(channel))
    overlap_count = np.zeros(len(channel))
    noise_scale = np.median(np.abs(channel)) / 0.6745
    with torch.no_grad():
        for start in range(0, len(channel) - window, step):
            segment = channel[start:start+window]
            if segment.std() < 1e-6:
                continue
            segment_normalised = segment / noise_scale
            reconstruction = denoiser(torch.tensor(segment_normalised, dtype=torch.float32)).numpy()
            denoised_output[start:start+window] += reconstruction * noise_scale
            overlap_count[start:start+window] += 1
    overlap_count[overlap_count == 0] = 1
    return denoised_output / overlap_count


def conv_ae_denoise(channel, denoiser, window=60, step=20):
    """Apply a trained conv autoencoder across a long signal with overlapping windows."""
    denoiser.eval()
    denoised_output = np.zeros(len(channel))
    overlap_count = np.zeros(len(channel))
    noise_scale = np.median(np.abs(channel)) / 0.6745
    with torch.no_grad():
        for start in range(0, len(channel) - window, step):
            segment = channel[start:start+window]
            if segment.std() < 1e-6:
                continue
            model_input = torch.tensor(segment / noise_scale, dtype=torch.float32).view(1, 1, window)
            reconstruction = denoiser(model_input).view(-1).numpy()
            denoised_output[start:start+window] += reconstruction * noise_scale
            overlap_count[start:start+window] += 1
    overlap_count[overlap_count == 0] = 1
    return denoised_output / overlap_count


def denoise_swt_channel(channel, wavelet='db4', level=4):
    """Stationary wavelet transform denoising: SWT, soft-threshold details, inverse SWT."""
    signal_length = len(channel)
    pad = (-signal_length) % (2 ** level)
    if pad:
        channel = np.pad(channel, (0, pad), mode='reflect')
    coeffs = pywt.swt(channel, wavelet, level=level)
    denoised_coeffs = []
    for cA, cD in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(cD)))
        cD_thresh = pywt.threshold(cD, threshold, mode='soft')
        denoised_coeffs.append((cA, cD_thresh))
    denoised = pywt.iswt(denoised_coeffs, wavelet)
    return denoised[:signal_length]

class DenseDAE(nn.Module):
    """
    Dense denoising autoencoder for one window of signal. Compresses a 60-sample
    window to a 15-unit bottleneck (encoder), then reconstructs it back to 60
    samples (decoder). The bottleneck forces the network to keep the spike shape
    and drop the noise. Encoder 60->30->15, decoder 15->30->60.
    """
    def __init__(self, window=60, hidden=15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, 30), nn.ReLU(),
            nn.Linear(30, hidden), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 30), nn.ReLU(),
            nn.Linear(30, window),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class ConvDAE(nn.Module):
    """
    Convolutional denoising autoencoder for one window of signal. Uses 1D
    convolutions instead of linear layers; a convolution slides a small filter
    (kernel=3) across the window, detecting local spike-shaped patterns wherever
    they occur. Encoder 60->30->15, decoder reverses back to 60 samples.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv1d(16, 1, 3, stride=1, padding=1),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))    


def template_match(signal, template):
    """Normalised cross-correlation: slide the zero-mean, unit-norm template
    along the signal, return a match score in [-1, 1] at each sample."""
    template_centred = template - template.mean()
    template_unit = template_centred / np.linalg.norm(template_centred)
    template_length = len(template_unit)
    raw_match = correlate(signal, template_unit, mode='same')
    local_mean = uniform_filter1d(signal, template_length, mode='nearest')
    local_mean_sqr = uniform_filter1d(signal**2, template_length, mode='nearest')
    local_variance = np.maximum(local_mean_sqr - local_mean**2, 0)
    local_norm = np.sqrt(local_variance * template_length)
    return np.divide(raw_match, local_norm, out=np.zeros_like(raw_match), where=local_norm > 1e-6)


def in_latency_window(times, ttls, lat_min=0.085, lat_max=0.160):
    """True where a detection's latency after the preceding stimulus falls in [lat_min, lat_max]."""
    indx = np.searchsorted(ttls, times) - 1
    has_ttl = indx >= 0
    latency = np.full(len(times), np.inf)
    latency[has_ttl] = times[has_ttl] - ttls[indx[has_ttl]]
    return (latency >= lat_min) & (latency <= lat_max)


def score_detections_counts(detected, ground_truth_times, tolerance=0.002):
    """Score detections against ground truth (one-to-one matching); return tp, fp, fn, precision, recall, f1, fp/tp."""
    matched_true = set()
    tp = 0
    for detection in detected:
        close = np.where(np.abs(ground_truth_times - detection) <= tolerance)[0]
        unmatched = [gt for gt in close if gt not in matched_true]
        if len(unmatched) > 0:
            tp += 1
            matched_true.add(unmatched[0])
    fp = len(detected) - tp
    fn = len(ground_truth_times) - len(matched_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fptp = fp / tp if tp > 0 else float('inf')
    return tp, fp, fn, precision, recall, f1, fptp

def template_match(signal, template):
    """
    Normalised cross-correlation between template and a signal
    Slides the zero-mean, unit-norm template along the signal and returns a match score at every sample. 

    Args:
        signal (np. ndarray): 1D signal to search
        template (np.ndarray): the average spike waveform to match. 

    Return:
        np.ndarray: match score in [-1, 1] at each sample.
    """
    template_centred = template - template.mean() #subtract the average so the template sits centred on zero, with equal weight above and below 
    template_unit = template_centred / np.linalg.norm(template_centred) # divide by its length (the square root of the sum of squared), so the template size 1 
                            # zero means removes the offset and unit norm removes the scale, together they turn a raw dot product into a proper correlation that lives in [-1. 1]
    template_length = len(template_unit) # template length in samples. 
    raw_match = correlate(signal, template_unit, mode='same') # At every position along the signal, slide the template, multiply and sum template against signal at every position, and give one value per sample. 
    
    # Measure the signal's size at every position, how big the signal is in each template width window. 
    local_mean   = uniform_filter1d(signal,    template_length, mode='nearest') # signal's local mean. Slide a window of width along the signal and average the values inside it.                                                   
    local_mean_sqr  = uniform_filter1d(signal**2, template_length, mode='nearest') #local mean of the signal squared
    local_variance = np.maximum(local_mean_sqr - local_mean**2, 0)  #local variance = (mean of squares) - (mean squared) and clipped at 0, so rounding cant make it negatie
    local_norm = np.sqrt(local_variance * template_length) 
    return np.divide(raw_match, local_norm, out=np.zeros_like(raw_match), where=local_norm > 1e-6) # Raw match / signal size = the final shape score [-1, 1]

def _metrics(labels, preds, scores):
    """
    Compute precision, recall, ROC-AUC, PR-AP (classification metrics) on one set of candidates
    """
    labels = np.asarray(labels) # is an array like for the true label for each candidate so (1 = target, 0 = non-target)
    preds = np.asarray(preds) # is an array like for the model's predicted label for each candidate. 
    scores = np.asarray(scores) # the model's decision score for each candidate, so higher = more target like which is used for ROC-AUC and PR-AP
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    # compute ROC-AUC and PR-AP
    both_classes = len(np.unique(labels)) == 2 
    roc = roc_auc_score(labels, scores) if both_classes else np.nan
    pr_ap = average_precision_score(labels, scores) if both_classes else np.nan
    return dict(precision=precision, recall=recall, f1=f1, roc_auc=roc, pr_ap=pr_ap)
def load_channel(channel_number, path, window_start, window_seconds=1800, sample_frequency=30000):
    """Load a channel's signal and annotations for a window of window_seconds."""
    signal_file = path + f"/100_RhythmData_CH{channel_number}_2_filtered_blanked.bin"
    start_indx = int(window_start * sample_frequency)
    n_samples = int(window_seconds * sample_frequency)
    signal = np.fromfile(signal_file, dtype='<f8', count=n_samples, offset=start_indx * 8)
    window_end = window_start + len(signal) / sample_frequency
    with h5py.File(path + "/grouped_unit_1.mat", "r") as f:
        zc = f[f"CH{channel_number}"]["aligned_zc_locs"][:].flatten()
    annotations = zc[(zc >= window_start) & (zc <= window_end)]
    return signal, annotations, window_end

