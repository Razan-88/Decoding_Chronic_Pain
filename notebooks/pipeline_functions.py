#Pipeline functions 
#leakage safe spike detection and classification on single channels
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

#1. Load the channels

def load_channel(channel_number, path, window_start, window_seconds=1800, sample_frequency=30000):
    """
    Load a channel's signal and its reference annotations for a window of window_seconds.
    Returns the signal, the annotated spike times in the window, and the window end time
    
    """
    #read the channel's signal file, starting at the window_start
    signal_file = path + f"/100_RhythmData_CH{channel_number}_2_filtered_blanked.bin"
    start_indx = int(window_start * sample_frequency) #window start -> sample index
    n_samples = int(window_seconds * sample_frequency) #duration -> number of samples
    signal = np.fromfile(signal_file, dtype='<f8', count=n_samples, offset=start_indx * 8) #*8: float64 =8 bytes
    window_end = window_start + len(signal) / sample_frequency
    #read the annotated spike times for this channel and only keep those in the window 
    with h5py.File(path + "/grouped_unit_1.mat", "r") as f:
        zc = f[f"CH{channel_number}"]["aligned_zc_locs"][:].flatten() #this channel's spike times
    annotations = zc[(zc >= window_start) & (zc <= window_end)] #pnly keep those in window
    return signal, annotations, window_end

#2. template

def build_template_from_annotations(signal, spike_times, reference_start, half_width=45, align_window=30, sample_frequency=30000):
    """
    Build the spike template by averaging the annotated spike waveform 
    each spike is cut out, aligned to its trough, and the aligned waveforms are averaged.
    
    """
    segments = []
    for spike_time in spike_times:
        centre_index = int(round((spike_time - reference_start) * sample_frequency)) #spike time -> sample index
        lo = centre_index - half_width - align_window #cut a wide window to find the trough
        hi = centre_index + half_width + align_window
        if lo < 0 or hi > len(signal):
            continue #skip spikes too close to the signal edge
        wide = signal[lo:hi]
        centre = half_width + align_window
        local = wide[centre-align_window:centre+align_window]# small region around the marked spike time
        trough_index = np.argmin(local) + (centre-align_window) #find the deepest point (the trough)
        segment = wide[trough_index-half_width:trough_index+half_width] #cut 90 samples centred on the trough
        if len(segment) == 2 * half_width:
            segments.append(segment)
    if not segments:
        raise ValueError("No complete annotated waveforms available.")
    return np.mean(np.asarray(segments), axis=0) #the template = average of all aligned spikes

#3. latency

def post_stimulus_latencies(spike_times, stimuli):
    """
    Latency in ms of each spike relative to the most recent preceding stimulus
    """
    
    index = np.searchsorted(stimuli, spike_times, side='right') - 1 #the preceding stimulus for each spike
    valid = index >= 0 #keep the spikes that have a preceding stimulus
    return (spike_times[valid] - stimuli[index[valid]]) * 1000 #time in ms since that sitimulus 

#4. denoiser training data
def make_training_pairs_real(template, n=40000, window=60, jitter=8, spike_fraction=0.5, seed=0):
    """
    Generate (noisy, clean) training pairs for the autoencoders from a spike template.
    Half the windows contain a shifted/scaled template; the other half are noise-only; noise is added to create the noisy input.
    """
    random_gen = np.random.default_rng(seed)
    spike_shape = template.copy()
    original_position = np.linspace(0, 1, len(spike_shape)) #resample the template
    new_position = np.linspace(0, 1, window) #to the window length with is 60
    spike_shape = np.interp(new_position, original_position, spike_shape)
    spike_shape = spike_shape / np.abs(spike_shape.min())#normalise so the trough is -1
    clean = np.zeros((n, window)) # the clean targets (ctart as zeros)
    for i in range(n):
        if random_gen.random() < spike_fraction: #this window gets a spike 
            spike_shift = random_gen.integers(-jitter, jitter + 1) #random position shift
            spike_amplitude = random_gen.uniform(0.5, 3.0) #random size
            clean[i] = np.roll(spike_shape, spike_shift) * spike_amplitude
    noise_std = random_gen.uniform(0.1, 0.6, size=(n, 1)) #random noise level per window
    noisy = clean + random_gen.normal(0, 1, size=clean.shape) * noise_std #add noise to make the input
    return (torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32))

#5.Denoiser models
class DenseDAE(nn.Module):
    """
    Dense denoising autoencoder for one window of signal. Compresses a 60-sample
    window to a 15-unit bottleneck (encoder so 60 -> 30 -> 15), then reconstructs it back to 60
    samples (decoder so 15 -> 30 -> 60) using fully connected layers. The bottleneck forces the network to keep the spike shape
    and drop the noise.
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
    Convolutional denoising autoencoder uses the same compression and reconstruction but with 1D convolutional (kernel =3) that detects
    local spike-shaped patterns wherever they occur. Encoder 60 -> 30 -> 15 and decoder 15 -> 30 -> 60. 
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
        

# Applting denoisers 
def ae_denoise_signal(channel, denoiser, window=60, step=20):
    """
    Apply a trained dense autoencoder across the whole long signal with overlapping windows.
    Each window is normalised by the noise level, denoised, rescaled back and overlaps are averaged. 
    """
    denoiser.eval()
    denoised_output = np.zeros(len(channel))
    overlap_count = np.zeros(len(channel))
    noise_scale = np.median(np.abs(channel)) / 0.6745 #MAD noise level for the normalising
    with torch.no_grad():
        for start in range(0, len(channel) - window, step): #slide a window 
            segment = channel[start:start+window]
            if segment.std() < 1e-6:
                continue # skip flat window
            segment_normalised = segment / noise_scale #normalise to match training
            reconstruction = denoiser(torch.tensor(segment_normalised, dtype=torch.float32)).numpy()
            denoised_output[start:start+window] += reconstruction * noise_scale #rescale back
            overlap_count[start:start+window] += 1
    overlap_count[overlap_count == 0] = 1 #dont divide by zero
    return denoised_output / overlap_count # average the overlaps


def conv_ae_denoise(channel, denoiser, window=60, step=20):
    """
    Apply a trained conv autoencoder across a long signal with overlapping windows but reshape each window to 1, 1, 60
    """
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
    """
    Stationary wavelet transform denoising, SWT, soft-threshold details coefficients  (remove small noise), reconstruct (inverse SWT)
    """
    signal_length = len(channel)
    pad = (-signal_length) % (2 ** level)
    if pad:
        channel = np.pad(channel, (0, pad), mode='reflect') #pad to a valid length
    coeffs = pywt.swt(channel, wavelet, level=level) #decompose into scales
    denoised_coeffs = []
    for cA, cD in coeffs: #cA means keep approximation, cD detail is noise 
        sigma = np.median(np.abs(cD)) / 0.6745 #noise estimate for this level
        threshold = sigma * np.sqrt(2 * np.log(len(cD))) #universal threshold
        cD_thresh = pywt.threshold(cD, threshold, mode='soft') # zero the small noise coefficients
        denoised_coeffs.append((cA, cD_thresh))
    denoised = pywt.iswt(denoised_coeffs, wavelet) #reconstruct
    return denoised[:signal_length] #trim padding
    
# 6. detection
def template_match(signal, template):
    """
    Normalised cross-correlation; slide the template along the signal; return a shape-match
    score in [-1, 1] at every sample (independent of amplitude)
    """
    template_centred = template - template.mean() #centre the template on zero 
    template_unit = template_centred / np.linalg.norm(template_centred) #unit size and remove scale 
    template_length = len(template_unit) # template length in samples. 
    raw_match = correlate(signal, template_unit, mode='same') # At every position along the signal, slide the template, multiply and sum the template against the signal at every position, and give one value per sample. 
    
    # Measure the signal's size at every position, how big the signal is in each template width window. 
    local_mean   = uniform_filter1d(signal, template_length, mode='nearest') # signal's local mean. Slide a window of width along        the signal and average the values inside it.                                                   
    local_mean_sqr  = uniform_filter1d(signal**2, template_length, mode='nearest') #local mean of the signal squared
    local_variance = np.maximum(local_mean_sqr - local_mean**2, 0)  #local variance = (mean of squares) - (mean squared) and clipped      at 0, so rounding can't make it negative
    local_norm = np.sqrt(local_variance * template_length) #lcal size of the signal
    return np.divide(raw_match, local_norm, out=np.zeros_like(raw_match), where=local_norm > 1e-6) # Raw match/signal size = the final shape score [-1, 1]


def in_latency_window(times, ttls, lat_min=0.085, lat_max=0.160):
    """
    True where a detection's latency after the preceding stimulus falls in [lat_min, lat_max].
    """
    indx = np.searchsorted(ttls, times) - 1 #preceding stimulus for each detection
    has_ttl = indx >= 0
    latency = np.full(len(times), np.inf)
    latency[has_ttl] = times[has_ttl] - ttls[indx[has_ttl]] #time since the stimulus
    return (latency >= lat_min) & (latency <= lat_max) #keep those in the latency window


#7. label and score
def score_detections_counts(detected, ground_truth_times, tolerance=0.002):
    """
    Score detections against ground truth annotation (one-to-one matching within the tolerance); return tp, fp, fn, precision, recall, f1, fp/tp.
    """
    matched_true = set()
    tp = 0
    for detection in detected:
        close = np.where(np.abs(ground_truth_times - detection) <= tolerance)[0] #annotations near this detection
        unmatched = [gt for gt in close if gt not in matched_true] 
        if len(unmatched) > 0:
            tp += 1 # a ture positive
            matched_true.add(unmatched[0]) #mark that annotation as matched
    fp = len(detected) - tp #detections that matched nothing
    fn = len(ground_truth_times) - len(matched_true) #anottaion never matched
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fptp = fp / tp if tp > 0 else float('inf')
    return tp, fp, fn, precision, recall, f1, fptp
    
def label_survivors(survivor_times, annotations, tolerance=0.002):
    """
    Label each survivor True (target) if it is within tolerance of an annotation, else is flase
    """
    return np.array([np.any(np.abs(annotations - t) <= tolerance) for t in survivor_times])

def extract_waveforms(survivor_times, signal, window_start, half_width=45, align_window=30, sample_frequency=30000):
    """
    Cut aligned 90-sample waveform around each survivor for the classifiers.
    retruns the wavefoms and a keep
    """
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
        trough = np.argmin(wide[centre-align_window:centre+align_window]) + (centre-align_window) #align on trough
        segment = wide[trough-half_width:trough+half_width]
        if len(segment) == 2*half_width:
            waveforms.append(segment)
            keep.append(True)
        else:
            keep.append(False)
    return np.array(waveforms), np.array(keep)

#8. classification
def oneclass_dev_eval(dev_waveforms, dev_labels, eval_waveforms, eval_labels, nu=0.1):
    """
    One class svm, train on development target waveforms only, and score on the evaluation waveforms.
    """
    dev_targets = dev_waveforms[dev_labels]#train on development targets onlly
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(dev_targets)
    eval_scores = svm.decision_function(eval_waveforms) #score on the evaluation waveforms 
    eval_preds = (eval_scores > 0).astype(int)
    return eval_labels.astype(int), eval_scores, eval_preds


def twoclass_dev_eval(dev_waveforms, dev_labels, eval_waveforms, eval_labels, C=1.0):
    scaler = StandardScaler().fit(dev_waveforms) #fit the scaler on the developemtn set only
    svm = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced")
    svm.fit(scaler.transform(dev_waveforms), dev_labels.astype(int))
    eval_scores = svm.decision_function(scaler.transform(eval_waveforms))
    eval_preds = svm.predict(scaler.transform(eval_waveforms))
    return eval_labels.astype(int), eval_scores, eval_preds

#9. metrics

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
    both_classes = len(np.unique(labels)) == 2 #ROC_AUC needs both classes present 
    roc = roc_auc_score(labels, scores) if both_classes else np.nan
    pr_ap = average_precision_score(labels, scores) if both_classes else np.nan
    return dict(precision=precision, recall=recall, f1=f1, roc_auc=roc, pr_ap=pr_ap)
    
def bootstrap_ci_stratified(labels, scores, preds=None, n_boot=2000, seed=0, alpha=0.05):
    """
    95% confidence intervals by stratified bootstraps: resample 2000 times, recompute the metrics each time
    """
    labels = np.asarray(labels); scores = np.asarray(scores)
    preds = (scores > 0).astype(int) if preds is None else np.asarray(preds)
    target_index = np.where(labels == 1)[0] #positions of the targets
    nontarget_index = np.where(labels == 0)[0]#position of the non target
    rng = np.random.default_rng(seed)
    keys = ["precision", "recall", "f1", "roc_auc", "pr_ap"]
    bootstrap_values = {metric_name: [] for metric_name in keys}
    for _ in range(n_boot):
        target_resample = rng.choice(target_index, len(target_index), replace=True) #resample targets
        nontarget_resample = rng.choice(nontarget_index, len(nontarget_index), replace=True) #resample non target
        index = np.concatenate([target_resample, nontarget_resample]) #commbine 
        if len(np.unique(labels[index])) < 2:
            continue
        m = _metrics(labels[index], preds[index], scores[index]) #mertcis for this resample
        for metric_name in keys:
            bootstrap_values[metric_name].append(m[metric_name])
    point = _metrics(labels, preds, scores)#the oint estimate 
    out = {}
    for metric_name in keys:
        array = np.asarray(bootstrap_values[metric_name], dtype=float)
        lo, hi = np.percentile(array, [100*alpha/2, 100*(1-alpha/2)]) #the 2.5th and 97.5th percentiles
        out[metric_name] = (point[metric_name], float(lo), float(hi))
    return out