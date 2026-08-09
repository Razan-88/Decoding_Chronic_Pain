# MSc Dissertation Plan: Decoding Chronic Pain
Signal Processing / ML pipeline to denoise rat peripheral nerve electrophysiology and detect/sort nociceptor action potentials. Single-chanel first, then multi-channel. 

## Context (~2 pages)
### 1.1 Motivations: 
What chronic pain is and why nociceptor activity matters
The idea that spike timing/patterns carry information about pain, so reading individual fibres matters.
Why microneurgraphy is the tool and what does it let you access. 
 ### 1.2 Challenges: 
. One electrode records many fibers mixed together
. Fiber overlap: other fibres fier at the same latency and have the same waveform shape.
. No clean reference: real recordings have no ground truth waveform for denoising
### investigative focus**: 
. The aim: a pipeline to denoise the recording and detect/sort the target unit's spikes.
. Two phase approach: single-channel first, then Multi-channel.
. Questons: Can denoising help? what limits detections? does spatial information (multi-channel) help?
### 1.4 Structure: 
one paragraph walking through the chapters. 

## Background/related work (~5 pages): organised by topic (every method used used in excution must be introduced here) 
### 2.1 preprocessing & denoising: 
Review filtering, wavelet, autoencoders (including fully-convolutional denoising AE, Kechris), self-supervised (N2V), and the MAD noise estimator (Quiroga). Funnel: for already filtered real data with no clean reference, which denoiser apply? (Sets up why I tested four models and why N2V's assumption matters). 
### 2.2 Spike detections & template matching:
thresholding, normalised cross-correlation, the overlapping-units problem. Funnel: shape-based methods can't separate co-shaped fibres -> motivates the latency constraint and the overlap diagnostic.
### 2.3 Realted pipelines**:
Troglio, hybrid knowledge/data-driven pipeline, latency constraint, SVM, Laboy-Juárez — normalised template matching; note it is multi-channel (tetrode, template concatenated across 4 electrodes) motivates multi-channel argument. Buccino / SpikeInterface — quality metrics fail to separate true from false when distributions overlap 
Funnel: these are what I build on and compare to. 
### 2.3 The gap: 
## Single-channel Fundamentals and methods (~ 5-6 pages)
### 3.1 The data
. Rat saphenous nerve microneurgraphy, single-channel
. What's recorded: 
. Sampling rate, the 300s analysis window, pre-filtering already applied(bandpass 300-6000 Hz, common average referenced, artefact blanked).
. The key point:the signal is already filtered and spike is small (~2-3x noise).
### 3.2 Noise estimation and preprocessing
. MAD noise estimator: median(|signal|).0.6745, and why (median ignores the spike outliers.
. Note that it is an estimator (assumes mostly noise, roughly Gaussian) but valid for consistent comparison
### 3.3 Template building and spike detection
. Building a template by averaging the ground truth spikes
. Two alignments: trough-aligned (the standard) and derivative aligned (Troglio's method) and why derivitave gives a sharper template.
. Template matching by normalised cross correlation (mean-subtracted unit-norm template, variance-normalised windows).
. Amplitude thresholding (K x MAD noise, detects downward trough) with the 3ms refactory period. 
### 3.4 The Latency constraint
. The unit fires at a consistent time (~100 ms) after each stimulus 
. Wide widow (85 - 160 ms) for the who;e recording (accounts for drift)
. the tight window state that 5 ms was chosen after sweeping the width and reading the precision/recall/F1 trade off. This is where to justify my tight window choice. 
. SVM on the candidates surviving the tight window.
. Features *raw waveform snippets), RBF kernel, class imbalanced, cross validation.
. WHy it needs the balanced tight set (because the wide set's imbalnce breaks it). 
### 3.5 Denoising
. Four denoisers with their architecture : wavelet (the classical), dense autoencoder, convolutional autoencoder (Kechris FC-DAE), noise2Void (self supervised). 
. Speak abot N2V assumption (pixel wise independent noise).
### 3.6 Evaluation Metrics
' Detection metrics: precision, recall, f1, fp-per-TP and the +/- 2ms match tolerence and what each is for. 
. Denoiser quality metrics: SNR, template match consistency, MAD, spike depth and what each measures 
**When writing chapter 3:** This chapter syas ""I did X and why"", every method I intreduce in chapter 2, gets its full details in this chaper (chapter 3), The order of writing this chapter should Match the pipeline: data, noie, detection, latency, classifier, denoising, metrics. 

