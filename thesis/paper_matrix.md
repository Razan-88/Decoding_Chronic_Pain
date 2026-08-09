# MSc Dissertation Plan: Decoding Chronic Pain
Signal Processing / ML pipeline to denoise rat peripheral nerve electrophysiology and detect/sort nociceptor action potentials. Single-chanel first, then multi-channel. 

## Context (~2 pages)
. **1.1 Motivations**: 
What chronic pain is and why nociceptor activity matters
The idea that spike timing/patterns carry information about pain, so reading individual fibres matters.
Why microneurgraphy is the tool and what does it let you access. 
. **1.2 Challenges**: 
. One electrode records many fibers mixed together
. Fiber overlap: other fibres fier at the same latency and have the same waveform shape.
. No clean reference: real recordings have no ground truth waveform for denoising
. **1.3 investigative focus**: 
. The aim: a pipeline to denoise the recording and detect/sort the target unit's spikes.
. Two phase approach: single-channel first, then Multi-channel.
. Questons: Can denoising help? what limits detections? does spatial information (multi-channel) help?
.**1.4 Structure**: one paragraph walking through the chapters. 

## Background/related work (~5 pages): organised by topic (every method used used in excution must be introduced here) 
. **2.1 preprocessing & denoising**: Review filtering, wavelet, autoencoders, self supervised (N2V). Funnel: for already filtered real data with no clean reference, which denoiser apply? (Sets up why I tested four models and why N2V's assumption matters). 
. **2.2 Spike detections & template matching**: thresholding, normalised cross-correlation, the overlapping-units problem. Funnel: shape-based methods can't separate co-shaped fibres -> motivates the latency constraint and the overlap diagnostic.
. **2.3 Realted pipelines**: Troglio, hybrid knowledge/data-driven pipeline, latency constraint, SVM, Laboy-Juárez — normalised template matching; note it is multi-channel (tetrode, template concatenated across 4 electrodes) motivates multi-channel argument. Buccino / SpikeInterface — quality metrics fail to separate true from false when distributions overlap 
Funnel: these are what I build on and compare to. 
. **2.3 The gap**: 

