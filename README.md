#  Decoding chronic pain: denosiing and spike sorting in single channel.
Data Science MSc dissertation project evaluating whether denoising improves single channel spike detection and classification in rat saphenous nerve recording, using a leakage safe evaluation across multiple channels.    

## Overview
This project develops a leakage safe pipeline that:
**Detects** spikes by template matching using normalised cross correlation with post stimulus constraint. 
**Denosies** the signal using three denoisers: wavelet, a dense autoencoder and convolutional autoencoder.
**Classifiers** detected candidates (target vs non targets spikes) using one class and two class SVMs
**Evaluates** detection and classification on held out data across five channels independently. 
All tuning (template, denoisers, detection parameters, classifiers) is performed on the developemnt set and performance is scored once on the held out evaluation set to avoid bias and information leakage that can inflate the results. 

## Repository structure
Notebooks/
Final_leak_safe.ipynb # main analysis
exploratory_with_leakage.ipynb # earlier exploratory analysis that contains data leakage **not used for final results**
pipeline_functions.py # contains reusable functions (template, denoiser, detection, classification, metrics)
figure/ #results figures
Dissertation/ #LaTeX source of the dissertation
planing/ #project plan and paper matrix
requirement.txt #Python dependencies  

## Requirements
- python 3.13
- Numpy, Scipy, PyTorch, PyWavelet, scikit-learn, h5py, Matplotlib

## Author 
Rzan, MSc Data science, University of Bristol. Supervised by Zahraa Abdallah
