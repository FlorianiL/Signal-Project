# -*- coding: utf-8 -*-
from __future__ import division
import os
import random 
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
from scipy.fftpack import dct
from scipy.signal import find_peaks,lfilter
from scipy.io.wavfile import read

# II Signal Pre-Processing
# Ex 1

def normalize(signal) :
    signal = np.array(signal)
    return signal / max(abs(signal))

# Ex 2

def split(signal, samplerate, frame_size_ms, frame_step_ms) :
    signal = np.array(signal)
    frame_size = int(frame_size_ms*samplerate/1000)
    frame_step = int(frame_step_ms*samplerate/1000)
    frames = []
    for i in range(0,len(signal)-len(signal)%frame_size, frame_step):
        frames.append(signal[i:i+frame_size])
    return frames

# III Features Extraction algorithms
# A. Signal Energy

def compute_energy(signal):
	signal = np.array(signal)
	return sum(abs(signal)**2)

# B. Pitch
    # 1 Autocorrolation-based pitch estimation system

	#  Used in autocorrelation1
def mono(signal):
	signal = np.array(signal.astype(float))
	if signal.ndim == 1:
		return signal
	else:
		return signal.sum(axis=1)/2
	# Main method
def autocorrelation1(audiopath):
	# Initialisation
	sample_rate, samples_int = read(audiopath)
	samples = np.array(samples_int)

	samples = mono(samples)

	threshold = 2.5 #subject to changes
	width = 30
	step = 10
	maxl = (width*sample_rate/1000)-1

	maxl = int(maxl) 

	# Normalize and split signal
	norm_samples = normalize(samples)

	frames = split(norm_samples, sample_rate, width, step)

	# Compute energy for each frames
	energy = []
	for i in range(len(frames)):
		energy.append(compute_energy(frames[i]))

	# Sort voiced and unvoiced
	to_check = np.array(np.zeros(len(energy)))
	for i in range(len(frames)):
		if energy[i] > threshold:
			to_check[i] = 1

	# Compute autocorrelation of each frame
	autocorrs = []
	for i in range(len(frames)) :
		if to_check[i] == 1 :
			a,tmp,b,c = plt.xcorr(frames[i],frames[i],maxlags = maxl)
			tmp = tmp[maxl:]
			autocorrs.append(tmp)
		else:
			autocorrs.append(np.zeros(maxl+1))
	
	# Find distances between the two highest peaks for each autocorrelated frames
	peaks = []
	for i in range(len(autocorrs)):
		if to_check[i] == 1:
			peaks_tmp, peaks_tmp_prop = find_peaks(autocorrs[i], height=0)
			index_max = peaks_tmp[np.argmax(peaks_tmp_prop["peak_heights"])]
			peaks.append(index_max)
		else:
			peaks.append(0)

	# Compute fondamental frequencies from distances between peaks and sample rate
	f_zeros = []
	for i in range(len(peaks)):
		if to_check[i] == 1 and peaks[i] != 0:
			f_zeros.append(sample_rate/peaks[i])
		else:
			f_zeros.append(0)

	return f_zeros
