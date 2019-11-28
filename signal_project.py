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
from lpc import lpc_ref as lpc


# II Signal Pre-Processing
# Ex 1

def normalize(signal) :
    signal = np.array(signal)
    return signal / max(abs(signal))

# Ex 2

def split(signal, sample_rate, frame_size_ms, frame_step_ms) :
    signal = np.array(signal)
    frame_size = int(frame_size_ms*sample_rate/1000)
    frame_step = int(frame_step_ms*sample_rate/1000)
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


    # 2 Cepstrum-Based Pitch Estimation System
def cepstrum(audiopath):

	sample_rate, samples_int = read(audiopath)
	samples = np.array(samples_int)

	samples = mono(samples)

	treshold = 2.5
	width = 30
	step = 10

	normalized = normalize(samples)

	frames = split(normalized, sample_rate, width, step)

	energy = []
	for i in range(len(frames)) :
		energy.append(compute_energy(frames[i]))

	to_study = np.array(np.zeros(len(energy)))
	for i in range(len(frames)) :
		if energy[i]>treshold :
			to_study[i] = 1
	
	cepstrums = []
	for i in range(len(frames)):
		if to_study[i]==1:
			frame=frames[i]
			fft_frame=fft(frame)
			log_frame=np.log(np.abs(fft_frame))
			ifft_frame=ifft(log_frame)
			cepstrums.append(ifft_frame.real)
		else:
			cepstrums.append(0)
	return cepstrums

    # 3 Formants
def formants(audiopath):

	sample_rate, samples_int = read(audiopath)
	samples = np.array(samples_int)

	width = 30
	step = 10

	frames = split(samples, sample_rate, width, step)

	filtered_frames = []
	for frame in frames:
		filtered_frames.append(lfilter([1],[1., 0.67],frame))

	windowed_frames = []
	for filtered_frame in filtered_frames:
		windowed_frames.append(filtered_frame*np.hamming(len(filtered_frame)))

	frames_LPC=[]
	ncoeff = int(2 + sample_rate / 1000)
	for windowed_frame in windowed_frames:
		a = lpc(windowed_frame, ncoeff)
		frames_LPC.append(a)

	frames_roots = []
	for frame_LPC_coeficient_A in frames_LPC:
		frame_roots = np.roots(frame_LPC_coeficient_A)
		frame_roots = [r for r in frame_roots if np.imag(r) >= 0]
		frames_roots.append(frame_roots)

	frames_angles = []
	for frame_roots in frames_roots:
		frames_angles.append(np.arctan2(np.imag(frame_roots), np.real(frame_roots)))

	formants = []
	for frame_angles in frames_angles:
		formants.append(sorted(frame_angles*(sample_rate/(2*math.pi))))

	return formants