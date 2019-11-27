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
    signal=np.array(signal)
    frame_size=int(frame_size_ms*samplerate/1000)
    frame_step=int(frame_step_ms*samplerate/1000)
    frames=[]
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

#  !!!!! Probablement utiliser plus loin
def mono(signal):
	signal=np.array(signal.astype(float))
	if signal.ndim==1:
		return signal
	else:
		return signal.sum(axis=1)/2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def autocorrelation_based_pitch_estimation_system(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples=np.array(samples_int)

	samples=mono(samples)

	treshold=2.5 #subject to changes
	width=30
	step=10
	_maxlags=(width*samplerate/1000)-1 #full cover

	_maxlags=int(_maxlags) 

	#1.normalize the signal
	normalized_samples=normalize(samples)

	#2.split the signal into frames
	frames=split(normalized_samples, samplerate, width, step)

	#3.compute the energy for each frames
	energy=[]
	for i in range(len(frames)):
		energy.append(compute_energy(frames[i]))

	#4.separate voiced/unvoiced
	to_study=np.array(np.zeros(len(energy)))
	for i in range(len(frames)):
		if energy[i]>treshold:
			to_study[i]=1

	#5.compute the autocorrelation of each frame
	autocorrs=[]
	for i in range(len(frames)):
		if to_study[i]==1:
			a,tmp,b,c=plt.xcorr(frames[i],frames[i],maxlags=_maxlags)
			tmp=tmp[_maxlags:]
			autocorrs.append(tmp)
		else:
			autocorrs.append(np.zeros(_maxlags+1))
	
	#6.Find distances between the two highest peaks for each autocorrelated frames.
	#  First one is always the highest (since we cut the autocorrelation result in half) so we only need 
	#  to find the highest remaining.
	peaks=[]
	for i in range(len(autocorrs)):
		if to_study[i]==1:
			peaks_tmp, peaks_tmp_prop = find_peaks(autocorrs[i], height=0)
			#if np.array_equal([],peaks_tmp):
			#	peaks.append(0)
			#	to_study[i]=0
			#else:
			index_max=peaks_tmp[np.argmax(peaks_tmp_prop["peak_heights"])]
			peaks.append(index_max)
		else:
			peaks.append(0)

	#7.compute the fondamental frequencies from the distances between the peaks and the samplerate
	f_zeros=[]
	for i in range(len(peaks)):
		if to_study[i]==1 and peaks[i]!=0:
			f_zeros.append(samplerate/peaks[i])
		else:
			f_zeros.append(0)

	return f_zeros