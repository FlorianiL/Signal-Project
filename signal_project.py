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

from filterbanks import filter_banks

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

    #  Used in autocorrelation
def mono(signal):
	signal = np.array(signal.astype(float))
	if signal.ndim == 1:
		return signal
	else:
		return signal.sum(axis=1)/2


    # Main method
def autocorrelation(audiopath):
	# Initialisation
	sample_rate, samples_int = read(audiopath)
	samples = np.array(samples_int)
	samples = mono(samples)
	threshold = 2.5
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
    # Normalize the signal into frames
	sample_rate, samples_int = read(audiopath)
	samples = np.array(samples_int)
	samples = mono(samples)
	treshold = 2.5
	width = 30
	step = 10
	normalized = normalize(samples)
	frames = split(normalized, sample_rate, width, step)
    # Compute the energy of each frame
	energy = []
	for i in range(len(frames)) :
		energy.append(compute_energy(frames[i]))
    # Compare energy with trehold => Voiced or not
	to_study = np.array(np.zeros(len(energy)))
	for i in range(len(frames)) :
		if energy[i]>treshold :
			to_study[i] = 1
	# Compute the cepstrum and from these determine f0
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


# C Formants
def formants(audiopath):
    # Split the signal in frames
	sample_rate, samples_int = read(audiopath)
	samples = np.array(samples_int)
	width = 30
	step = 10
	frames = split(samples, sample_rate, width, step)
    # Filter the signal
	filter_frames = []
	for frame in frames:
		filter_frames.append(lfilter([1],[1., 0.67],frame))
    # Apply hamming window onto the signal
	windowed_frames = []
	for filtered_frame in filter_frames:
		windowed_frames.append(filtered_frame*np.hamming(len(filtered_frame)))
    # Extract LPC coefficients
	frames_LPC=[]
	ncoeff = int(2 + sample_rate / 1000)
	for windowed_frame in windowed_frames:
		a = lpc(windowed_frame, ncoeff)
		frames_LPC.append(a)
    # Find roots of the LPC
	frames_roots = []
	for frame_LPC_coeficient_A in frames_LPC:
		frame_roots = np.roots(frame_LPC_coeficient_A)
		frame_roots = [r for r in frame_roots if np.imag(r) >= 0]
		frames_roots.append(frame_roots)
    # Find angle
	frames_angles = []
	for frame_roots in frames_roots:
		frames_angles.append(np.arctan2(np.imag(frame_roots), np.real(frame_roots)))
    # Deduce Hz values
	formants = []
	for frame_angles in frames_angles:
		formants.append(sorted(frame_angles*(sample_rate/(2*math.pi))))

	return formants


# D MFCC
def MFCC(audiopath):
	#0.init
	samplerate, samples_int = read(audiopath)
	samples = np.array(samples_int)

	NTFD = 512

	#1.split the signal into frames
	width = 30
	step = 10

	frames = split(samples, samplerate, width, step)

	#2.pass the signal into a first order high pass filter
	filter_frames = []
	for frame in frames:
		filter_frames.append(lfilter([1],[1., 0.97],frame))

	#3.apply a hamming window onto the signal
	windowed_frames = []
	for filtered_frame in filter_frames:
		windowed_frames.append(filtered_frame*np.hamming(len(filtered_frame)))

	#4.compute the power spectrum
	power_frames = []
	for windowed_frame in windowed_frames:
		power_frames.append((fft(windowed_frame)**2)/2*NTFD)

	#5.mel-filter
	filter_banks_frames = filter_banks(power_frames,samplerate,nfilt=len(power_frames[0]),NFFT=2*(len(power_frames[0])-1))

	#6.apply direct cosine transform
	_dct = dct(filter_banks_frames,type=2,axis=1,norm='ortho')

	#7.keep the first 13
	_dct=_dct[:13]

	return _dct

    # Visualize 
def visualize(n):
	male_audiopaths = []
	female_audiopaths = []
	for i in range(n) :
		male_audiopaths.append('../resources/cmu_us_bdl_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_bdl_arctic/wav/')))
		female_audiopaths.append('../resources/cmu_us_slt_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_slt_arctic/wav/')))
	male_f_zeros = []
	female_f_zeros = []
	male_f1 = []
	male_f2 = []
	male_f3 = []
	female_f1 = []
	female_f2 = []
	female_f3 = []
	i = 0
	for audiopath in male_audiopaths + female_audiopaths :
		print("processing path : ",audiopath)
		abpes = autocorrelation(audiopath)
		plt.gcf().clear()
		fzero = 0
		total = 0
		for f0 in abpes :
			if(f0 != 0) and (f0 < 500):
					fzero += f0
					total += 1
		if(i < n):
			male_f_zeros.append(int(fzero/total))
		else:
			female_f_zeros.append(int(fzero/total))

		f = formants(audiopath)
		f1 = 0
		f2 = 0
		f3 = 0
		total1 = 0
		total2 = 0
		total3 = 0
		for form in f :
			if int(form[0]) != 0 :
				f1 = f1 + form[0]
				total1 = total1 + 1
			if int(form[1]) != 0 :
				f2 = f2 + form[1]
				total2 = total2 + 1
			if int(form[2]) != 0 :
				f3 = f3 + form[2]
				total3 = total3 + 1
		if(i < n) :
			male_f1.append(int(f1/total1))
			male_f2.append(int(f2/total2))
			male_f3.append(int(f3/total3))
		else:
			female_f1.append(int(f1/total1))
			female_f2.append(int(f2/total2))
			female_f3.append(int(f3/total3))
		i += 1

	print('male f0 : ',male_f_zeros,'; avg = ',int(sum(male_f_zeros)/len(male_f_zeros)))
	print('female f0 : ',female_f_zeros,'; avg = ',int(sum(female_f_zeros)/len(female_f_zeros)))

	print('male F1 : ',male_f1,'; avg = ',int(sum(male_f1)/len(male_f1)))
	print('male F2 : ',male_f2,'; avg = ',int(sum(male_f2)/len(male_f2)))
	print('male F3 : ',male_f3,'; avg = ',int(sum(male_f3)/len(male_f3)))

	print('female F1 : ',female_f1,'; avg = ',int(sum(female_f1)/len(female_f1)))
	print('female F2 : ',female_f2,'; avg = ',int(sum(female_f2)/len(female_f2)))
	print('female F3 : ',female_f3,'; avg = ',int(sum(female_f3)/len(female_f3)))


def rule_based_system(n):
	audiopaths=[]

	for i in range(n):
		if npr.choice([True,False]):
			audiopaths.append('../resources/cmu_us_bdl_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_bdl_arctic/wav/')))
		else:
			audiopaths.append('../resources/cmu_us_slt_arctic/wav/'+random.choice(os.listdir('../resources/cmu_us_slt_arctic/wav/')))
	
	for audiopath in audiopaths:
		print('processing ',audiopath)
		abpes=autocorrelation_based_pitch_estimation_system(audiopath)
		fzero=0
		total=0
		for f0 in abpes:
			if f0!=0 and f0<500:
				fzero=fzero+f0
				total=total+1
		fzero=fzero/total

		f=formants(audiopath)
		f1=0
		f2=0
		f3=0
		total1=0
		total2=0
		total3=0
		for form in f:
			if int(form[0])!=0:
				f1=f1+form[0]
				total1=total1+1
			if int(form[1])!=0:
				f2=f2+form[1]
				total2=total2+1
			if int(form[2])!=0:
				f3=f3+form[2]
				total3=total3+1

		f1=f1/total1
		f2=f2/total2
		f3=f3/total3

		if fzero<150:
			if f3<1400:
				print("C'est l'homme !")
			else:
				print("C'est certainement l'homme...")

		else:
			if f3>=1400:
				print("C'est la femme !")
			else:
				print("C'est certainement la femme...")

rule_based_system(10)