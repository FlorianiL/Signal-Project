# -*- coding: utf-8 -*-
import numpy as np

# 7.5 Signal Pre-Processing
# Ex 1

def normalize(signal) :
    signal = np.array(signal)
    return signal / max(abs(signal))

# Ex 2

def split(array, width, step, sample_rate) :
    array = np.array(array)
    start = 0
    step = step*(sample_rate/1000)      #/1000 car conversion vers ms 
    width = width*(sample_rate/1000) 
    print(step)
    print(width)
    end = width
    i =0
    tab =[]
    while (i < len (array)-1):
        subtab =[]
        for j in range(start , end):
            print(start)
            print(end)
            subtab.append(array[j])
        start = start+step
        end = start+width
        i= start
        tab.append(subtab)
    return tab 


t = [0.0,2,3,4,6,8,2]
print(t)
a=normalize(t)
print(a)

print(t)
b = split(t, 2., 1., 10.0)
print(b)


# 7.6

# Signal Energy

def get_energy(signal) :
    energy = 0
    for i in range(len(signal)) :
        energy = energy + abs(signal[i])**2
    return energy

t = [0,1,2,-1,3]
print(t)
a = get_energy(t)
print a

# Pitch



