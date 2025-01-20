'''
Created on 24-Jul-2021

@author: paulstp
'''

import re
import math
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter

pipefile = open("Body_0_TipPositions.out",'r') #call the output file for the mentioned body
lines = pipefile.readlines()[1:] #skipping first 1 row

t_s_array = []
t_array = []
ytip_array = []

y_array = np.loadtxt("periodicy.txt")
t_array = np.loadtxt("periodict.txt")

#print("Data Length: ",len(lines)) #line count for verification

# for line in lines:
    
#     data = re.sub(r"[\([{})\]]","", line)
#     split_data = data.split()
#     line_len = len(split_data)
    
#     t_s_array.append(float(split_data[0]))
#     t_array.append(float(split_data[1])/10)
    
#     y_data = (float((split_data[3])) - 0.06)/0.01
#     ytip_array.append(y_data)

#ytip_smooth = savgol_filter(ytip_array, 91, 3) 

N = len(t_array)

T = t_array[1] - t_array[0]

F = 1/T

f = np.linspace(0, 0.5*F, N)

sig_noise_fft = fftpack.fft(y_array)
sig_noise_amp = 2 / N * np.abs(sig_noise_fft)
sig_noise_freq = np.abs(fftpack.fftfreq(N, T))


# fft = fftpack.fft(ytip_smooth)   #p is a list
# #fftfreq=np.fft.fftfreq(0, 1/0.01, N)     #h is the step size of my grid
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.plot(sig_noise_freq,sig_noise_amp)
# X = f[:N // 2]
# Y = np.abs(fft)[:N // 2]
# plt.plot(X, Y)
plt.xlim([0,50])
plt.savefig("FFT.png", dpi=600)



# #Calculate Frequency Magnitude
# magnitudes = abs(sig_noise_fft[np.where(sig_noise_freq >= 0)])

# #Get index of top 2 frequencies
# peak_frequency = np.sort((np.argpartition(magnitudes, -2)[-2:])/t_array[-1])
# print(len(peak_frequency))
# print(peak_frequency)


maximum = np.max(sig_noise_amp)
index_of_maximum = np.argmax(sig_noise_amp)
print(sig_noise_freq.size)
print(index_of_maximum)
print("Frequency: ",sig_noise_freq[index_of_maximum])
plt.show()
print("Plot saved with name " + "FFT.png")

        
    
