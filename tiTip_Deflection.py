'''
Created on 24-Jul-2021

@author: paulstp
'''

import re
import os
import math
import numpy as np
import scipy.fftpack
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy.fftpack as fftpack
from pathlib import Path

directory_path = os.getcwd()
p = Path(directory_path)
path_list = p.parts
folder_name = "_" + path_list[-4] + "_" + path_list[-3] + "_" + path_list[-2] + "_" + path_list[-1]

data_id = 3 #int(input("Enter the number of datasets : \n"))

colors = ['k', 'b', 'r']
typeline = ['-', '--', '--']

matplotlib.rc( 'text', usetex=True )
plt.rc( 'font', size=20, family="Times" )
plt.rc( 'text', usetex=True )

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

periodic_y_1 = []
periodic_y_2 = []
periodic_y_3 = []
periodic_t = []
frequencies_array = []
timesteps_cycle = []

#Plotting tip deflection history
f1 = plt.figure(figsize=(8,5))
ax1 = f1.add_subplot(111)
ax1.set_xticks([])
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
ax1.set_ylim(-1,1)
ax1.set_xlim([120,130])
#ax1.set_xlabel(r"\textit{t (s)}", fontsize=16) ####
ax1.set_ylabel(r'\textit{y/L}', fontsize=20) ####
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=18)
#ax1.grid(color='dimgray', linestyle='--', linewidth=0.4)

# Storing frequency of flapping of filaments
f2 = plt.figure(figsize=(8,5))
ax2 = f2.add_subplot(111)
ax2.set_xticks([])
ax2.grid(color='dimgray', linestyle='--', linewidth=0.4)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=18)
ax2.set_ylabel(r'\textit{Amplitude}', fontsize=20)
#ax2.set_xlabel("Frequency [Hz]")


for n in range(data_id):

    if n == 0: mid = 1.5
    elif n == 1: mid = 1.525
    elif n == 2: mid = 1.475

    pipefile = open("Body_"+str(n)+"_TipPositions.out",'r') #call the output file for the mentioned body
    lines = pipefile.readlines()[1:] #skipping first 1 row
    t_s_array = []
    t_array = []
    xtip_array = []
    ytip_array = []
    t_array_cycles = []

    for line in lines:

        data = re.sub(r"[\([{})\]]","", line)
        split_data = data.split()
        line_len = len(split_data)

        t_s_array.append(float(split_data[0]))
        t_array.append(float(split_data[1]))
		
        y_data = (float((split_data[3])) - mid)/0.1
        ytip_array.append(y_data)

    for i in range(len(t_array)):
    
    	if (t_array[i] > 20):
    		
    		if (n==0):
    			periodic_y_1.append(ytip_array[i])
    			periodic_t.append(t_array[i])
    		elif (n==1):
    			periodic_y_2.append(ytip_array[i])
    		elif (n==2):
    			periodic_y_3.append(ytip_array[i])
    if n==0:
    	y_array = periodic_y_1
    elif n==1:
    	y_array = periodic_y_2
    elif n==2:
    	y_array = periodic_y_3
    y_array = np.array(y_array)
    y_array = y_array - y_array.mean()
    
    N = len(periodic_t)
    
    T = periodic_t[1] - periodic_t[0]
    
    F = 1/T
    f = np.linspace(0, 0.5*F, N)
    
    sig_noise_fft = fftpack.fft(y_array)
    sig_noise_amp = 2 / N * np.abs(sig_noise_fft)
    sig_noise_freq = np.abs(fftpack.fftfreq(N, T))
    
    ax2.plot(sig_noise_freq,sig_noise_amp, label=r"$Filament$"+" "+ str(n+1))
    ax2.set_xlim([0,10])
    legend = ax2.legend(loc="upper right", fontsize = 16, ncol=1)
    legend.get_frame().set_alpha(None)
    
    maximum = np.max(sig_noise_amp)
    index_of_maximum = np.argmax(sig_noise_amp)
    freq = sig_noise_freq[index_of_maximum]
    print("Frequency of Body "+str(n+1)+  " :", freq)
    frequencies_array.append("Frequency of Body "+str(n+1)+" : " + str(sig_noise_freq[index_of_maximum]))
    
    for i in range(len(t_array)):
    	t_array_cycles.append(t_array[i]*freq)
    	if t_array[i]*freq > t_array[-1]*freq - 1 and t_array[i]*freq <= t_array[-1]*freq and t_s_array[i]%100 == 0:
    		timesteps_cycle.append(t_s_array[i])
    		
    '''	
    if n==0:
    	ax1.plot(t_array_cycles, smooth(ytip_array,19), typeline[n], color = colors[n], linewidth=2.0, label=r"$Filament$"+" "+ str(n+1))
    else:
    	ax1.plot(t_array_cycles, smooth(ytip_array,19), typeline[n], color = colors[n], linewidth=2.0)
    '''
    ax1.plot(t_array_cycles, smooth(ytip_array,19), typeline[n], color = colors[n], linewidth=2.0)
    #legend = ax1.legend(loc="upper left", fontsize = 22, ncol=2)
    #legend.get_frame().set_alpha(None)
     
f1.savefig("tip_hist"+str(folder_name)+".png",bbox_inches='tight', dpi=600)
print("Plot saved with name " + "tip_histories.png")
#plt.show()
print("\n")

np.array(frequencies_array)
f2.savefig("yFFT"+str(folder_name)+".png", bbox_inches='tight', dpi=600)
np.savetxt('Frequency_values.txt',frequencies_array, fmt= '%s')

np.array(timesteps_cycle)
np.savetxt('Envelope_timestep_values.txt',timesteps_cycle, fmt= '%s')

# Storing tip deflection amplitudes of filaments
deflections_array = []
for n in range(data_id):

	if n==0:
		y_array = periodic_y_1
	elif n==1:
		y_array = periodic_y_2
	elif n==2:
		y_array = periodic_y_3
	periodic_y = np.array(y_array)
		
	dmax = np.max(periodic_y) - np.min(periodic_y)
	print("Tip Def. Amp. of Body "+str(n+1)+": " + str(dmax))
	deflections_array.append("Tip Def. Amp. of Body "+str(n+1)+": " + str(dmax))
np.array(deflections_array)
np.savetxt('Deflection_Amplitude_values.txt',deflections_array, fmt= '%s')
print("\n")
