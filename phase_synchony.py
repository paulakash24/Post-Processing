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
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
from pathlib import Path

directory_path = os.getcwd()
p = Path(directory_path)
path_list = p.parts
folder_name = "_" + path_list[-4] + "_" + path_list[-3] + "_" + path_list[-2] + "_" + path_list[-1]

data_id = 3 #int(input("Enter the number of datasets : \n"))

colors = ['k', 'b', 'r']
typeline = ['-', '-', '-']

matplotlib.rc( 'text', usetex=True )
plt.rc( 'font', size=20, family="Times" )
plt.rc( 'text', usetex=True )

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

periodic_x_1 = []
periodic_x_2 = []
periodic_x_3 = []
periodic_y_1 = []
periodic_y_2 = []
periodic_y_3 = []
periodic_t = []

start = 0.625
L = 0.1 # length of filament

for n in range(data_id):

    if n == 0: 
    	midx = 0.625 + L
    	midy = 1.5
    elif n == 1: 
    	midx = start + L + float(path_list[-1]) * L + L
    	midy = 1.525
    elif n == 2: 
    	midx = start + L + float(path_list[-1]) * L + L
    	midy = 1.475

    pipefile = open("Body_"+str(n)+"_TipPositions.out",'r') #call the output file for the mentioned body
    lines = pipefile.readlines()[1:] #skipping first 1 row
    t_s_array = []
    t_array = []
    xtip_array = []
    ytip_array = []

    for line in lines:

        data = re.sub(r"[\([{})\]]","", line)
        split_data = data.split()
        line_len = len(split_data)
        
        if (float(split_data[1]) > 20):
        	t_s_array.append(float(split_data[0]))
        	t_array.append(float(split_data[1]))
        	x_data = (float((split_data[2])) - midx)
        	xtip_array.append(x_data)
        	y_data = (float((split_data[3])) - midy)
        	ytip_array.append(y_data)
    if n==0:
    	periodic_x_1 = xtip_array
    	periodic_y_1 = ytip_array
    elif n == 1: 
    	periodic_x_2 = xtip_array
    	periodic_y_2 = ytip_array
    elif n == 2: 
    	periodic_x_3 = xtip_array
    	periodic_y_3 = ytip_array
		
periodic_y_1_h = hilbert(periodic_y_1)
periodic_y_2_h = hilbert(periodic_y_2)
periodic_y_3_h = hilbert(periodic_y_3)

#phase_synchrony_1 = 1-np.sin(np.abs(periodic_y_1_h-periodic_y_2_h)/2)
#phase_synchrony_2 = 1-np.sin(np.abs(periodic_y_1_h-periodic_y_3_h)/2)

phase_synchrony_1 = np.unwrap(np.angle(periodic_y_1_h)) - np.unwrap(np.angle(periodic_y_2_h))
phase_synchrony_2 = np.unwrap(np.angle(periodic_y_1_h)) - np.unwrap(np.angle(periodic_y_3_h))
phase_synchrony_3 = np.unwrap(np.angle(periodic_y_2_h)) - np.unwrap(np.angle(periodic_y_3_h))

phase_synchrony_1 = (phase_synchrony_1 + np.pi) % (2 * np.pi) - np.pi
phase_synchrony_2 = (phase_synchrony_2 + np.pi) % (2 * np.pi) - np.pi
phase_synchrony_3 = (phase_synchrony_3 + np.pi) % (2 * np.pi) - np.pi

print("Phase difference between 1 and 2: "+str(np.mean(phase_synchrony_1)))
print("Phase difference between 1 and 3: "+str(np.mean(phase_synchrony_2)))
print("Phase difference between 2 and 3: "+str(np.mean(phase_synchrony_3)))

shifts = []
shifts.append(np.mean(phase_synchrony_1))
shifts.append(np.mean(phase_synchrony_2))
shifts.append(np.mean(phase_synchrony_3))
np.array(shifts)
np.savetxt('Phase shift value.txt',shifts, fmt= '%s')

#Plotting tip deflection history
f1 = plt.figure(figsize=(15,4))
ax1 = f1.add_subplot(111)
#ax1.set_xticks([])
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
#ax1.set_ylim(-3.3, -2.5)
ax1.set_xlim([20,80])
ax1.set_xlabel(r"\textit{t (s)}", fontsize=16) ####
ax1.set_ylabel(r'$\phi_{inst.}$', fontsize=20) ####
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
#ax1.grid(color='dimgray', linestyle='--', linewidth=0.4)

ax1.plot(t_array, phase_synchrony_1, "--", color = "dodgerblue", linewidth=1.0, label="Filament 1 and 2")
ax1.plot(t_array, phase_synchrony_2, "-", color = "blueviolet", linewidth=1.0, label="Filament 1 and 3")
legend = ax1.legend(loc="upper left", fontsize = 16, ncol=2)
legend.get_frame().set_alpha(None)

f1.savefig("phase_synchorny"+str(folder_name)+".png",bbox_inches='tight', dpi=600)
print("Plot saved with name " + "phase_synchorny"+str(folder_name)+".png")
#plt.show()
print("\n")


