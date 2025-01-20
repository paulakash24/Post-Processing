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
typeline = ['-', '-', '-']

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


for n in range(data_id):

    if n == 0: 
    	midx = 0.725 
    	midy = 1.5
    elif n == 1: 
    	midx = 0.875 
    	midy = 1.525
    elif n == 2: 
    	midx = 0.875
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

    			
    #Plotting tip deflection history
    f1 = plt.figure(figsize=(5,5))
    ax1 = f1.add_subplot(111)
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth('1')
    ax1.set_xlabel(r"\textit{x}", fontsize=16) ####
    ax1.set_ylabel(r'\textit{y}', fontsize=20) ####
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.plot(xtip_array, ytip_array, typeline[n], color = colors[n], linewidth=2.0)
    f1.savefig("phase_plot_filament_"+str(n+1)+"_"+str(folder_name)+".png", bbox_inches='tight', dpi=600)
    print("Plot saved with name " + "phase_plot_filament_"+str(n+1)+"_"+str(folder_name)+".png")
