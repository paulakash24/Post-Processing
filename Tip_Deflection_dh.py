'''
Created on 24-Jul-2021

@author: paulstp
'''

import re
import math
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

data_id = 3#int(input("Enter the number of datasets : \n"))

Cos = ["1750", "2000", "2250"]
colors = {"1750":'g',"2000":'k', "2250":'m'}
typeline = {"1750":'--', "2000":'-', "2250":'--'}

sns.set_style("darkgrid")
f1 = plt.figure(figsize=(15,5))
ax1 = f1.add_subplot(111)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

for n in range(data_id):

	pipefile = open("Body_0_TipPositions"+str(n)+".out",'r') #call the output file for the mentioned body
	lines = pipefile.readlines()[1:] #skipping first 1 row

	t_s_array = []
	t_array = []
	xtip_array = []
	ytip_array = []

	for line in lines:
	    
	    data = re.sub(r"[\([{})\]]","", line)
	    split_data = data.split()
	    line_len = len(split_data)
	    
	    t_s_array.append(float(split_data[0]))
	    t_array.append(float(split_data[1]))
	    
	    y_data = (float((split_data[3])) - 0.06)/0.01
	    ytip_array.append(y_data)

	# cubic_interploation_model = interp1d(t_array, ytip_array, kind = "cubic")
	# t_smooth = np.linspace(min(t_array), max(t_array), 2*len(ytip_array))
	# y_smooth = cubic_interploation_model(t_smooth)

	# print('length',len(ytip_array))
	# ytip_smooth = savgol_filter(ytip_array, 51, 3)


	        
	# Plotting
	ax1.patch.set_edgecolor('black')  
	ax1.patch.set_linewidth('1') 
	ax1.set_ylim(-1.5,1.5)
	#ax1.set_xlim([0,10])
	ax1.set_xlabel(r'$t (s)$', fontsize=18) ####
	ax1.set_ylabel(r'$d_{max}$', fontsize=18) ####
	ax1.set_title("Tip deflection history", fontsize=16) #### change label
	ax1.plot(t_array, smooth(ytip_array,19), typeline[Cos[n]], color = colors[Cos[n]], linewidth=0.8, label='Resolution = '+Cos[n])
	ax1.legend(loc="lower center", fontsize = 12, ncol=len(Cos))



plt.savefig("dh_inde_Tip deflection History.png", dpi=600)
#plt.get_current_fig_manager().window.showMaximized()
#plt.show()

print("Plot saved with name " + "Tip deflection History.png")

        
    
