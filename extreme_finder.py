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
from scipy.signal import savgol_filter

# body_id = int(input("Enter the body ID of the body for which Cd Cl will be plotted : \n"))


pipefile = open("Body_0_TipPositions.out",'r') #call the output file for the mentioned body
lines = pipefile.readlines()[1:] #skipping first 1 row

rho = 1000
U = 1

t_s_array = []
t_array = []
xtip_array = []
ytip_array = []


non_dim_f = 0.5 * rho * U**2 * 0.1*1

print("Data Length: ",len(lines)) #line count for verification

for line in lines:
    
    data = re.sub(r"[\([{})\]]","", line)
    split_data = data.split()
    line_len = len(split_data)
    
    t_s_array.append(float(split_data[0]))
    t_array.append(float(split_data[1]))
    
    xtip_array.append(float(split_data[2]))
    y_data = (float((split_data[3])) - 0.06)/0.01
    ytip_array.append(y_data)
    
    

xtip_smooth = savgol_filter(xtip_array, 91, 3)
ytip_smooth = savgol_filter(ytip_array, 91, 3) 

Arms = np.sqrt(np.mean(np.square(ytip_array)))


periodic_y = []
periodic_t = []
ext_y = []
ext_ts = []

        
for i in range(len(t_array)):
    if (t_array[i] > 10):
        periodic_t.append(t_array[i])
        
        periodic_y.append(ytip_array[i])
        
        
        
    if (t_array[i] >= 17.5 and t_array[i] <= 17.8):
    	ext_ts.append(t_s_array[i])
    	ext_y.append(ytip_array[i])
        
min_y = min(ext_y)
max_y = max(ext_y)

min_index = ext_y.index(min_y)
max_index = ext_y.index(max_y)

ts_min = ext_ts[min_index]

ts_max = ext_ts[max_index]
    	

print("min timestep: ", ts_min)
print("max timestep: ", ts_max)



dmax = np.max(periodic_y) - np.min(periodic_y)

np.savetxt('periodicy.txt',periodic_y)
np.savetxt('periodict.txt',periodic_t)

print("Arms = ",Arms)
print("dmax = ",dmax/2)
        


        
    
