import os
import numpy as np
import h5py
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path

directory_path = os.getcwd()
p = Path(directory_path)
path_list = p.parts
folder_name = "_" + path_list[-4] + "_" + path_list[-3] + "_" + path_list[-2] + "_" + path_list[-1]

x_values_for_line = [0.765, 1.075]

time_step = 400000 #int(input("Enter the time step at which you want to plot : "))
lx = 6.0 #float(input("Enter the x dimension: "))
ly = 3.0 #float(input("Enter the y dimension: "))

# Render Latex Style
matplotlib.rc( 'text', usetex=True )
plt.rc( 'font', size=20, family="Times" )
plt.rc( 'text', usetex=True )

# Actual dimension and time
grid_spacing = 0.002 #change accordingly
dt = 0.0002
ulbm2ud_factor = grid_spacing/dt
Lx = lx + grid_spacing
Ly = ly + 2 * grid_spacing

time = 'Time_'+str(time_step)

# Finding metadata
f = h5py.File('hdf_R0N0.h5', 'r')
ls = list(f.keys())

u_data = f[time]['Ux']
v_data = f[time]['Uy']

npu_data = np.array(u_data)*ulbm2ud_factor
npv_data = np.array(v_data)*ulbm2ud_factor

cols = list(npu_data.shape)[0]
rows = list(npu_data.shape)[1]

x = np.linspace(0, Lx, cols)
y = np.linspace(0, Ly, rows)
X, Y = np.meshgrid(x, y)

colors = ['teal', 'darkviolet']
typeline = ['-', '--']

# Plotting
f1 = plt.figure(figsize=(4,5))
ax1 = f1.add_subplot(111)
#ax1.set_xticks([])
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
ax1.set_ylim([1,2])
#ax1.set_xlim([1.2,1.5])
ax1.set_xlabel(r"$U_{x}$", fontsize=16) ####
ax1.set_ylabel(r'\textit{y}', fontsize=20) ####
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=18)

for n in range(len(x_values_for_line)):
	
	index = int(x_values_for_line[n]/grid_spacing)
	Ux_loc1 = npu_data[index]
	
	ax1.plot(Ux_loc1, y, typeline[n], color = colors[n], linewidth=2.0, label=r"$x\:=$"+" "+ str(x_values_for_line[n]))
	legend = ax1.legend(loc="upper left", fontsize = 16, ncol=1)
	legend.get_frame().set_alpha(None)
    
f1.savefig("plot_over_line"+str(folder_name)+".png",bbox_inches='tight', dpi=600)
#plt.show()


