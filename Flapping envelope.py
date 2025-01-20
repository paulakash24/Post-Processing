import os
import numpy as np
import h5py
import math
import pandas as pd
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path

directory_path = os.getcwd()
p = Path(directory_path)
path_list = p.parts
folder_name = "_" + path_list[-4] + "_" + path_list[-3] + "_" + path_list[-2] + "_" + path_list[-1]

no_of_bodies = 3 #int(input("Enter the number of bodies in simulation : "))

# Plotting
fig = plt.figure(figsize=(7,2))
plt.xlim([0.6,1.34])
plt.ylim([1.4,1.6])
#plt.box(False)
plt.xticks([])
plt.yticks([])
#For plotting bodies

pipefile = open("./Envelope_timestep_values.txt",'r') #call the output file for the mentioned body
lines = pipefile.readlines()

start_time_step = int(float(lines[0]))
end_time_step = int(float(lines[-1]))

for n in range(no_of_bodies):
    if (n%2 == 0 ): rank = 0
    elif (n%2 != 0): rank = 1
    
    for time_step in range(start_time_step, end_time_step+100, 100):
		

	    filename = "vtk_out.Body"+str(n)+"."+str(n)+"."+str(time_step)+".vtk"

	    reader = vtk.vtkPolyDataReader()
	    reader.SetFileName(filename)
	    reader.ReadAllScalarsOn()
	    reader.Update()

	    nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
	    nodes_nummpy_array = vtk_to_numpy(nodes_vtk_array)
	    x,y = nodes_nummpy_array[:,0] , nodes_nummpy_array[:,1]
	    plt.plot(x, y, color = 'k', linewidth=0.3)
	    
	    '''
	    if (time_step == end_time_step):
	    	plt.plot(x, y, color = 'fuchsia', linewidth=1.0)
	    else:	
	    	plt.plot(x, y, color = 'k', linewidth=0.2)
		'''
		

# plt.fill_between(circle_x_array, top_y_array, bottom_y_array, interpolate = True, color = 'mintcream')

plt.savefig("envelope"+str(folder_name)+".png",bbox_inches='tight', dpi = 1200)
#plt.show()
