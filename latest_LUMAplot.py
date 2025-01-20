import numpy as np
import h5py
import math
import pandas as pd
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


time_step = int(input("Enter the time step at which you want to plot : "))
no_of_bodies = 2#int(input("Enter the number of bodies in simulation : "))
Field_Name = input("Enter the field variable of interest, i.e., 'Ux', 'Uy', 'Resultant', 'Wz' : ")
lx = 0.195#float(input("Enter the x dimension: "))
ly = 0.12#float(input("Enter the y dimension: "))

if (Field_Name == "Wz"):
    wz_limit = float(input("Enter the vorticity range i.e., if (-2 to 2), type only '2': "))

# Actual dimension and time
grid_spacing = 0.0005 #change accordingly
dt = 0.00002
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


# Plotting
plt.axes().set_aspect('equal')

if (Field_Name == 'Resultant'):

	resultant = np.zeros(cols*rows).reshape(cols, rows)
	for i in range(0, cols):
		for j in range(0, rows):
			resultant[i][j] = math.sqrt((npu_data[i][j])**2 + (npv_data[i][j])**2)

	plt.contourf(X, Y, resultant.T, levels=500, cmap='jet')
	plt.colorbar()

elif (Field_Name == 'Wz'):
    vorticity = np.zeros((cols-2)*(rows-2)).reshape((cols-2), (rows-2))
    for i in range(0, cols-2):
        for j in range(0, rows-2):
            dvdx = (npv_data[i+1][j] - npv_data[i][j])/grid_spacing
            dudy = (npu_data[i][j+1] - npu_data[i][j])/grid_spacing
            vorticity[i][j] = 0.5*(dvdx - dudy)

    m = np.linspace(0,Lx,cols-2)
    n = np.linspace(0,Ly,rows-2)

    M, N = np.meshgrid(m,n)
    levels = np.linspace(-wz_limit, +wz_limit, 500)
    plt.contourf(M,N, vorticity.T,levels=levels, cmap='bwr', extend="both")
    plt.colorbar()	
		
elif (Field_Name == 'Ux'):
	plt.contourf(X,Y, npu_data.T,levels=500, cmap='magma')
	plt.colorbar()


elif (Field_Name == 'Uy'):
	plt.contourf(X,Y, npv_data.T,levels=500, cmap='magma')
	plt.colorbar()

#For plotting bodies

for n in range(no_of_bodies):
    if (n%2 == 0 ): rank = 0
    elif (n%2 != 0): rank = 1

    filename = "vtk_out.Body"+str(n)+"."+str(rank)+"."+str(time_step)+".vtk"

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.Update()

    nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
    nodes_nummpy_array = vtk_to_numpy(nodes_vtk_array)
    x,y = nodes_nummpy_array[:,0] , nodes_nummpy_array[:,1] + grid_spacing

    plt.plot(x, y, color = 'k', linewidth=1.0)


# plt.fill_between(circle_x_array, top_y_array, bottom_y_array, interpolate = True, color = 'mintcream')

plt.savefig(str(time_step)+"_"+str(Field_Name)+".png", dpi = 600)
#plt.show()
