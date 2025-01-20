'''
Created on 05-Feb-2021

@author: PAUL AKASH
'''
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter

'''Contsants (Enter Values Here)'''
rho = 1 #density (kg/m^3)
U_fs = 0.98696 #max trailing edge velocity (m/s)
b = 0.01 #width (m)
C = 1 #chord lenth (m)
A = b*C #Area (m^2)
Const_Force = 0.5*rho*(U_fs**2)*A
f = 1.5
T = 1/f

# make regular expressions
scalarStr = r"([0-9.eE\-+]+)"
vectorStr = r"\(([0-9.eE\-+]+)\s([0-9.eE\-+]+)\s([0-9.eE\-+]+)\)"
space =  r"\s+"
threeVectorStr = r"\({}{}{}{}{}\)".format(vectorStr,space,vectorStr,space,vectorStr)
forceRegex = r"{}{}{}{}{}".format(scalarStr,space,threeVectorStr,space,threeVectorStr)

t = []
fpx = []; fpy = []; fpz = []
fpox = []; fpoy = []; fpoz = []
fvx = []; fvy = []; fvz = []
mpx = []; mpy = []; mpz = []
mpox = []; mpoy = []; mpoz = []
mvx = []; mvy = []; mvz = []

pipefile = open('./forces.dat','r') #call the .dat file here
lines = pipefile.readlines()

print("Data Length: ",len(lines)) #line count for verification

count = 0
for line in lines:
    match = re.search(forceRegex,line)
    if match:
        t.append(float(match.group(1)))
        fpx.append(float(match.group(2)))
        fpy.append(float(match.group(3)))
        fpz.append(float(match.group(4)))
        fvx.append(float(match.group(5)))
        fvy.append(float(match.group(6)))
        fvz.append(float(match.group(7)))
        fpox.append(float(match.group(8)))
        fpoy.append(float(match.group(9)))
        fpoz.append(float(match.group(10)))
        mpx.append(float(match.group(11)))
        mpy.append(float(match.group(12)))
        mpz.append(float(match.group(13)))
        mvx.append(float(match.group(14)))
        mvy.append(float(match.group(15)))
        mvz.append(float(match.group(16)))
        mpox.append(float(match.group(17)))
        mpoy.append(float(match.group(18)))
        mpoz.append(float(match.group(19)))
        
    count+=1
print("Data Reading Done..") # to ensure reading is taking place properly

   
fpx=np.array(fpx)
fvx=np.array(fvx)
fpox=np.array(fpox)
Drag = np.add(fpx,fvx,fpox)
C_d = Drag/Const_Force

fpy=np.array(fpy)
fvy=np.array(fvy)
fpoy=np.array(fpoy)
Lift = np.add(fpy,fvy,fpoy)
C_l = Lift/Const_Force

Cd_smooth = savgol_filter(C_d, 51, 3)
Cl_smooth = savgol_filter(C_l, 51, 3)

# print('Cd',len(C_d))
# print('Cl',len(C_l))
# print('t',len(t))
t_nndim = []
t_nndim = [x/T for x in t]

#print("t: ",t_nndim)

# Plotting
sns.set_style("darkgrid")
#sns.set_style("ticks")
#sns.set_context("talk")

'''======================== Cd ========================'''

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) #### change for precision
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['right'].set_color('black')
plt.ylim([-1.2,0.7]) #### change y limit
plt.xlabel('t / T', fontsize=18) ####
plt.ylabel('$C_d$', fontsize=18) ####
plt.suptitle('NACA0012 (Pitching)',fontsize = 16) #### change Title
plt.title('a8f1.5u0.5', fontsize=12) #### change label
plt.plot(t_nndim,Cd_smooth, 'r-' ,label='$C_d$')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc="upper right", fontsize = 18)
plt.show()

'''======================== Cl ========================'''

plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.ylim([-14,12]) #### change y limit
plt.xlabel('t / T', fontsize=18)
plt.ylabel('$C_l$', fontsize=18)
plt.suptitle('NACA0012 (Pitching)',fontsize = 14) #### change Title
plt.title('a8f1.5u0.5', fontsize=12) #### change label
plt.plot(t_nndim,Cl_smooth,label='$C_l$')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc="upper right", fontsize = 18)
plt.show()

