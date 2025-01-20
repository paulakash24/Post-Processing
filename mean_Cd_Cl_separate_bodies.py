'''
Created on 22-Jul-2021

@author: paulstp
'''
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter

matplotlib.rc( 'text', usetex=True )
plt.rc( 'font', size=20, family="Times" )
plt.rc( 'text', usetex=True )

no_of_bodies = 3
Cd = []
for body_no in range(0, no_of_bodies):

	pipefile = open("./Body_"+str(body_no)+"_LD_rank"+str(body_no)+".out",'r') #call the output file for the mentioned body
	lines = pipefile.readlines()[1:] #skipping first 1 row

	rho = 1000
	U = 1

	t_s_array = []
	t_array = []
	Cd_array = []
	Cl_array = []


	non_dim_f = 0.5 * rho * U**2 * 0.1 * 1

	for n in range(len(lines)):

		data = re.sub(r"[\([{})\]]","", lines[n])
		split_data = data.split()
		line_len = len(split_data)
		
		

		t_s_array.append(float(split_data[0]))
		t_array.append(float(split_data[1]))

		cd_markers = 0.0
		cl_markers = 0.0

		for i in range(2, line_len, 2): # (1st body) First 2 entries are time-step and dimensional time
		
			if (float(split_data[1]) > 20):

				cd_markers += float(split_data[i])/non_dim_f
				cl_markers += float(split_data[i+1])/non_dim_f

		Cd_array.append(cd_markers)
		Cl_array.append(cl_markers)

	print("Cd of Body"+str(body_no+1)+": " + str(sum(Cd_array)/len(Cd_array)))
	Cd.append("Cd of Body"+str(body_no+1)+": " + str(sum(Cd_array)/len(Cd_array)))
	
	
	
	# Plotting
	#sns.set_style("darkgrid")
	sns.set_style("ticks")
	#sns.set_context("talk")

	f1 = plt.figure(figsize=(18,5))
	ax1 = f1.add_subplot(111)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) #### change for precision
	ax1.spines['bottom'].set_color('black')
	ax1.spines['top'].set_color('black')
	ax1.spines['left'].set_color('black')
	ax1.spines['right'].set_color('black')
	ax1.set_title('Cd')
	#ax1.set_xlim([34, 35])
	ax1.set_xlabel('time (s)')
	ax1.set_ylabel('$C_d$')
	ax1.plot(t_array,Cd_array, color='r')
	f1.savefig("Cd_History_Body_"+str(body_no + 1)+".png", dpi = 300)

	f2 = plt.figure(figsize=(18,5))
	ax2 = f2.add_subplot(111)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) #### change for precision
	ax2.spines['bottom'].set_color('black')
	ax2.spines['top'].set_color('black')
	ax2.spines['left'].set_color('black')
	ax2.spines['right'].set_color('black')
	ax2.set_title('Cl')
	#ax2.set_xlim([34, 35])
	ax2.set_xlabel('time (s)')
	ax2.set_ylabel('$C_l$')
	ax2.plot(t_array,Cl_array, color='b')
	f2.savefig("Cl_History_Body_"+str(body_no + 1)+".png", dpi = 300)

	#print("Plot saved with name" + "Cd_History_Body_"+str(body_no + 1)+".png")
	#print("Plot saved with name" + "Cl_History_Body_"+str(body_no + 1)+".png")
np.array(Cd)
np.savetxt('Cd value.txt',Cd, fmt= '%s')
