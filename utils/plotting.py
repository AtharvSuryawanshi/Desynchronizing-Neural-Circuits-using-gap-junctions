import matplotlib.pyplot as plt
import sys, os
if os.getcwd() not in sys.path:  
    sys.path.append(os.getcwd())


plt.style.use('cfg/naturefigs.mplstyle')
fp_data='data_for_plots/' #filepath data
fp_figmain='fig/main/' #filepath for main figures
fp_figextra = 'fig/extra/' #filepath for extra figures

# if paths don't exist yet, create new directories
import os
for path in [fp_data,fp_figmain,fp_figextra]:
    if not os.path.exists(path):
       os.makedirs(path)

# colors in F3 outside of mplsyle file
color_model='#8eba42ff'
color_exp='#a79ecdff'
