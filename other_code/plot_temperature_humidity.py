# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:08:52 2024

plot temperature and humidity

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 


#%% read data 
date = '2024-07-04'
log_path = r'Z:\Dinghao\temp_humid_monitor\log_{}.txt'.format(date)
f = open(log_path, 'r')

times = []
temps = []
humids = []
for line in f:
    if line[:10]==date:
        times.append(line[11:19])  # to get hh:mm:ss
    if 'C' in line:  # if it is temperature
        temp = float(line[line.index('C')-5:line.index('C')])
        temps.append(temp)
    if '%' in line:  # if it is humidity 
        humid = float(line[line.index('%')-5:line.index('%')])
        humids.append(humid)
        
f.close()


#%% labels 
xaxis = np.arange(len(times))
times_ticks = [i for i, t in enumerate(times) if np.mod(i, 60)==0]  # 60 samples is 10 minutes
times_10min = [t[:5] for i, t in enumerate(times) if np.mod(i, 60)==0]  # same as above


#%% plot data 
fig, axt = plt.subplots(figsize=(4,2))

axh = axt.twinx()

axt.plot(xaxis, temps, 'cornflowerblue')
axt.set(ylabel='temperature (C)')

axh.plot(xaxis, humids, 'burlywood')
axh.set(ylabel='humidity (%)')

axt.set(xticks=times_ticks, xticklabels=times_10min)

axt.spines['left'].set_color('cornflowerblue')
axt.spines['left'].set_linewidth(1)
axt.yaxis.label.set_color('royalblue')
axt.tick_params(axis='y', colors='royalblue')

axt.spines['right'].set_color('burlywood')
axt.spines['right'].set_linewidth(1)
axh.yaxis.label.set_color('darkgoldenrod')
axh.tick_params(axis='y', colors='darkgoldenrod')

axt.spines['bottom'].set_color('grey')
axt.spines['bottom'].set_linewidth(1)
axt.tick_params(axis='x', colors='dimgrey')

axt.spines['top'].set_visible(False)
for s in ['top', 'left', 'right', 'bottom']:
    axh.spines[s].set_visible(False)