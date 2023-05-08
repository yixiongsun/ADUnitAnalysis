import pandas as pd
import scipy.io as sio
import os
import unit_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
#directory = 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\'
#rating_file = 'ClusterRating_2022-03-23_09-17-15-p.csv'


#directory = 'D:\\TetrodeData\\2022-03-24_09-28-38-p\\'
#rating_file = 'ClusterRating_2022-03-24_09-28-38-p.csv'

#directory = 'D:\\TetrodeData\\2022-04-12-09-29-11-p\\'
#rating_file = 'ClusterRating_2022-04-12_09-29-11-p.csv'

#directory = 'D:\\TetrodeData\\2022-04-13_09-31-41-p\\'
#rating_file = 'ClusterRating_2022-04-13_09-31-41-p.csv'


#directory = 'D:\\TetrodeData\\2022-07-08_10-01-42-p\\'
#rating_file = 'ClusterRating_2022-07-08_10-01-42-p.csv'

#directory = 'D:\\TetrodeData\\2022-07-09_09-54-32-p\\'
#rating_file = 'ClusterRating_2022-07-09_09-54-32-p.csv'

#directory = 'D:\\TetrodeData\\2022-07-10_13-27-33-p\\'
#rating_file = 'ClusterRating_2022-07-10_13-27-33-p.csv'

#directory = 'D:\\TetrodeData\\2022-07-11_13-47-33-p\\'
#rating_file = 'ClusterRating_2022-07-11_13-47-33-p.csv'

directory = 'D:/Data/04-02-2023/'
rating_file = 'ClusterRating_2023-04-02_09-35-57-p.csv'

#inter = {1:[6], 2:[12], 3:[15, 26, 28], 4:[10, 13, 15, 18], 5:[10, 13, 15, 21, 22]}

# only first 5 tetrodes
tetrodes = [1,2,3,4,5,8]
units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)

def acg(x,a,b,c,d,e,f,g,h):
    ans = c*np.exp(-(x-f)/a)-d*np.exp(-(x-f)/b)+h*np.exp(-(x-f)/g)+e
    ans[ans<0] = 0
    return ans

interneurons = []
pyramidal = []


pk_troughs = []
tau = []
colours = []
print(len(units))
for unit in units:

    # get avg pk_trough width
    # 32000 hz sample rate
    # 32 points / ms
    # each point = 0.0312 ms
    pk_trough = np.mean(unit['PeakToTroughPts']) / 20
    # take the channel with the largest amplitude


    y = unit['AutoCorr']
    x = unit['AutoCorrXAxisMsec']
    popt, pcov = curve_fit(acg, x, y, [20, 1, 30, 2, 0.5, 5, 1.5, 2],
                           bounds=([1, 0.1, 0, 0, -30, 0, 0.1, 0], [500, 50, 500, 15, 50, 20, 5, 100]), maxfev=5000)
    acg_tau_rise = popt[1]




    pk_troughs.append(pk_trough)
    tau.append(acg_tau_rise)


    #if pk_trough < 0.425:
    print(str(unit['tetrode']) + " " + str(unit['cluster']))
        #print(acg_tau_rise)
        #print(popt)
    print(pk_trough)

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    ax1.plot(unit['wf'].T)
    ax2.bar(x,y, color='black')
    plt.title(str(unit['tetrode']) + " " + str(unit['cluster']) + " pktr: " + str(pk_trough))
    #plt.plot(x, acg(x, *popt))
    #plt.show()


    #plt.show()

    #if unit['cluster'] in inter[unit['tetrode']]:
    #    colours.append(1)
    #else:
    #    colours.append(0)

    # for each unit, classify if interneuron or pyr
    # interneuron: peaktotrough < 0.425 ms (narrow interneuron), or peaktotrough > 0.425 ms and tau_rise > 6 ms
    # otherwise PYR
    """
    
    if pk_troughs < 0.425:
        interneurons.append(unit)
    else:
        popt, pcov = curve_fit(acg, x, y, [20, 1, 30, 2, 0.5, 5, 1.5,2],bounds=([1, 0.1, 0, 0, -30, 0,0.1,0], [500, 50, 500, 15, 50, 20,5,100]), maxfev=5000)
        acg_tau_rise = popt[1]
        if acg_tau_rise > 6:
            interneurons.append(unit)
        else:
            pyramidal.append(unit)
    """





plt.figure()
#ax = fig.add_subplot(projection='3d')
plt.scatter(pk_troughs, tau)#, c=colours)
plt.show()
#fig = plt.figure()
#plt.scatter(pk_troughs, tau, c=colours)
#plt.show()


# classify based on peaktotrough width and autocorrelogram

"""
plt.hist()
pk_troughs = np.array(pk_troughs) / 10
spikes = np.array(spikes) / 19292.2771880859

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(auto, pk_troughs, spikes, c=colours)
plt.show()
"""