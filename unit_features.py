import pandas as pd
import scipy.io as sio
import os
import unit_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import subjects
from scipy import signal



def acg(x, a, b, c, d, e, f, g, h):
    ans = c * np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b) + h * np.exp(-(x - f) / g) + e
    ans[ans < 0] = 0
    return ans

def features(units):
    pk_troughs = []
    tau = []
    for unit in units:
        pk_trough = np.mean(unit['PeakToTroughPts']) / 20
        # take the channel with the largest amplitude

        y = unit['AutoCorr']
        x = unit['AutoCorrXAxisMsec']
        popt, pcov = curve_fit(acg, x, y, [20, 1, 30, 2, 0.5, 5, 1.5, 2],
                               bounds=([1, 0.1, 0, 0, -30, 0, 0.1, 0], [500, 50, 500, 15, 50, 20, 5, 100]), maxfev=5000)
        acg_tau_rise = popt[1]



        pk_troughs.append(pk_trough)
        tau.append(acg_tau_rise)

    return pk_troughs, tau


ids = ['18-1', '36-1', '64-30']
tasks = ['hab', 'olm']
pyr_x = []
pyr_y = []
int_x = []
int_y = []
for id in ids:
    subject = subjects.subjects[id]

    data = subjects.load_data(subject, units=True)
    print(id)
    for task in tasks:

        pk_troughs, tau = features(data[task]['pyr_units'])
        pyr_x += pk_troughs
        pyr_y += tau

        pk_troughs, tau = features(data[task]['int_units'])
        int_x += pk_troughs
        int_y += tau


plt.figure()
# ax = fig.add_subplot(projection='3d')
plt.scatter(pyr_x, pyr_y, marker='.',c="black")
plt.scatter(int_x, int_y, marker='.',c="black")# , c=colours)
plt.show()
# fig = plt.figure()
# plt.scatter(pk_troughs, tau, c=colours)
# plt.show()

