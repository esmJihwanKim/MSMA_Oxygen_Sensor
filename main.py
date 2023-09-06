import glob
import pandas as pd
import process_signal as ps
import numpy as np

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from scipy import signal

path = glob.glob('../../Dataset/B2-2-9/*.DTA')
loop=0
allpeak=[]
saline_peak=[]

for file in path:
    # READ FILES
    datafile = open(file)
    lines = datafile.readlines()
    count = 84
    for i in lines[84:len(lines)]:
        lines[count] = i.split()
        count += 1
    df = pd.DataFrame(lines[85:len(lines)], columns=[0, 1, 'voltage', 'current', 4, 5, 6, 7, 8, 9, 10])
    df = df[['voltage', 'current']]
    df['voltage'] = df['voltage'].astype(float)
    df['current'] = df['current'].astype(float)
    df['current'] = df['current'].apply(lambda x: x * 1000000)
    x = df['voltage']
    y = df['current']
    list_x = list(x)
    list_y = list(y)

    # COMPUTE DERIVATIVE FOR RAW SIGNAL
    raw_d1 = []
    for i in range(len(list_x)-1):
        raw_d1.append((list_y[i + 1] - list_y[i]) / (list_x[i + 1] - list_x[i]))
    raw_d2 = []
    for i in range(len(raw_d1) - 1):
        raw_d2.append((raw_d1[i + 1] - raw_d1[i]) / (list_x[i + 1] - list_x[i]))

    # Pad the array with final values,
    raw_d1.append(raw_d1[len(raw_d1) - 1])
    raw_d2.append(raw_d2[len(raw_d2) - 1])
    raw_d2.append(raw_d2[len(raw_d2) - 1])

    # APPLY SAVITSKY-GOLAY TO RAW SIGNAL
    savgol_list = ps.optimize_savgol(list_y)
    opt_window_size1 = savgol_list[1]
    opt_order1 = savgol_list[2]
    list_y_filtered = signal.savgol_filter(list_y, 53, 3), # savgol_filter(data, window size, order of polynomial)

    # TODO: APPLY FFT-LPF TO RAW SIGNAL
    signal_frequency = []
    signal_amplitude = []
    # signal_frequency, signal_amplitude = ps.fft(list_y)
    # plt.plot(signal_frequency, signal_amplitude)
    # plt.show

    # COMPUTE FILTERED DERIVATIVE
    filtered_d1 = []
    for i in range(len(x) - 1):
        filtered_d1.append((list_y_filtered[0][i + 1] - list_y_filtered[0][i]) / (list_x[i + 1] - list_x[i]))
    filtered_d1.append(filtered_d1[len(filtered_d1) - 1])

    # EXTRACT AXIS LABEL FROM FILE NAMES
    label_text = file.split("VISHAL-")[1]
    label_text = label_text.split((".DTA"))[0]

    # PLOT (for algorithm design purpose)
    # comment out when testing peak detection
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(x, list_y)
    ax[0, 0].set_title(label_text + ":::raw y")
    ax[0, 1].plot(x, list_y_filtered[0])
    ax[0, 1].set_title(label_text + ":::filtered y")
    ax[1, 0].plot(x, raw_d1)
    ax[1, 0].set_title(label_text + ":::raw d1")
    ax[1, 1].plot(x, filtered_d1)
    ax[1, 1].set_title(label_text + ":::filtered d1")
    plt.show()
    # plt.savefig(label_text+'.png')

    # TODO: save the plots in disk

    # PEAK FINDING - Trickle Down Algorithm
    # @@
    # @@
    # @@
    peak = []
    # TODO: CONDITION: peaks exist before exponential increase

    # TODO: CONDITION: voltage range from 0.35 - 0.65
    # sort out the data points that are in the voltage range

    # TODO: CONDITION: derivative range between 0-0.5

    # TODO: CONDITION: peaks after positive gradient







"""
    for p in saline_peak:
        plt.plot(p[0], p[1], marker="x", markersize=4, markeredgecolor="red", markerfacecolor="green")
"""