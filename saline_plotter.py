import glob
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

import process_signal as ps
path = './Dataset/B2-2-10/*.DTA'
loop = 0
allpeak = []
saline_peak = []


for data in glob(path):
    if("fish") not in data:
        loop += 1
        datafile = open(data)
        lines = datafile.readlines()
        count = 84
        # Read files
        for i in lines[84:len(lines)]:
            lines[count] = i.split()
            count += 1
        df = pd.DataFrame(lines[85:len(lines)], columns=[0,1,'voltage','current', 4, 5, 6, 7, 8, 9, 10])
        df = df[['voltage', 'current']]
        df['voltage'] = df['voltage'].astype(float)
        df['current'] = df['current'].astype(float)
        df['current'] = df['current'].apply(lambda x: x * 1000000)
        label1 = data.split("VISHAL-")[1]
        label1 = label1.split(".DTA")[0]
        x = df['voltage']
        if len(x) == 0: continue
        y = df['current']
        dev = []
        list_x = list(x)
        list_y = list(y)

        # PLOT DERIVATIVE
        savgol_list = ps.optimize_savgol(list_y)
        opt_window_size1 = savgol_list[1]
        opt_order1 = savgol_list[2]
        d1 = []
        d2 = []
        for i in range(len(list_x)-1):
            d1.append((list_y[i+1]-list_y[i]) / list_x[i+1]-list_x[i])
        for i in range(len(d1) - 1):
            d2.append((d1[i+1] - d1[i]) / (list_x[i+1] - list_x[i]))
        sumd2 = 0
        for i in d2:
            sumd2 += abs(i) / len(d2)
        res = True in (abs(ele) > 1000 for ele in d2)
        if sumd2>150 or res:
            # savgol_filter(x,y,z)
            # x : data list
            # y : window size for filtering
            # z : order of fitted polynomial
            list_ys = signal.savgol_filter(list_y, 53, 3)

        for i in range(len(x) - 1):
            dev.append((list_ys[0][i+1] - list_ys[0][i]) / (list_x[i+1] - list_x[i]))

        list_x = list(x)
        list_y = list(y)
        list_xo = list_x
        list_yo = list_y
        number = len(list_x)
        list_x.pop()
        list_x = list(x)
        dev = []

        # GET DERIVATIVE
        for i in range(len(x) - 1):
            dev.append((list_y[i+1] - list_y[i]) / (list_x[i+1] - list_x[i]))

        savgol_list = ps.optimize_savgol(dev)
        opt_window_size = savgol_list[1]
        opt_order = savgol_list[2]
        devs = signal.savgol_filter(dev, 53, 3)
        list_x.pop()

        # PEAK FINDER
        peak = []
        list_x1 = list_x
        list_x = np.asarray(list_x)
        devs = np.squeeze(np.asarray(devs))
        devs1 = devs[40:]
        lastpeak = 0
        lastindex = 0
        devs[0] = 1
        for index, ele in enumerate(devs):
            if index >= 4 and index <= len(devs) - 11:
                if sum(1 for x in dev[index-10:index+1] if x > 0) >= 6 and dev[index] > 0.2 and sum((element) for )
                    and abs(list_x[index]-lastpeak) > sum(abs(x)) for x in devs[index:index+10]):
                    peakindex = index
                    if