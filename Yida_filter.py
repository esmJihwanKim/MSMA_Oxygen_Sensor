import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
from scipy import signal
from numpy import diff
from statistics import mean
import copy

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

def optimize_savgol(dev):
    error=100
    opt_window_size=10
    opt_order=1
    for i in range(11,100):
        for j in range(1,10):
            devs=signal.savgol_filter(dev,
                           i, # window size used for filtering
                           j), # order of fitted polynomial
            devs = np.squeeze(np.asarray(devs))
            error_try=mean_squared_error(dev,devs)
            if error>error_try:
                if np.mean(np.absolute(diff(devs))) < 0.06:
                    error=error_try
                    opt_window_size=i
                    opt_order=j
    return [error,opt_window_size,opt_order]

def optimize_savgol_orig(dev):
    error=100
    opt_window_size=10
    opt_order=1
    for i in range(11,100):
        for j in range(1,10):
            devs=signal.savgol_filter(dev,
                           i, # window size used for filtering
                           j), # order of fitted polynomial
            devs = np.squeeze(np.asarray(devs))
            error_try=mean_squared_error(dev,devs)
            if error>error_try:
                if np.mean(np.absolute(diff(devs))) < 0.0001:
                    error=error_try
                    opt_window_size=i
                    opt_order=j
    return [error,opt_window_size,opt_order]


def CheckForLess(list1, val):
    for x in list1:
        if val < x:
            return False
    return True


# for the smooth data overlaying the original dev plot
path='/Users/yidawang/Desktop/peak/B2-2-10/*.DTA'
loop=0
allpeak=[]
saline_peak=[]
'''read files'''
for data in glob.glob(path):
    if "fish" not in data:
        loop+=1
        datafile = open(data)
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
        label1 = data.split("VISHAL-")[1]
        label1 = label1.split(".DTA")[0]
        x = df['voltage']
        if len(x) == 0: continue
        y = df['current']
        dev = []
        list_x = list(x)
        list_y = list(y)

        '''PLOT DERIVATIVE'''
        savgol_list = optimize_savgol(list_y)
        opt_window_size1 = savgol_list[1]
        opt_order1 = savgol_list[2]
        d1=[]
        for i in range(len(list_x)-1):
            d1.append((list_y[i + 1] - list_y[i]) / (list_x[i + 1] - list_x[i]))
        d2=[]
        for i in range(len(d1) - 1):
            d2.append((d1[i + 1] - d1[i]) / (list_x[i + 1] - list_x[i]))
        sumd2=0
        for i in d2:
            sumd2+=abs(i)/len(d2)
        res = True in (abs(ele) > 1000 for ele in d2)
        # if sumd2>150 or res:
        list_ys=signal.savgol_filter(list_y,
                               53, # window size used for filtering
                               3), # order of fitted polynomial
        # else: list_ys=[list_y]

        for i in range(len(x) - 1):
            dev.append((list_ys[0][i + 1] - list_ys[0][i]) / (list_x[i + 1] - list_x[i]))

        list_x = list(x)
        list_y = list(y)
        list_xo=list_x
        list_yo=list_y
        number=len(list_x)
        list_x.pop()
        list_x = list(x)
        dev = []
        '''GET DERIVATIVE'''
        for i in range(len(x)-1):
            dev.append((list_y[i + 1] - list_y[i]) / (list_x[i + 1] - list_x[i]))

        savgol_list=optimize_savgol(dev)
        opt_window_size=savgol_list[1]
        opt_order=savgol_list[2]
        devs=signal.savgol_filter(dev,
                               53, # window size used for filtering
                               3), # order of fitted polynomial
        # devs=signal.savgol_filter(dev,
        #                        53, # window size used for filtering
        #                        3), # order of fitted polynomial
        list_x.pop()

        '''PEAK FINDER'''
        peak=[]
        list_x1=list_x
        list_x=np.asarray(list_x)
        devs=np.squeeze(np.asarray(devs))
        devs1=devs[40:]
        lastpeak=0
        lastindex=0
        devs[0]=1
        for index, ele in enumerate(devs):
            if index >= 4 and index <= len(devs) - 11:
                if sum(1 for x in dev[index-10:index+1] if x > 0)>=6 and dev[index]<0.2 and sum((element) for element in dev[index:index+6])<2 \
                    and abs(list_x[index]-lastpeak)>0.001 and max(devs[lastindex:index])>0.05 and \
                        9*abs(devs[lastindex])>sum(abs(x) for x in devs[index:index+10]):
                    peakindex=index
                    if lastpeak!=0:
                        peak.remove([list_x[lastindex], list_y[lastindex]])
                    peak.append([list_x[index], list_y[index]])
                    if lastpeak!=0: saline_peak.remove([lastpeak, list_y[lastindex]])
                    saline_peak.append([list_x[index],list_y[index]])
                    if lastpeak!=0: allpeak.remove([lastpeak, list_y[lastindex]])
                    print(dev[index],devs[index],list_x[index],list_y[index])
                    allpeak.append([list_x[index], list_y[index]])
                    lastpeak = list_x[index]
                    lastindex = index

        if peak==[]:
            lastindex = 0
            lastpeak = 10
            minres = 10
            minindex = 0
            for i, ele in enumerate(devs):
                if i >= 5 and i <= len(devs) - 7:
                    if devs[i - 1] > devs[i] and devs[i - 2] > devs[i] and devs[i - 3] > devs[i] \
                            and devs[i + 1] > devs[i] and devs[i + 2] > devs[i] and devs[i + 3] > devs[i] \
                            and devs[i + 4] > devs[i] and devs[i + 5] > devs[i] and devs[i - 4] > devs[i] and devs[
                        i - 5] > devs[i] \
                            and sum(devs[i - 15:i]) / 15 > sum(devs[i - 2:i + 3]) / 5 and sum(
                        devs[i + 1:i + 16]) / 15 > sum(devs[i - 2:i + 3]) / 5 \
                            and devs[i] > 0 and devs[i]<devs[lastindex]:
                        res = devs[i]
                        minres = res
                        if lastindex != 0: peak.remove([list_x[lastindex], list_y[lastindex]])
                        if lastindex != 0: saline_peak.remove([lastpeak, list_y[lastindex]])
                        if lastindex != 0: allpeak.remove([lastpeak, list_y[lastindex]])
                        peak.append([list_x[i], list_y[i]])
                        allpeak.append([list_x[i], list_y[i]])
                        saline_peak.append([list_x[i], list_y[i]])
                        lastpeak = list_x[i]
                        lastindex = i
        print(label1 + ":")
        print('error: ' + str(mean_squared_error(dev, devs)))
        print('saline peak: ' + str(peak))
        devd2 = []
        for i in range(len(devs) - 1):
            devd2.append((devs[i + 1] - devs[i]) / (list_x1[i + 1] - list_x1[i]))
        devd2 = signal.savgol_filter(devd2,
                                     53,  # window size used for filtering
                                     3),  # order of fitted polynomial
        devd2 = list(devd2)[0]
        fig = plt.figure(label1)
        plt.gca().invert_xaxis()
        plt.subplot(2, 1, 1)
        plt.plot(list_x, dev, label=label1)
        plt.gca().invert_xaxis()
        plt.subplot(2, 1, 2)
        list_x = list(list_x)
        list_x.pop()
        plt.plot(list_x, devd2)
        plt.gca().invert_xaxis()



loop=0
fish_peak=[]
'''read files'''
for data in glob.glob(path):
    if "fish" in data:
        loop+=1
        datafile = open(data)
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
        label1 = data.split("VISHAL-")[1]
        label1 = label1.split(".DTA")[0]
        x = df['voltage']
        if len(x) == 0: continue
        y = df['current']
        dev = []
        list_x = list(x)
        list_y = list(y)

        '''PLOT DERIVATIVE'''
        savgol_list = optimize_savgol(list_y)
        opt_window_size1 = savgol_list[1]
        opt_order1 = savgol_list[2]
        d1=[]
        for i in range(len(list_x)-1):
            d1.append((list_y[i + 1] - list_y[i]) / (list_x[i + 1] - list_x[i]))
        d2=[]
        for i in range(len(d1) - 1):
            d2.append((d1[i + 1] - d1[i]) / (list_x[i + 1] - list_x[i]))
        sumd2=0
        for i in d2:
            sumd2+=abs(i)/len(d2)
        res = True in (abs(ele) > 1000 for ele in d2)
        if sumd2>150 or res:
            list_ys=signal.savgol_filter(list_y,
                                   53, # window size used for filtering
                                   3), # order of fitted polynomial
        else: list_ys=[list_y]

        for i in range(len(x) - 1):
            dev.append((list_ys[0][i + 1] - list_ys[0][i]) / (list_x[i + 1] - list_x[i]))

        list_x = list(x)
        list_y = list(y)
        list_xo=list_x
        list_yo=list_y
        number=len(list_x)
        list_x.pop()
        list_x = list(x)
        dev = []
        '''GET DERIVATIVE'''
        for i in range(len(x)-1):
            dev.append((list_ys[0][i + 1] - list_ys[0][i]) / (list_x[i + 1] - list_x[i]))

        savgol_list=optimize_savgol(dev)
        opt_window_size=savgol_list[1]
        opt_order=savgol_list[2]
        devs=signal.savgol_filter(dev,
                               opt_window_size, # window size used for filtering
                               opt_order), # order of fitted polynomial
        # devs=signal.savgol_filter(dev,
        #                        53, # window size used for filtering
        #                        3), # order of fitted polynomial
        list_x.pop()

        '''PEAK FINDER'''
        peak = []
        list_x1 = list_x
        list_x = np.asarray(list_x)
        devs = np.squeeze(np.asarray(devs))
        devs1 = devs[40:]
        lastpeak = 0
        lastindex = 0
        for index, ele in enumerate(devs):
            if index >= 4 and index <= len(devs) - 11:
                if devs[index] > 0 and (devs[index + 1]) < 0.1 and devs[index + 3] < 0.1 and devs[index + 5] < 0.1 and \
                        devs[index + 10] < 0.1 \
                        and abs(list_x[index] - lastpeak) > 0.05 and max(devs[lastindex:index]) > 0.15:
                    lastpeak = list_x[index]
                    lastindex = index
                    peakindex = index
                    peak.append([list_x[index], list_y[index]])
                    fish_peak.append([list_x[index], list_y[index]])
        minres = 10
        minindex = 0
        for i, ele in enumerate(devs):
            if i >= 5 and i <= len(devs) - 7:
                if devs[i - 1] > devs[i] and devs[i - 2] > devs[i] and devs[i - 3] > devs[i] \
                        and devs[i + 1] > devs[i] and devs[i + 2] > devs[i] and devs[i + 3] > devs[i] \
                        and devs[i + 4] > devs[i] and devs[i + 5] > devs[i] and devs[i - 4] > devs[i] and devs[i - 5] > \
                        devs[i] \
                        and sum(devs[i - 15:i]) / 15 > sum(devs[i - 2:i + 3]) / 5 and sum(
                    devs[i + 1:i + 16]) / 15 > sum(devs[i - 2:i + 3]) / 5 \
                        and devs[i] > 0:
                    res = devs[i]
                    minres = res
                    peak.append([list_x[i], list_y[i]])
                    fish_peak.append([list_x[i], list_y[i]])

        devd2 = []
        for i in range(len(devs) - 1):
            devd2.append((devs[i + 1] - devs[i]) / (list_x1[i + 1] - list_x1[i]))
        devd2 = signal.savgol_filter(devd2,
                                     53,  # window size used for filtering
                                     3),  # order of fitted polynomial
        devd2 = list(devd2)[0]
        i = 30
        while (i <= len(devs) - 22):
            if (10 * (mean(devd2[i - 20:i]))) > (mean(devd2[i:i + 20])) and CheckForLess(devd2[i - 20:i],
                                                                                         0) and CheckForLess(
                    devd2[i + 20:i], 0):
                peak.append([list_x[i], list_y[i]])
                fish_peak.append([list_x[i], list_y[i]])
                i += 10
            if ((mean(devd2[i - 20:i])) - 15) > (mean(devd2[i:i + 20])) and CheckForLess(devd2[i - 20:i],
                                                                                         0) and CheckForLess(
                    devd2[i + 20:i], 0):
                peak.append([list_x[i], list_y[i]])
                fish_peak.append([list_x[i], list_y[i]])
                i += 10
            i += 1

        for i in range(21, len(devd2) - 22):
            if devd2[i] > max(devd2[i + 1:i + 21]) and devd2[i] > max(devd2[i - 21:i]) \
                    and max(abs(devd2[i - 21:i] - devd2[i])) > 0.1 and max(abs(devd2[i + 1:i + 21] - devd2[i])) > 0.1:
                peak.append([list_x[i], list_y[i]])
                fish_peak.append([list_x[i], list_y[i]])
            if devd2[i] < min(devd2[i + 1:i + 21]) and devd2[i] < min(devd2[i - 21:i]):
                peak.append([list_x[i], list_y[i]])
                fish_peak.append([list_x[i], list_y[i]])
        print(label1+":")
        print('error: '+str(mean_squared_error(dev, devs)))
        print('fish peak: '+str(peak))

first_scan_peak_current=[0,0]
# for sp in saline_peak:
#     if sp[1]<first_scan_peak_current[1]:
#         first_scan_peak_current=sp
# saline_peak_copy=copy.deepcopy(saline_peak)
# saline_peak_copy.remove(first_scan_peak_current)

# x_values = [coord[0] for coord in saline_peak_copy]
# y_values = [coord[1] for coord in saline_peak_copy]
#
# average_x = mean(x_values)
# average_y = mean(y_values)
# window_voltage=[average_x/2, average_x]
# print(window_voltage)

fish_peak_plot=[]
for fish in fish_peak:
    item=fish[0]
    if item>-0.2 or item<-0.7:
        continue
    else: fish_peak_plot.append(fish)


fig = plt.figure('all_original')
for data in glob.glob(path):
    loop+=1
    datafile = open(data)
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
    label1 = data.split("VISHAL-")[1]
    label1 = label1.split(".DTA")[0]
    x = df['voltage']
    y = df['current']
    dev = []
    list_x = list(x)
    list_y = list(y)
    plt.plot(list_x, list_y, label=label1)
for p in saline_peak:
    plt.plot(p[0], p[1], marker="x", markersize=4, markeredgecolor="red", markerfacecolor="green")
for p in fish_peak_plot:
    plt.plot(p[0], p[1], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.grid()
plt.legend()

plt.show()
