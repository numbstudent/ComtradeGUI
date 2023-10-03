from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')

import numpy as np
import math
import os
# drive.mount('/content/gdrive')

plt.rcParams["figure.figsize"] = (15, 3)


def moving_average(arr):
    window_size = 50

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)


def get_fs3(signal):
    inslope = False
    index = 0
    offset = 0

    original_len = len(signal)

    cur_wave = 1
    wanted_wave = 4

    hill_mode = True
    if float(signal[1]) - float(signal[0]) < 0:
        hill_mode = False

    for idx in range(1, len(signal)):
        if hill_mode == True:
            if inslope == False and signal[idx] < signal[idx-1] and signal[idx] < 0:
                inslope = True
            if signal[idx] > signal[0] and inslope == True and signal[idx] > signal[idx-1]:
                index = idx-1
                if cur_wave < wanted_wave:
                    inslope = False
                    cur_wave = cur_wave + 1
                else:
                    break
        else:
            if inslope == False and signal[idx] > signal[idx-1] and signal[idx] > 0:
                inslope = True
            if signal[idx] < signal[0] and inslope == True and signal[idx] < signal[idx-1]:
                index = idx-1
                if cur_wave < wanted_wave:
                    inslope = False
                    cur_wave = cur_wave + 1
                else:
                    break

    index = index + offset  # experimental
    index = math.ceil(index/4)
    return index


def get_rms(df_time, data):
    s_rate = get_fs3(data)

    Fs = s_rate//4
    Fs = s_rate * 5

    def RMS(x): return np.sqrt(np.mean(x**2))
    # print(len(data)-Fs)
    sample = np.arange(len(data)-Fs)
    RMS_of_sample = []
    for ns in sample:
        # here you can apply the frequency window for the sample
        RMS_of_sample.append(RMS(data[ns:ns+Fs]))
    return np.array(RMS_of_sample)


def get_stdev(df_time, data):
    s_rate = get_fs3(data)

    Fs = s_rate//4
    Fs = s_rate * 5

    def stdev(x): return np.std(x)
    # print(len(data)-Fs)
    sample = np.arange(len(data)-Fs)
    stdev_of_sample = []
    for ns in sample:
        # here you can apply the frequency window for the sample
        stdev_of_sample.append(stdev(data[ns:ns+Fs]))
    return np.array(stdev_of_sample)


def get_normal_2(signal):
    inslope = False
    index = 0
    offset = 0

    original_len = len(signal)

    hill_mode = True
    if float(signal[1]) - float(signal[0]) < 0:
        hill_mode = False

    step = 0
    n_wave = 2

    for idx in range(1, len(signal)):
        if hill_mode == True:
            if inslope == False and signal[idx] < signal[idx-1] and signal[idx] < 0:
                inslope = True
            if signal[idx] > signal[0] and inslope == True and signal[idx] > signal[idx-1]:
                index = idx-1
                if step < n_wave:
                    step = step + 1
                    inslope = False
                else:
                    break
        else:
            if inslope == False and signal[idx] > signal[idx-1] and signal[idx] > 0:
                inslope = True
            if signal[idx] < signal[0] and inslope == True and signal[idx] < signal[idx-1]:
                index = idx-1
                if step < n_wave:
                    step = step + 1
                    inslope = False
                else:
                    break

    index = index + offset  # experimental
    sample_wave = list(signal[:index])
    normal_wave_5 = sample_wave+sample_wave+sample_wave+sample_wave+sample_wave
    normal_wave_10 = normal_wave_5+normal_wave_5
    normal_wave = normal_wave_10

    while len(normal_wave) < original_len:
        normal_wave = normal_wave + normal_wave_10
    normal_wave = normal_wave[:original_len]
    return normal_wave


def analyze(df):
    result = []
    try:
        len_df = len(df.TIME)

        df_temp = df.copy()[["TIME", "VR", "VS", "VT", "IR", "IS", "IT"]]
        df_temp["N_VR"] = get_normal_2(df["VR"])
        df_temp["N_VS"] = get_normal_2(df["VS"])
        df_temp["N_VT"] = get_normal_2(df["VT"])
        df_temp["N_IR"] = get_normal_2(df["IR"])
        df_temp["N_IS"] = get_normal_2(df["IS"])
        df_temp["N_IT"] = get_normal_2(df["IT"])

        for i in ["VR", "VS", "VT", "IR", "IS", "IT"]:

            fs = df_temp[i]
            ns = df_temp["N_"+i]

            
            # os.remove(os.path.join("./static", i+".png"))
            try:
                os.remove("static/"+i+".png")
            except:
                None

            faulty_found = False
            datasplit = [0, int(len(fs)//5), int(len(fs)*2//5),
                         int(len(fs)*3//5), int(len(fs)*4//5), int(len(fs))]
            datasplit = [0]
            nsplit = 50
            for k in range(1,nsplit):
                datasplit.append(int(len(fs)*k//nsplit))
            datasplit.append(int(len(fs)))
            x1 = []
            x2 = []
            for j in range(0, len(datasplit)-1):
                stdn = np.std(ns[datasplit[j]:datasplit[j+1]])
                stdf = np.std(fs[datasplit[j]:datasplit[j+1]])

                ratio = 0
                if stdf < stdn:
                    ratio = stdn/stdf
                else:
                    ratio = stdf/stdn
                # print("ratio "+i+": "+str(ratio))
                if ratio > 1.1:
                    x1.append(datasplit[j])
                    x2.append(datasplit[j+1]-1)
                    faulty_found = True
            if faulty_found:
                result.append(i+" is faulty!")
                # print("Line "+i+" is faulty!")
            else:
                result.append(i+" is normal.")
                # print("Line "+i+" is normal.")
            try:
                maxh = np.max(fs)
                minh = np.abs(np.min(fs))
                if maxh > minh:
                    maxh = minh
                plt.plot(df.TIME, fs, label=i)
                if len(x1) > 0:
                    for ii in range(0,len(x1)):
                        # print(x1[ii],x2[ii])
                        plt.fill_betweenx(x1=df.TIME[x1[ii]],
                                        x2=df.TIME[x2[ii]], y=[-1*maxh, maxh], alpha=0.1, color='orange')
                # plt.plot(df.TIME, ns, label='N_'+i)
                plt.legend()
                # plt.show()
                plt.savefig("static/"+i+".png")
                plt.clf()
            except:
                print("Unable to save figures.")
    except x:
        result.append("Error found when running.")
        # print("Error found when running.")

    return result
