from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.fftpack import rfft, irfft, fftfreq

xgb_clf2 = pkl.load(open("model_xgboostPLN1.pkl", "rb"))


def analyze(df):
    to_save = pd.DataFrame(columns=['VR', 'VS', 'VT', 'IR', 'IS', 'IT'])
    to_save = to_save.head(0)
    df_temp = df.copy()[["TIME", "VR", "VS", "VT", "IR", "IS", "IT"]]

    arr_channel = []
    arr_time = []
    arr_signal = []
    arr_fft1 = []

    for item in ["VR", "VS", "VT", "IR", "IS", "IT"]:
        arr_channel.append(item)
        arr_time.append(df_temp.TIME)
        arr_signal.append(df_temp[item])
        arr_fft1.append(plot_fft_segmented(df_temp.TIME, df_temp[item]))

    vmax = 0
    vmin = 0
    imax = 0
    imin = 0
    for k in range(0, len(arr_channel)):
        try:
            if "V" in arr_channel[k]:
                q75, q25 = np.percentile(arr_fft1[k], [75, 25])
                iqr = q75 - q25
                maxlim = np.mean(arr_fft1[k])+(3*iqr)
                if maxlim > vmax:
                    vmax = maxlim
                q75, q25 = np.percentile(arr_fft2[k], [75, 25])
                iqr = q75 - q25
                maxlim = np.mean(arr_fft2[k])+(3*iqr)
                if maxlim > vmax:
                    vmax = maxlim

            if "I" in arr_channel[k]:
                q75, q25 = np.percentile(arr_fft1[k], [75, 25])
                iqr = q75 - q25
                maxlim = np.mean(arr_fft1[k])+(3*iqr)
                if maxlim > imax:
                    imax = maxlim
                q75, q25 = np.percentile(arr_fft2[k], [75, 25])
                iqr = q75 - q25
                maxlim = np.mean(arr_fft2[k])+(3*iqr)
                if maxlim > imax:
                    imax = maxlim
        except:
            print("Error on analysis. check advancedanalysis.py.")

    for j in range(0, len(arr_channel)):
        hmax = 0
        hmin = 0
        if "V" in arr_channel[j]:
            hmax = vmax
            hmin = vmin
        if "I" in arr_channel[j]:
            hmax = imax
            hmin = imin
        print(to_save['IR'])
        if 'VR' in arr_channel[j] or 'VA' in arr_channel[j] or 'Tegangan R' in arr_channel[j] or ':V A' in arr_channel[j]:
            to_save['VR'] = arr_fft1[j]
    #         print(arr_channel[j])
        elif 'VS' in arr_channel[j] or 'VB' in arr_channel[j] or 'Tegangan S' in arr_channel[j] or ':V B' in arr_channel[j]:
            to_save['VS'] = arr_fft1[j]
    #         print(arr_channel[j])
        elif 'VT' in arr_channel[j] or 'VC' in arr_channel[j] or 'Tegangan T' in arr_channel[j] or ':V C' in arr_channel[j]:
            to_save['VT'] = arr_fft1[j]
    #         print(arr_channel[j])
        elif 'IR' in arr_channel[j] or 'IA' in arr_channel[j] or 'Arus R' in arr_channel[j] or ':I A' in arr_channel[j]:
            if to_save['IR'].isnull().values.any() == True:
                to_save['IR'] = arr_fft1[j]
    #           print(arr_channel[j])
        elif 'IS' in arr_channel[j] or 'IB' in arr_channel[j] or 'Arus S' in arr_channel[j] or ':I B' in arr_channel[j]:
            if to_save['IS'].isnull().values.any() == True:
                to_save['IS'] = arr_fft1[j]
    #           print(arr_channel[j])
        elif 'IT' in arr_channel[j] or 'IC' in arr_channel[j] or 'Arus T' in arr_channel[j] or ':I C' in arr_channel[j]:
            if to_save['IT'].isnull().values.any() == True:
                to_save['IT'] = arr_fft1[j]
    #           print(arr_channel[j])
    return run_ml(to_save)

def plot_fft_segmented(time, sgl):
    time = list(time)
    sgl = list(sgl)
    n_seg = 10
    segment_len = len(sgl)//n_seg
    time_arr = []
    sgl_arr = []

    for j in range(0, n_seg-1):
        time_arr.append(time[j*segment_len:((j+1)*segment_len)-1])
        sgl_arr.append(sgl[j*segment_len:((j+1)*segment_len)-1])

    time_arr.append(time[(n_seg-1)*segment_len:])
    sgl_arr.append(sgl[(n_seg-1)*segment_len:])

    result_arr = []
    for i in range(0, len(sgl_arr)):
        W = fftfreq(len(sgl_arr[i]), d=time_arr[i][1]-time_arr[i][0])
        f_signal = rfft(sgl_arr[i])

        Z = [x for _, x in sorted(
            zip(W, np.abs(f_signal)))][(len(W)//2)+1:][:170]

        result_arr.append(Z)
    return result_arr


def run_ml(df):
    try:
        testdata = []
        vr = list(np.array(list(df.VR), dtype=object).ravel()) + \
            list(np.array(list(df.IR), dtype=object).ravel())
        vs = list(np.array(list(df.VS), dtype=object).ravel()) + \
            list(np.array(list(df.IS), dtype=object).ravel())
        vt = list(np.array(list(df.VT), dtype=object).ravel()) + \
            list(np.array(list(df.IT), dtype=object).ravel())
        testdata.append(vr)
        testdata.append(vs)
        testdata.append(vt)
        col_list = ['c' + str(x) for x in range(0, 3400)]
        testdata = np.expand_dims(np.mean(np.array(testdata), axis=0), axis=0)
        print(testdata.shape)
        test_df = pd.DataFrame(testdata, columns=col_list)
        print(test_df.shape)
        # make predictions for test data
        y_pred = xgb_clf2.predict(test_df)
        predictions = [round(value) for value in y_pred]
        acc_pohon = accuracy_score([0], predictions)
    #   print("Accuracy: %.2f%%" % (accuracy * 100.0))
        acc_layang = accuracy_score([1], predictions)
    #   print("Accuracy: %.2f%%" % (accuracy * 100.0))
        acc_petir = accuracy_score([2], predictions)
    #   print("Accuracy: %.2f%%" % (accuracy * 100.0))
        if acc_pohon > acc_layang and acc_pohon > acc_petir:
            print("POHON")
            return "GANGGUAN POHON"
        elif acc_layang > acc_pohon and acc_layang > acc_petir:
            print("LAYANG-LAYANG")
            return "GANGGUAN LAYANG-LAYANG"
        elif acc_petir > acc_layang and acc_petir > acc_pohon:
            print("PETIR")
            return "GANGGUAN PETIR"
        else:
            print("Cannot determine result!")
            return "Cannot determine result!"
    except:
        return "Program Error!"
