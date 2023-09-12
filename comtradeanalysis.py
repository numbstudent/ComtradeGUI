import pip
import os
from os import listdir
from os.path import isfile, join
import numpy as np

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

import_or_install('comtrade')

import comtrade

def read_comtrade_channels():
  # fname = 'upload/'+'toanalyze'
  mypath = './'
  onlyfiles = [mypath+'/'+os.path.splitext(f)[:-1][0] for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[-1].upper() == 'CFG']
  onlyfiles = list(set(onlyfiles))
  
  if len(onlyfiles) > 0:
    fname = onlyfiles[0]
    rec = comtrade.load(fname+".cfg", fname+".dat")
    print("Trigger time = {}s".format(rec.trigger_time))
    # try:
    #   rec = comtrade.load(fname+".cfg", fname+".dat")
    # except:
    #   rec = comtrade.load(fname+".CFG", fname+".DAT")
    # print("Trigger time = {}s".format(rec.trigger_time))
    # print("Channels:")
    return rec.analog_channel_ids

def read_comtrade_channels(channels):
  # fname = 'upload/'+'toanalyze'
  mypath = './'
  onlyfiles = [mypath+'/'+os.path.splitext(f)[:-1][0] for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[-1].upper() == 'CFG']
  onlyfiles = list(set(onlyfiles))
  
  if len(onlyfiles) > 0:
    fname = onlyfiles[0]
    rec = comtrade.load(fname+".cfg", fname+".dat")
    print("Trigger time = {}s".format(rec.trigger_time))
    # vrchannel = rec.analog_channel_ids[int(vr)]
    # vschannel = rec.analog_channel_ids[int(vs)]
    # vtchannel = rec.analog_channel_ids[int(vt)]
    # vrdata = str(rec.analog[int(vr)][0])
    # return "Channel to analyze: "+vrdata

    len_df = len(rec.time)

    df['N_VR'] = get_normal_2(df.VR)[:len_df]
    df['N_VS'] = get_normal_2(df.VS)[:len_df]
    df['N_VT'] = get_normal_2(df.VT)[:len_df]
    df['N_VAVG'] = get_normal_2(df.VAVG)[:len_df]

    df_temp = df.copy()[["VR","N_VR","VS","N_VS","VT","N_VT","VAVG","N_VAVG"]]

    corr = df_temp.corr()
    corr_VR = corr["VR"]["N_VR"]
    corr_VS = corr["VS"]["N_VS"]
    corr_VT = corr["VT"]["N_VT"]
    # sns.heatmap(corr)
    # plt.show()
    faulty = ""
    if corr_VR < corr_VS and corr_VR < corr_VT:
      faulty = "VR"
    elif corr_VS < corr_VR and corr_VS < corr_VT:
      faulty = "VS"
    elif corr_VT < corr_VR and corr_VT < corr_VS:
      faulty = "VT"

    df2 = pd.DataFrame(columns=["VR","VS","VT","VAVG","N_VR","N_VS","N_VT","N_VAVG"])
    # for i in [faulty]:
    for i in ["VR","VS","VT","VAVG"]:
      # df_temp = df.copy()[[faulty]]
      # df_temp = df_temp.set_axis(['V'], axis=1)
      # labels = list(np.repeat(1, len(df_temp)))
      # df_temp['labels'] = labels
      # df_new = pd.concat([df_new, df_temp])


      # df_temp = df.copy()[['N_'+faulty]]
      # df_temp = df_temp.set_axis(['V'], axis=1)
      # labels = list(np.repeat(0, len(df_temp)))
      # df_temp['labels'] = labels
      # df_new = pd.concat([df_new, df_temp])
      # # df_temp = df_temp.rename(columns={'N_VR': 'VR', 'N_VS': 'VS', 'N_VT': 'VT', 'N_VAVG': 'VAVG'})
      # # labels = list(np.repeat(0, len(df_temp)))
      # # df_temp['labels'] = labels
      # # df_new = pd.concat([df_new, df_temp])

      scaling_factor = get_scaling_factor(list(df['N_'+faulty]))

      # df2[i] = get_plot_freq(df.TIME,np.array(df[i])*scaling_factor)
      # df2['N_'+i] = get_plot_freq(df.TIME,np.array(df['N_'+i])*scaling_factor)

      # plot_freq(df.TIME,np.array(df[faulty])*scaling_factor)
      # plot_freq(df.TIME,np.array(df['N_'+faulty])*scaling_factor)
      # plt.plot(df.TIME,np.array(df[faulty])*scaling_factor)
      # plt.plot(df.TIME,np.array(df['N_'+faulty])*scaling_factor)
      # plt.show()
      # plot_rms(df.TIME,df[faulty]*scaling_factor)
      # plot_rms(df.TIME,df['N_'+faulty]*scaling_factor)
      # plt.show()

      df2[i] = get_rms(df.TIME, np.array(df[i])*scaling_factor)[:3000]
      df2['N_'+i] = get_rms(df.TIME, np.array(df['N_'+i])*scaling_factor)[:3000]

    newname = fname.split(".")
    newname.pop()
    newname = "".join(newname)+"_rms.csv"
    # print(newname)
    print(df2)
    df2.to_csv(newname)


    # for i in channels:
      
      # observed_signal = rec.analog[i]
      # normal_wave = get_normal_2(observed_signal)
      # # print(normal_wave)
      # max_val = get_maximum(normal_wave)
      # cur_max = 100
      # scaling_factor = cur_max/max_val
      # max_len = 0
      # print(len(rec.time),len(normal_wave))
      # if len(normal_wave) > len(rec.time):
      #   max_len = len(observed_signal)
      # else:
      #   max_len = len(normal_wave)
      # print(max_len)

      # final_time = scaling_factor*signal.resample(df.TIME[:max_len], 3000)
      # final_normal = scaling_factor*signal.resample(normal_wave[:max_len], 3000)
      # final_original = scaling_factor*signal.resample(observed_signal[:max_len], 3000)

  return "meong"

def get_normal_2(signal):
  inslope = False
  index = 0
  offset = 0

  hill_mode = True
  if float(signal[1]) - float(signal[0]) < 0:
    hill_mode = False

  for idx in range(1,len(signal)):
    if hill_mode == True:
      if inslope == False and signal[idx] < signal[idx-1] and signal[idx] < 0:
        inslope = True
      if signal[idx] > signal[0] and inslope == True and signal[idx] > signal[idx-1]:
        index = idx-1
        break
    else:
      if inslope == False and signal[idx] > signal[idx-1] and signal[idx] > 0:
        inslope = True
      if signal[idx] < signal[0] and inslope == True and signal[idx] < signal[idx-1]:
        index = idx-1
        break

  index = index + offset #experimental
  sample_wave = list(signal[:index])
  normal_wave_5 = sample_wave+sample_wave+sample_wave+sample_wave+sample_wave
  normal_wave_10 = normal_wave_5+normal_wave_5
  normal_wave = normal_wave_10+normal_wave_10+normal_wave_10+normal_wave_10+normal_wave_10+normal_wave_10
  return normal_wave

def get_maximum(arr):
  x = sorted(list(arr), reverse=True)
  # maxval = sum(x[:20])/20
  # return maxval
  x2 = np.array(x[:1500])
  Q1 = np.quantile(x2, 0.25)
  Q3 = np.quantile(x2, 0.75)
  IQR = Q3 - Q1
  Lower_Fence = Q1 - (1.5 * IQR) * -1
  Upper_Fence = Q3 + (1.5 * IQR)
  if Lower_Fence < Upper_Fence:
    return Lower_Fence
  else:
    return Upper_Fence