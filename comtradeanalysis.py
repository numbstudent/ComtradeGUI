import pip
import os
from os import listdir
from os.path import isfile, join

import standardanalysis as sa
import advancedanalysis as adva

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

import_or_install('comtrade')
import_or_install('numpy')
import_or_install('pandas')
import_or_install('seaborn')
import_or_install('matplotlib')

import comtrade
import numpy as np
import pandas as pd
import math
import seaborn as sns
from matplotlib import pyplot as plt

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

def standard_analyze(channels):
  mypath = './'
  onlyfiles = [mypath+'/'+os.path.splitext(f)[:-1][0] for f in listdir(
      mypath) if isfile(join(mypath, f)) and f.split('.')[-1].upper() == 'CFG']
  onlyfiles = list(set(onlyfiles))
  df_new = pd.DataFrame(columns=["V"])
  faulty = ""
  if len(onlyfiles) > 0:
    fname = onlyfiles[0]
    rec = comtrade.load(fname+".cfg", fname+".dat")
    print("Trigger time = {}s".format(rec.trigger_time))

    len_df = len(rec.time)
    df = pd.DataFrame(list(rec.time), columns=['TIME'])

    df['VR'] = rec.analog[channels[0]][:len_df]
    df['VS'] = rec.analog[channels[1]][:len_df]
    df['VT'] = rec.analog[channels[2]][:len_df]
    df['IR'] = rec.analog[channels[3]][:len_df]
    df['IS'] = rec.analog[channels[4]][:len_df]
    df['IT'] = rec.analog[channels[5]][:len_df]

    result,error = sa.analyze(df)
    # if error == False:
    ml_result = adva.analyze(df)
    # else:
    #   ml_result = "File is not compatible"
    print("capek",ml_result)
    return result,ml_result
    # print(result)

def analyze(channels):
  # fname = 'upload/'+'toanalyze'
  mypath = './'
  onlyfiles = [mypath+'/'+os.path.splitext(f)[:-1][0] for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[-1].upper() == 'CFG']
  onlyfiles = list(set(onlyfiles))
  df_new = pd.DataFrame(columns=["V"])
  faulty = ""

  if len(onlyfiles) > 0:
    fname = onlyfiles[0]
    rec = comtrade.load(fname+".cfg", fname+".dat")
    print("Trigger time = {}s".format(rec.trigger_time))

    len_df = len(rec.time)
    df = pd.DataFrame(list(rec.time), columns =['TIME'])

    df['VR'] = rec.analog[0][:len_df]
    df['VS'] = rec.analog[1][:len_df]
    df['VT'] = rec.analog[2][:len_df]
    df['N_VR'] = get_normal_2(df.VR,len_df)[:len_df]
    df['N_VS'] = get_normal_2(df.VS,len_df)[:len_df]
    df['N_VT'] = get_normal_2(df.VT,len_df)[:len_df]

    df_temp = df.copy()[["VR","N_VR","VS","N_VS","VT","N_VT"]]

    corr = df_temp.corr()
    corr_VR = corr["VR"]["N_VR"]
    corr_VS = corr["VS"]["N_VS"]
    corr_VT = corr["VT"]["N_VT"]
    hm = sns.heatmap(corr)
    fig = hm.get_figure()
    fig.savefig("heatmap.png")

    if corr_VR < corr_VS and corr_VR < corr_VT:
      faulty = "VR"
    elif corr_VS < corr_VR and corr_VS < corr_VT:
      faulty = "VS"
    elif corr_VT < corr_VR and corr_VT < corr_VS:
      faulty = "VT"

    df2 = pd.DataFrame(columns=["VR","VS","VT","N_VR","N_VS","N_VT"])

    for i in ["VR","VS","VT"]:

      scaling_factor = get_scaling_factor(list(df['N_'+faulty]))

      df2[i] = get_rms(df.TIME, np.array(df[i])*scaling_factor)[:3000]
      df2['N_'+i] = get_rms(df.TIME, np.array(df['N_'+i])*scaling_factor)[:3000]

      plt.switch_backend('agg')
      plt.clf()
      plt.plot(np.array(df2[i]))
      plt.savefig(i+".png")

    for i in [faulty]:
      df_temp = df2.copy()[[faulty]]
      df_temp = df_temp.set_axis(['V'], axis=1)
      plt.switch_backend('agg')
      plt.clf()
      plt.plot(np.array(df_temp['V']))
      plt.savefig("faulty_wave.png")
      # labels = list(np.repeat(1, len(df_temp)))
      # df_temp['labels'] = labels
      # df_new = pd.concat([df_new, df_temp])

      df_temp = df2.copy()[['N_'+faulty]]
      df_temp = df_temp.set_axis(['V'], axis=1)
      plt.switch_backend('agg')
      plt.clf()
      plt.plot(np.array(df_temp['V']))
      plt.savefig("normal_wave.png")
      labels = list(np.repeat(0, len(df_temp)))
      df_temp['labels'] = labels
      df_new = pd.concat([df_new, df_temp])
      # print(df_new)

  try:
    arr_to_analyze = np.reshape(list(df_temp['V']),(len(df_temp['V']),1,1))
    # print(arr_to_analyze.shape)
    result = run_ml_predict(arr_to_analyze)
    # print(result)
    print("prediction shape: ", result.shape)
    if np.mean(result[0]) > 0.5:
      return "Faulty occurs on "+faulty+" ("+str(np.mean(result[0])*100)+"%)"
    else:
      return "Data is normal."
  except:
    return 'Prediction process error.'

def run_ml_predict(faultychannellist):
  import_or_install('keras')
  from tensorflow import keras
  from keras.models import load_model
  opt_adam = keras.optimizers.Adam(learning_rate=0.001)
  model = load_model('bestmodel.h5')
  model.compile(optimizer=opt_adam,
                  loss=['binary_crossentropy'],
                  metrics=['accuracy'])
  prediction = model.predict(faultychannellist)
  # prediction = model.predict_classes(faultychannellist)
  return prediction

def get_normal_2(signal,len_time):
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
  normal_wave = normal_wave_10
  while len(normal_wave) < len_time:
    normal_wave = normal_wave + normal_wave_10
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

def get_scaling_factor(arr):
  x = sorted(list(arr), reverse=True)
  # maxval = sum(x[:20])/20
  # return maxval
  x2 = np.array(x[:1500])
  Q1 = np.quantile(x2, 0.25)
  Q3 = np.quantile(x2, 0.75)
  IQR = Q3 - Q1
  Lower_Fence = Q1 - (1.5 * IQR) * -1
  Upper_Fence = Q3 + (1.5 * IQR)
  max_val = 0
  if Lower_Fence < Upper_Fence:
    max_val = Lower_Fence
  else:
    max_val = Upper_Fence
  cur_max = 100
  scaling_factor = cur_max/max_val
  return scaling_factor

def get_rms(df_time,data):
  s_rate = get_fs(df_time)

  Fs = s_rate//10

  RMS = lambda x: np.sqrt(np.mean(x**2))
  # print(len(data)-Fs)
  sample = np.arange(len(data)-Fs)
  RMS_of_sample = []
  for ns in sample:
      # here you can apply the frequency window for the sample
      RMS_of_sample.append(RMS(data[ns:ns+Fs]))
  return np.array(RMS_of_sample)

def get_fs(df_time):
  s_rate = math.floor(len(df_time)/(df_time[len(df_time)-1]-df_time[0]))
  s_rate_floor = math.floor(s_rate/100)*100
  s_rate_ceil = math.ceil(s_rate/100)*100
  if abs(s_rate-s_rate_floor) < abs(s_rate-s_rate_ceil):
    s_rate = s_rate_floor
  else:
    s_rate = s_rate_ceil
  return s_rate
