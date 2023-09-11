import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

import_or_install('comtrade')

import comtrade

def read_comtrade_channels():
    fname = 'upload/'+'toanalyze'
    rec = comtrade.load(fname+".cfg", fname+".dat")
    try:
      rec = comtrade.load(fname+".cfg", fname+".dat")
    except:
      rec = comtrade.load(fname+".CFG", fname+".DAT")
    print("Trigger time = {}s".format(rec.trigger_time))
    print("Channels:")
    return rec.analog_channel_ids