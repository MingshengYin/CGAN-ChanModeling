"""
plot_bf_gain_error:  Plots the cdf of error between optimal beamforming gain
and the beamforming gain using angles estimated under 2.3 GHz path loss

To plot the error in Beijing_Tokyo:
    python plot_bf_gain_error.py --model_dir models/Beijing_Tokyo --plot_dir plots --plot_fn bf_gain_error_Beijing_Tokyo.png
    
To plot the error in London_Moscow:
    python plot_bf_gain_error.py --model_dir models/London_Moscow --plot_dir plots --plot_fn bf_gain_error_London_Moscow.png
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

import tensorflow.keras.backend as K

    
from mmwchanmod.learn.models import load_model 
from mmwchanmod.sim.antenna import Elem3GPP, ElemDipole
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import dir_bf_gain_multi_sect
    
"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the SNR distribution')    
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='bf_gain_error.png', help='plot file name')        
    
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
plot_fn = args.plot_fn

"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

# Paramters
bw_f2 = 400e6   # f2 Bandwidth in Hz
bw_f1 = 20e6    # f1 Bandwidth in Hz
nf = 6  # Noise figure in dB
kT = -174   # Thermal noise in dBm/Hz
tx_pow = 23  # TX power in dBm
npts = 100    # number of points for each (x,z) bin
aer_height=30  # Height of the aerial cell in meterss
downtilt = 10  # downtilt in degrees

fc1 = 2.3e9 # freq 1
fc2 = 28e9  # carrier frequency in Hz freq 2
nant_gnb_fc1 = np.array([1,4])  # gNB array size Tx
nant_ue_fc1 = np.array([1,1])   # UE/UAV array size Rx
nant_gnb_fc2 = np.array([8,8])  # gNB array size Tx
nant_ue_fc2 = np.array([1,8])   # UE/UAV array size Rx
nsect = 3  # number of sectors for terrestrial gNBs   

"""
Create the arrays
"""
# Terrestrial gNB.
# We downtilt the array and then replicate it over three sectors
elem_gnb_f1 = ElemDipole() # 2.3GHz
elem_gnb_f2 = Elem3GPP(thetabw=82, phibw=82) # 28GHz
arr_gnb0_f1 = URA(elem=elem_gnb_f1, nant=nant_gnb_fc1, fc=fc1)
arr_gnb0_f2 = URA(elem=elem_gnb_f2, nant=nant_gnb_fc2, fc=fc2)

arr_gnb_list_f1 = multi_sect_array(\
        arr_gnb0_f1, sect_type='azimuth', theta0=-downtilt, nsect=nsect)
arr_gnb_list_f2 = multi_sect_array(\
        arr_gnb0_f2, sect_type='azimuth', theta0=-downtilt, nsect=nsect)

# UE array.  Array is pointing down.
elem_ue_f1 = ElemDipole() # 2.3GHz
elem_ue_f2 = Elem3GPP(thetabw=82, phibw=82) # 28GHz
arr_ue0_f1 = URA(elem=elem_ue_f1, nant=nant_ue_fc1, fc=fc1)
arr_ue0_f2 = URA(elem=elem_ue_f2, nant=nant_ue_fc2, fc=fc2)
arr_ue_f1 = RotatedArray(arr_ue0_f1,theta0=90) # point up
arr_ue_f2 = RotatedArray(arr_ue0_f2,theta0=90)
    

"""
Generate chan_list for test data
"""
# use the ray tracing data (real path data)
chan_list, ls = data_to_mpchan(test_data, cfg) 

dvec = test_data['dvec']
d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
I = np.where((d3d>=100)& (d3d<=200))[0].astype(int) # only plot distance in (100,200) m

"""
Plot SNR 2.3 vs 28
"""
snr_plot = np.zeros((len(I), 2))
bf_gain_error_ls = []
for i, itest in enumerate(I):
    is_valid, bf_gain_error = dir_bf_gain_multi_sect(\
                    arr_gnb_list_f1, arr_gnb_list_f2, \
                    [arr_ue_f1], [arr_ue_f2], chan_list[itest])
    if is_valid:
        bf_gain_error_ls.append(bf_gain_error[0][0])


plt.plot(np.sort(bf_gain_error_ls), np.arange(len(bf_gain_error_ls))/len(bf_gain_error_ls)) 
plt.xlabel('Beamforming Gain Error (dB)', fontsize=14)
plt.ylabel('CDF', fontsize=14)
plt.grid()
# Print plot

"""
Load the pre-trained model
"""
# Construct and load the channel model object
mod_name = model_name
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(model_name, cfg)
    
"""
Generate chan_list for generated data
"""
fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
for ifreq in range(cfg.nfreq):
    fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]

# Generate chan by the trained model (input is 'dvec')
chan_list, ls = chan_mod.sample_path(test_data['dvec'], \
                                    fspl_ls, test_data['rx_type'])
dvec = test_data['dvec']
d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
I = np.where((d3d>=100)& (d3d<=200))[0].astype(int) # only plot distance in (100,200) m

"""
Plot SNR 2.3 vs 28
"""
snr_plot = np.zeros((len(I), 2))
bf_gain_error_ls = []
for i, itest in enumerate(I):
    is_valid, bf_gain_error = dir_bf_gain_multi_sect(\
                    arr_gnb_list_f1, arr_gnb_list_f2, \
                    [arr_ue_f1], [arr_ue_f2], chan_list[itest])
    # Compute the effective SNR
    if is_valid:
        bf_gain_error_ls.append(bf_gain_error[0][0])

plt.plot(np.sort(bf_gain_error_ls), np.arange(len(bf_gain_error_ls))/len(bf_gain_error_ls))
plt.legend(['Real Data', 'Train Model'])
plt.xlim(-1,25)

if 1:
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    print('Figure saved to %s' % plot_path)