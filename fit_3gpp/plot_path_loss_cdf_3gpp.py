"""
plot_path_loss_cdf_3gpp:  Plots the CDF of the path loss on the 1)test data; 2)
Default 3GPP model; 3) Refitted 3GPP; 4) our proposed GAN

To plot the cdf in Beijing-Tokyo at 2.3GHz:
    
    python plot_path_loss_cdf_3gpp.py --model_dir ../models/Beijing_Tokyo --fc 2.3e9

To plot the cdf in London_Moscow at 2.3GHz:
    
    python plot_path_loss_cdf_3gpp.py --model_dir ../models/London_Moscow --fc 2.3e9
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse


import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K


path = os.path.abspath('../')
if not path in sys.path:
    sys.path.append(path)

from mmwchanmod.common.constants import LinkState
from mmwchanmod.learn.models import load_model
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.learn.param_est_3gpp import ParamEst3gppUAS, Custom3gppPathLossLayer


"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the omni directional path loss CDF')
parser.add_argument('--model_dir',action='store',default= '../models/Beijing_Tokyo', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='../plots', help='directory for the output plots')        
parser.add_argument('--fc',action='store',default=28e9,type=float,
    help='frequency to fit')    

        
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
fc = args.fc
if round(fc/1e9,2) >= 10:
    print_fc = str(int(fc/1e9))
else:
    print_fc = str(round(fc/1e9,2))
print(f'++++++++++++++++++++++++ {print_fc}')


"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

use_true_ls = False
# use_true_ls = True

        
"""
Find the path loss CDFs
"""
pl_omni_plot = []
ls_plot = []

for i in range(2):
    
    if (i == 0):
        """
        For first city, use the city data
        """
        # Load the data
        # Convert data to channel list
        chan_list, ls = data_to_mpchan(test_data, cfg)
        
        
    else:
        """
        For subsequent cities, generate data from model
        """
        
        # Construct the channel model object
        K.clear_session()
        chan_mod = load_model(model_name, cfg)
        mod_name = model_name
        # Load the configuration and link classifier model
        print('Simulating model %s'%mod_name)       
        
        # Generate samples from the path
        if use_true_ls:
            ls = test_data['link_state']
        else:
            ls = None
        fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
        for ifreq in range(cfg.nfreq):
            fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]
        chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)
                  
            
        
    # Compute the omni-directional path loss for each link    
    n = len(chan_list)
    pl_omni = np.zeros(n)
    for i, chan in enumerate(chan_list):
        if chan.link_state != LinkState.no_link:
            if fc == cfg.freq_set[0]:
                pl_omni[i] = chan.comp_omni_path_loss()[0]
            elif fc == cfg.freq_set[1]:
                pl_omni[i] = chan.comp_omni_path_loss()[1]
    
    # Save the results    
    ls_plot.append(ls)
    pl_omni_plot.append(pl_omni)
        
                           
"""
Create plot
"""
param = 'pathloss'
city = model_name
rx_type = 1
if rx_type == 0:
    cell = "aer"
    hbs = 20
else:
    cell = "ter"
    hbs = 10

fn = cell + "_" + param + "_3GPPUAS_pathloss_50_" + city + "_" + print_fc
print(f'++++++++++++++++++++++++ {fn}')

print('\nLoading data\n')
print(f'++++++++++++++++++++++++ {fc}')
est = ParamEst3gppUAS(param, fc, model_dir+'/test_data.p', rx_type, hbs)
est.load_data()
est.format_data_path_loss(3)
x_data = est.xtr_pl
y_data = est.ytr_pl

# Load the trained model
custom_3gpp_path_loss_layer = Custom3gppPathLossLayer(name="custom_layer_pathloss", fc = fc)
trained_model = tf.keras.models.load_model(fn+".h5",
                                           custom_objects={'Custom3gppPathLossLayer': custom_3gpp_path_loss_layer})
# create 3GPP model
model_3gpp = tf.keras.models.Sequential(Custom3gppPathLossLayer(name="custom_layer_pathloss", fc = fc))

# Compile the model
model_3gpp.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

# 3GPP model
pl_3gpp_out = model_3gpp.predict(x_data)

# 3GPP trained model
pl_trained_out = trained_model.predict(x_data)

plt.figure()
I = np.where((ls_plot[0] != LinkState.no_link))[0]
plt.plot(np.sort(pl_omni_plot[0][I]), np.linspace(0, 1, pl_omni_plot[0][I].shape[0]), '-', linewidth=2, label='Test data')
plt.plot(np.sort(pl_3gpp_out), np.linspace(0, 1, pl_3gpp_out.shape[0]), '--', linewidth=2, label='Default 3GPP')
plt.plot(np.sort(pl_trained_out), np.linspace(0, 1, pl_trained_out.shape[0]), ':', linewidth=2, label='Refitted 3GPP')
I = np.where((ls_plot[1] != LinkState.no_link))[0]
plt.plot(np.sort(pl_omni_plot[1][I]), np.linspace(0, 1, pl_omni_plot[1][I].shape[0]), '-.', linewidth=2, label='Proposed GAN')
plt.ylabel("CDF", fontsize=12)
plt.xlabel("Path Loss [dB]", fontsize=12)

if print_fc == '2.3':
    plt.xlim([50, 180])
elif print_fc == '28':
    plt.xlim([60, 220])

plt.title(print_fc +"GHz - "+str(city)+ " - Path Loss Fitting")
plt.legend()
plt.grid()
fig_name = "./eval_3gpp_UAS_pathloss_" + str(city) + "_" + print_fc
plt.savefig(plot_dir+'/'+fig_name + ".png")
plt.show()