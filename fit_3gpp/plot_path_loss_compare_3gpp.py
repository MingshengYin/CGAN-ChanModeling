"""
plot_path_loss_compare_3gpp:  Plot the scatter diagram for the omnidirectional path
loss in 2.3GHz and 28GHz under the 4 models, 1) test data from the ray tracing data; 
2) Trained GAN Model; 3) Refitted 3GPP model; 4) Default 3GPP Model

To plot Beijing_Tokyo models:
    
    python plot_path_loss_compare_3gpp.py --model_dir ../models/Beijing_Tokyo --plot_fn pl_models_compare_3gpp_Beijing_Tokyo.png 
    
To plot London_Moscow models:    
    python plot_path_loss_compare_3gpp.py --model_dir ../models/London_Moscow --plot_fn pl_models_compare_3gpp_London_Moscow.png 
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
parser = argparse.ArgumentParser(description='Plots the omni directional path loss in two frequencies')
parser.add_argument('--model_dir',action='store',default= '../models/Beijing_Tokyo', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='../plots', help='directory for the output plots')     
parser.add_argument(\
    '--plot_fn',action='store',\
    default='pl_models_compare_3gpp_Beijing_Tokyo.png', help='plot file name')          
        
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

# use_true_ls = False
use_true_ls = True
print(len(test_data['dvec']))

"""
Find the path loss CDFs
"""
pl_omni_plot = []
pl2_omni_plot = []
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
    pl2_omni = np.zeros(n)
    for i, chan in enumerate(chan_list):
        if chan.link_state != LinkState.no_link:
            pl_omni[i], pl2_omni[i]= chan.comp_omni_path_loss()

    # Save the results    
    ls_plot.append(ls)
    pl_omni_plot.append(pl_omni)
    pl2_omni_plot.append(pl2_omni)

dvec = test_data['dvec']
d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)

# Find the links that match the type and are not in outage
I = np.where(ls_plot[0] != LinkState.no_link)[0]
print(len(I))



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

est_2_3 = ParamEst3gppUAS(param, 2.3e9, model_dir+'/test_data.p', rx_type, hbs)
est_2_3.load_data()
est_2_3.format_data_path_loss(3)
x_data_plot_2_3 = est_2_3.xtr_pl
y_data_2_3 = est_2_3.ytr_pl


est_28 = ParamEst3gppUAS(param, 28e9, model_dir+'/test_data.p', rx_type, hbs)
est_28.load_data()
est_28.format_data_path_loss(3)
x_data = est_28.xtr_pl
y_data = est_28.ytr_pl
x_data_plot_28 = np.copy(x_data) 


fn_2_3 = cell + "_" + param + "_3GPPUAS_pathloss_50_" + city + "_2.3"
print(f'++++++++++++++++++++++++ {fn_2_3}')
fn_28 = cell + "_" + param + "_3GPPUAS_pathloss_50_" + city + "_28"
print(f'++++++++++++++++++++++++ {fn_28}')

# Load the trained model
custom_3gpp_path_loss_layer = Custom3gppPathLossLayer(name="custom_layer_pathloss", fc = 2.3e9)
trained_model_2_3 = tf.keras.models.load_model(fn_2_3+".h5",
                                           custom_objects={'Custom3gppPathLossLayer': custom_3gpp_path_loss_layer})
custom_3gpp_path_loss_layer = Custom3gppPathLossLayer(name="custom_layer_pathloss", fc = 28e9)
trained_model_28 = tf.keras.models.load_model(fn_28+".h5",
                                           custom_objects={'Custom3gppPathLossLayer': custom_3gpp_path_loss_layer})
pl_trained_2_3 = trained_model_2_3.predict(x_data_plot_2_3)
pl_trained_28 = trained_model_28.predict(x_data_plot_28)

# create 3GPP model
model_3gpp_2_3 = tf.keras.models.Sequential(Custom3gppPathLossLayer(name="custom_layer_pathloss", fc = 2.3e9))
# Compile the model
model_3gpp_2_3.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
model_3gpp_28 = tf.keras.models.Sequential(Custom3gppPathLossLayer(name="custom_layer_pathloss", fc = 28e9))
model_3gpp_28.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

pl_3gpp_2_3 = model_3gpp_2_3.predict(x_data_plot_2_3)
pl_3gpp_28 = model_3gpp_28.predict(x_data_plot_28)


fig = plt.figure(tight_layout=True)
ax1 = plt.subplot(221)
plt.scatter(np.array(y_data_2_3),np.array(y_data),c='b',marker=',',s=1, alpha = 0.1)
ax1.set_title('Ray Tracing Smaples Truth', fontsize=10)
ax1.set_xlim([50,175])
ax1.set_ylim([50,200])
ax1.label_outer()
ax2 = plt.subplot(222, sharex=ax1, sharey = ax1)
plt.scatter(pl_omni_plot[1][I], pl2_omni_plot[1][I],c='b',marker=',',s=1, alpha = 0.1)
ax2.set_title('GAN Model', fontsize=10)
ax2.label_outer()
ax3 = plt.subplot(223, sharex=ax1, sharey = ax1)
plt.scatter(pl_trained_2_3,pl_trained_28, c='b',marker=',',s=1, alpha = 0.1)
ax3.set_title('Refitted 3GPP Model', fontsize=10)
ax4 = plt.subplot(224, sharex=ax1, sharey = ax1)
plt.scatter(pl_3gpp_2_3,pl_3gpp_28, c='b',marker=',',s=1, alpha = 0.1)
ax4.set_title('Default 3GPP Model', fontsize=10)
ax4.label_outer()
fig.supxlabel('2.3GHz Path loss(dB)')
fig.supylabel('28GHz Path loss(dB)')
fig.suptitle(model_name)
plt.show()
fig.savefig(plot_dir+'/'+plot_fn, dpi=1200)