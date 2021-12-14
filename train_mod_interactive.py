"""
train_mod.py:  Training of the channel model

This program trains both the link state predictor
and path GAN models from the ray tracing data.  

Example to train for Beijing-Tokyo:
    python train_mod_interactive.py --model_dir models/Beijing-Tokyo --data_dir data/dict_BeijingTokyo.p --nepochs_path 5000 
"""

import numpy as np
import pickle


import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K
import argparse
from datetime import date
    
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.common.constants import DataConfig

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Trains the channel model')
parser.add_argument('--freq_set',action='store',nargs='+',default=[2.3e9,28e9],
    type=float,help='list of frequencies')
parser.add_argument('--max_ex_pl',action='store',default=80,type=int,
    help='maximum pathloss')
parser.add_argument('--tx_pow_dbm',action='store',default=23,type=int,
    help='transmit power in dBm')
parser.add_argument('--npaths_max',action='store',default=20,type=int,
    help='max number of paths per link')
parser.add_argument('--test_size',action='store',default=0.80,type=int,
    help='size of test set')
parser.add_argument('--nlatent',action='store',default=20,type=int,
    help='number of latent variables')
parser.add_argument('--nepochs_link',action='store',default=50,type=int,
    help='number of epochs for training the link model')
parser.add_argument('--lr_link',action='store',default=1e-4,type=float,
    help='learning rate for the link model')   
parser.add_argument('--nepochs_path',action='store',default=3000,type=int,
    help='number of epochs for training the path model')
parser.add_argument('--lr_path',action='store',default=1e-4,type=float,
    help='learning rate for the path model') 
parser.add_argument('--batch_size_path',action='store',default=1024,type=float,
    help='batch size for the path model')        
parser.add_argument('--out_var_min',action='store',default=1e-4,type=float,
    help='min variance in the decoder outputs.  Used for conditioning')     
parser.add_argument('--init_stddev',action='store',default=10.0,type=float,
    help='weight and bias initialization')
parser.add_argument('--nunits_enc',action='store',nargs='+',default=[200,80],
    type=int,help='num hidden units for the encoder')    
parser.add_argument('--nunits_dec',action='store',nargs='+',default=[80,200],
    type=int,help='num hidden units for the decoder')    
parser.add_argument('--nunits_link',action='store',nargs='+',default=[25,10],
    type=int,help='num hidden units for the link state predictor')        
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument('--data_dir',action='store',default= 'data/dict.p', 
    help='directory to store data')
parser.add_argument('--no_fit_link', dest='no_fit_link', action='store_true',
    help="Does not fit the link model")
parser.add_argument('--no_fit_path', dest='no_fit_path', action='store_true',
    help="Does not fit the path model")
parser.add_argument('--checkpoint_period',action='store',default=100,type=int,
    help='Period in epochs for storing checkpoint.  A value of 0 indicates no checkpoints')      

args = parser.parse_args()

freq_set = args.freq_set
max_ex_pl = args.max_ex_pl
tx_pow_dbm = args.tx_pow_dbm
npaths_max = args.npaths_max
test_size = args.test_size
nlatent = args.nlatent
nepochs_path = args.nepochs_path
lr_path = args.lr_path
batch_size_path = args.batch_size_path
nepochs_link = args.nepochs_link
lr_link = args.lr_link
init_stddev = args.init_stddev
nunits_enc = args.nunits_enc
nunits_dec = args.nunits_dec
nunits_link = args.nunits_link
model_dir = args.model_dir
data_dir = args.data_dir
out_var_min = args.out_var_min
fit_link = not args.no_fit_link
fit_path = not args.no_fit_path
checkpoint_period = args.checkpoint_period

"""
Load the data
"""
# Load pre_processed data (.p format)
with open(data_dir, 'rb') as handle:
    data = pickle.load(handle)

nlink = data['dvec'].shape[0]
data['rx_type'] = np.zeros((nlink,))
nts = int(nlink*test_size) #number of test samples
ntr = nlink-nts #number of train 
I = np.random.permutation(nlink)

# train test split
train_data = dict()
test_data = dict()
for key in data:
    # train_data[key] = data[key][I[:ntr]]
    train_data[key] = data[key]
    test_data[key] = data[key][I[ntr:]]
    
# Create configuration object and set the values
cfg = DataConfig()
cfg.date_created = date.today().strftime("%d-%m-%Y")  
cfg.max_ex_pl = max_ex_pl
cfg.tx_pow_dbm = tx_pow_dbm
cfg.npaths_max = npaths_max  

# Configure arbitrary freq
cfg.freq_set = freq_set
cfg.nfreq = len(freq_set)

"""
Build the model
"""
# Construct the channel model object
K.clear_session()

chan_mod = ChanMod(nlatent=nlatent,cfg=cfg,\
    nunits_link=nunits_link,\
    model_dir=model_dir)    
chan_mod.save_config()

"""
Train the link classifier
"""
fit_link = True
if fit_link:
    # Build the link model
    chan_mod.build_link_mod()

    # Fit the link model 
    chan_mod.fit_link_mod(train_data, test_data, lr=lr_link,\
                          epochs=nepochs_link)
    
    # Save the link classifier model
    chan_mod.save_link_model()
    
else:
    # Load the link model
    chan_mod.load_link_model()  

"""
Train the path loss model
"""
fit_path = True
if fit_path:
    chan_mod.build_path_mod()
    
    chan_mod.fit_path_mod(train_data, test_data, lr=lr_path,\
                          epochs=nepochs_path,\
                          batch_size=batch_size_path,\
                          checkpoint_period=checkpoint_period)

# Save train and test data
with open(model_dir+'/train_data.p', 'wb') as handle:
    pickle.dump(train_data, handle)

with open(model_dir+'/test_data.p', 'wb') as handle:
    pickle.dump(test_data, handle)

# Save configuration
with open(model_dir+'/cfg.p', 'wb') as fp:
        pickle.dump(cfg, fp)  