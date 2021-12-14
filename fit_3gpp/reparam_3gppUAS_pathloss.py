# -*- coding: utf-8 -*-

"""
@author: Tommy Azzino

@modified by: Mingsheng Yin, Yaqi Hu

To train 2.3GHz for Beijing_Tokyo:
    python reparam_3gppUAS_pathloss.py --fc 2.3e9 --model_dir ../models/Beijing_Tokyo
"""

import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

path = os.path.abspath('../')

if path not in sys.path:
    sys.path.append(path)
print(sys.path)
from mmwchanmod.learn.param_est_3gpp import ParamEst3gppUAS, Custom3gppPathLossLayer, BoundCallback

parser = argparse.ArgumentParser()
parser.add_argument('--fc',action='store',default=28e9,type=float,
    help='frequency to fit')
parser.add_argument('--model_dir',action='store',default= '../models/Beijing', 
    help='directory of training data')

# scenario parameters
args = parser.parse_args()
fc = args.fc
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
param = 'pathloss'

city = model_name
rx_type = 1
if rx_type == 0:
    hbs = 20
else:
    hbs = 10
save_model = True

print('\nLoading data\n')
est = ParamEst3gppUAS(param, fc, model_dir+'/train_data.p', rx_type, hbs)

est.load_data()
est.format_data_path_loss(3)

xtr = est.xtr_pl
ytr = est.ytr_pl

est = ParamEst3gppUAS(param, fc, model_dir+'/test_data.p', rx_type, hbs)

est.load_data()
est.format_data_path_loss(3)

xts = est.xtr_pl
yts = est.ytr_pl


# print CDF of the path loss for the training data
'''plt.figure()
plt.plot(np.sort(ytr), np.linspace(0, 1, ytr.shape[0]))
plt.ylabel("CDF")
plt.xlabel("Path Loss")
plt.grid()
plt.show()'''

# Use tensorflow to train the model
# the following initial parameters come from the 3GPP document TR 36.777 (Table B-2: Pathloss models)
model = tf.keras.models.Sequential(Custom3gppPathLossLayer(name="custom_layer_pathLoss", fc = fc))

# Create the callback to bound the weights
layer = model.get_layer("custom_layer_pathLoss")
cb = BoundCallback(layer, low=0.01, high=10)

# Compile the model
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    run_eagerly=False,  # for debugging (if True --> debug mode)
)

print('Start training\n')
n_epochs = 50
hist = model.fit(xtr, ytr, batch_size=128, epochs=n_epochs, verbose=1, validation_data=(xts, yts), callbacks=[cb])

print('\nTraining Completed...\n')
print('\n3GPP Model Parameters LOS: ', [32.4, 21.0, 20.0, 32.4, 40.0, 20.0, -9.5])
print('Trained Model Parameters LOS: ', model.get_weights()[0]*np.array(layer.param_los_nom))
print('\n3GPP Model Parameters NLOS: ', [35.3, 22.4, 21.3, -0.3])
print('Trained Model Parameters NLOS: ', model.get_weights()[1]*np.array(layer.param_nlos_nom))

# plot results
plt.figure()
plt.plot(np.arange(n_epochs)+1, hist.history['loss'], linewidth=2, label='Train Loss')
plt.plot(np.arange(n_epochs)+1, hist.history['val_loss'], linewidth=2, label="Val Loss")
plt.ylabel("Metric")
plt.xlabel("Epoch")
plt.ylim([0, 2000.0])
plt.legend()
if round(fc/1e9,2) >= 10:
    print_fc = str(int(fc/1e9))
else:
    print_fc = str(round(fc/1e9,2))
plt.title(print_fc +"GHz - Loss vs Epoch")
plt.grid()
fig_name = "./training_loss_3gpp_UAS_pathloss_" + str(city) + "_" + print_fc
plt.savefig(fig_name + ".png")
plt.show()

fn = None
if save_model:
    if rx_type == 0:
        cell = "aer"
    elif rx_type == 1:
        cell = "ter"
    else:
        raise NotImplemented
    fn = cell + "_" + param + "_3GPPUAS_pathloss_" + str(n_epochs) + "_" + city + "_" + print_fc
    model.save(fn + ".h5", overwrite=True)

    loaded_model = tf.keras.models.load_model(fn + ".h5",
                                              custom_objects={'Custom3gppPathLossLayer': Custom3gppPathLossLayer})
    out = loaded_model.predict(xtr)
    print('Checking weights of trained model:\n')
    print('Loaded Model Parameters LOS: ', loaded_model.get_weights()[0]*np.array(layer.param_los_nom))
    print('Loaded Model Parameters NLOS: ', loaded_model.get_weights()[1]*np.array(layer.param_nlos_nom))
