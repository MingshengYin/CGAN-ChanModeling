# MultiFreqChanModeling-by-GAN-DL-Project
This is a course project for ECE-7123 DL. Mingsheng Yin(my1778); Yaqi Hu(yh2829). 

The work is extended from 
* [mmwchanmod](https://github.com/nyu-wireless/mmwchanmod)

## The folder 'data' stores the datasets for training and test
'dict_BeijingTokyo.p' stores all raw data from Beijing and Tokyo maps required for CWGAN-GP.
'dict_LondonMoscow.p' stores all raw data from London and Moscow maps required for CWGAN-GP.
'dict_BeijingTokyo.csv' and 'dict_LondonMoscow.csv' can show the insights of the above pickle files.

## The folder 'fit_3gpp' stores
1) code for refitting the 3gpp default model, '.h5' files that save the refitted model, and plots of loss changing over epochs
2) code for plotting path loss that compare with our proposed GAN model

## The folder 'mmwchanmod' stores all core codes related to training the GAN model.

## The foler 'models' saves two pre-trained GAN model
1) Beijing + Tokyo
2) London + Moscow

## The folder 'plots' saves all figures that compare our generated data with test data
1) Compare the path loss: 'pl_models_compare_3gpp_Beijing_Tokyo.png', 'pl_models_compare_3gpp_London_Moscow.png', '3gpp_pl_London_Moscow.png', '3gpp_pl_Beijing_Tokyo.png'
2) Compare the delay: 'London_Moscow_dly_rms.png', 'London_Moscow_dly_rms_rho.png', 'Beijing_Tokyo_dly_rms.png', 'Beijing_Tokyo_dly_rms_rho.png'
3) Compare the snr: 'snr_compare_two_freq_Beijing_Tokyo.png', 'snr_compare_two_freq_London_Moscow.png'
4) Compare the beamforming gain error: 'bf_gain_error_Beijing_Tokyo.png', 'bf_gain_error_London_Moscow.png'

## Training model from scratch	
Go to the current folder and run(e.g.):
```
python train_mod_interactive.py --model_dir models/Beijing-Tokyo --data_dir data/dict_BeijingTokyo.p --nepochs_path 5000 
```
'train_mod_interactive.py' has commands to change the number of epochs and model parameters, etc.


## plot the path-loss cdf
'fit_3gpp/plot_path_loss_cdf_3gpp.py' plots the CDF of the path loss on the 1)test data; 2) Default 3GPP model; 3) Refitted 3GPP; 4) our proposed GAN
Go to the 'fit_3gpp' folder and run(e.g.):
```
python plot_path_loss_cdf_3gpp.py --model_dir ../models/Beijing_Tokyo --fc 2.3e9
```

## plot the path-loss compared in two frequencies
'plot_path_loss_compare_3gpp.py' plots the scatter diagram for the omnidirectional path
loss in 2.3GHz and 28GHz under the 4 models, 1) test data from the ray tracing data; 
2) Trained GAN Model; 3) Refitted 3GPP model; 4) Default 3GPP Model
Go to the 'fit_3gpp' folder and run(e.g.):
```
python plot_path_loss_compare_3gpp.py --model_dir ../models/Beijing_Tokyo --plot_fn pl_models_compare_3gpp_Beijing_Tokyo.png 
```

## plot rms excess delay and its correlation coefficients
'plot_dly_rms.py' plots 1) the RMS excess delay taking average every 40m from 0 to 1300 m;
2) Correlation coefficient of the RMS excess delay
Go to the current folder and run(e.g.):
```
python plot_dly_rms.py --model_dir models/Beijing_Tokyo --plot_fn Beijing_Tokyo_dly_rms
```

## plot the SNR
'plot_snr_cmp_two_freq.py': plot the SNR distribution in a single cell environment.
Go to the current folder and run(e.g.):
```
python plot_snr_cmp_two_freq.py --model_dir models/Beijing_Tokyo --plot_fn snr_compare_two_freq_Beijing_Tokyo.png
```

## plot beamforming gain error
'plot_bf_gain_error.py' plots the cdf of error between optimal beamforming gain and the beamforming gain using angles estimated under 2.3 GHz path loss
Go to the current folder and run(e.g.):
```
python plot_bf_gain_error.py --model_dir models/Beijing_Tokyo --plot_dir plots --plot_fn bf_gain_error_Beijing_Tokyo.png
```

## To modify the code for the network, please go to "mmwchanmod/learn/models.py". In "models.py": 
1. the class 'CondGAN' describes the structure of GAN
2. 'fit_path_mod' in class 'ChanMod' is where we train GAN
