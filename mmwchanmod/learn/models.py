"""
models.py:  Classes for the modeling, partial code reference from 
    1) https://github.com/nyu-wireless/mmwchanmod
    2) https://keras.io/examples/generative/wgan_gp/

"""
import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import numpy as np
import sklearn.preprocessing
import pickle
from tensorflow.keras.optimizers import Adam
import os

from mmwchanmod.common.spherical import spherical_add_sub, cart_to_sph
from mmwchanmod.common.constants import PhyConst, AngleFormat
from mmwchanmod.common.constants import LinkState, DataConfig 
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.learn.preproc_param import preproc_to_param, param_to_preproc
    

class CondGAN(object):
    '''
    Object for bulit GAN
    '''
    def __init__(self, nlatent, npaths_max, nparams, ncond,\
         nunits_dsc=(1120,560,280,), nunits_gen=(280,560,1120)):

        self.nlatent = nlatent
        self.npaths_max = npaths_max
        self.nparams = nparams        
        self.ncond = ncond
        self.nunits_dsc = nunits_dsc
        self.nunits_gen = nunits_gen

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):

        # conditional input
        cond = tfkl.Input((self.ncond,), name='cond')
        n_nodes = self.nunits_dsc[-1]
        li = tfkl.Dense(n_nodes)(cond)
        li = tfkl.Reshape((n_nodes,1))(li)

        # channel parameter input
        in_lat = tfkl.Input((self.nlatent,), name='in_lat')
        gen = tfkl.Dense(n_nodes)(in_lat)
        gen = tfkl.Reshape((n_nodes,1))(gen)

        # merge channel gen and condition
        merge = tfkl.Concatenate(name='gen_cond')([gen, li])
        gen = tfkl.Flatten()(merge)
        gen = tfkl.LeakyReLU(alpha=0.2)(gen)
        gen = tfkl.BatchNormalization()(gen)
        layer_names = []
        for i in range(len(self.nunits_gen)):           
            gen = tfkl.Dense(self.nunits_gen[i],\
                           name='FC%d' % i)(gen)
            gen = tfkl.LeakyReLU(alpha=0.2)(gen)
            layer_names.append('FC%d' % i)

        # output
        out_layer = tfkl.Dense(self.npaths_max*self.nparams)(gen)
        # define model
        g_model = tfk.Model([in_lat, cond], out_layer)

        # save network architecture
        # dot_img_file = 'nnArch/gen.png'
        # tf.keras.utils.plot_model(g_model, to_file=dot_img_file, show_shapes=True)   

        return g_model

    def build_discriminator(self):
        # conditional input
        cond = tfkl.Input((self.ncond,), name='cond')

        # scale up dimensions with linear activation
        n_nodes = self.npaths_max * self.nparams
        li = tfkl.Dense(n_nodes)(cond)
        # reshape to additional channel
        li = tfkl.Reshape((self.npaths_max*self.nparams,1))(li)

        # channel parameter input
        x = tfkl.Input((self.npaths_max*self.nparams,1), name='x')

        # concat data and condition
        dat_cond = tfkl.Concatenate(name='dat_cond')([x, li])
        fe = tfkl.Flatten()(dat_cond)
        layer_names = []
        for i in range(len(self.nunits_dsc)):           
            fe = tfkl.Dense(self.nunits_dsc[i],
                           name='FC%d' % i)(fe)
            fe = tfkl.LeakyReLU(alpha=0.2)(fe)
            fe = tfkl.Dropout(0.3)(fe)
            layer_names.append('FC%d' % i)

        # output
        out_layer = tfkl.Dense(1, activation='linear')(fe)
        # define model
        d_model = tfkm.Model([x, cond], out_layer)
        
        # save network architecture
        # dot_img_file = 'nnArch/dsc.png'
        # tf.keras.utils.plot_model(d_model, to_file=dot_img_file, show_shapes=True) 

        return d_model

class ChanMod(object):
    """
    Object for modeling mmWave channel model data.
    
    There are two parts in the model:
        * link_mod:  This predicts the link_state (i.e. LOS, NLOS or no link)
          from the link conditions.  This is implemented a neural network
        * path_mod:  This predicts the other channel parameters (right now,
          this is the vector of path losses) from the condition and link_state.
        
    Each model has a pre-processor on the data and conditions that is also
    trained.
          
    """    
    def __init__(self, cfg=None, nlatent=50,\
                 nunits_link=(25,10), add_zero_los_frac=0.1,\
                 model_dir='models'):
        """
        Constructor

        Parameters
        ----------
        nunits_link:  list of integers
            number of hidden units in each layer of the link classifier
        nlatent : int
            number of latent states in the GAN model 
        add_zero_los_frac: scalar
            in the link state modeling, a fraction of points at the origin
            are added to ensure the model predicts a LOS link there.
        model_dir : string
            path to the directory for all the model files.
            if this path does not exist, it will be created 
        """
        
        if cfg is None:
            cfg = DataConfig()
        
        self.ndim = 3  # number of spatial dimensions
        self.nunits_link = nunits_link
        self.model_dir = model_dir
        
        self.nlatent = nlatent
        self.add_zero_los_frac = add_zero_los_frac     
        self.rx_types = cfg.rx_types
        self.npaths_max = cfg.npaths_max
        self.max_ex_pl = cfg.max_ex_pl

        # Arbitrary Freq
        self.freq_set = cfg.freq_set
        self.nfreq = cfg.nfreq
        self.nparams = cfg.nfreq + 5
        
        # File names
        self.config_fn = 'config.p'
        self.loss_hist_fn = 'loss_hist.p'
        self.link_weights_fn='link_weights.h5'
        self.link_preproc_fn='link_preproc.p'
        self.path_weights_fn='path_weights.h5'
        self.path_preproc_fn='path_preproc.p'
        self.el_preproc_fn='el_preproc.p'
        
        # Version number for the pre-processing file format
        self.version = 0
                
        
    def save_config(self):
        """
        Saves the configuration parameters

        Parameters
        ----------
        config_fn : string
            File name within the model_dir
        """
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        config_path = os.path.join(self.model_dir, self.config_fn)
        
        
        # Save the pre-processors
        with open(config_path,'wb') as fp:
            pickle.dump(\
                [self.freq_set, self.rx_types, self.max_ex_pl, self.npaths_max], fp)
        
    def load_config(self):
        """
        Loads the configuration parameters

        Parameters
        ----------
        config_fn : string
            File name within the model_dir
        """
        # Create the file paths       
        config_path = os.path.join(self.model_dir, self.config_fn)        
        
        # Read the config file
        with open(config_path,'rb') as fp:
            self.freq_set, self.rx_types, self.max_ex_pl, self.npaths_max =\
                pickle.load(fp)      
        
    def transform_link(self,dvec,rx_type,fit=False):
        """
        Pre-processes input for the link classifier network

        Parameters
        ----------
        dvec : (nlink,3) array
            vector from cell to UAV
        rx_type : (nlink,) array of ints
            RX type of each link

        Returns
        -------
        X:  (nlink,nin_link) array:
            transformed data for input to the NN
        """
        
        # 3D distance and vertical distance.
        # Note that vertical distance can be negative
        dx = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2)
        dz = dvec[:,2]
        
        # Create the one hot encoder for the rx_type
        if fit:
            self.rx_type_enc = sklearn.preprocessing.OneHotEncoder(\
                sparse=False)
            rx_one = self.rx_type_enc.fit_transform(rx_type[:,None])
        else:
            rx_one = self.rx_type_enc.transform(rx_type[:,None])
        n = rx_one.shape[0]
        nt = rx_one.shape[1]
        
        # Transformed input creates one (dx,dz) pair for each
        # RX type
        X0 = np.zeros((n, nt*2), dtype=np.float32)
        for i in range(nt):
            X0[:,2*i] = rx_one[:,i]*dx
            X0[:,2*i+1] = rx_one[:,i]*dz      
        
        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.link_scaler = sklearn.preprocessing.StandardScaler()
            X = self.link_scaler.fit_transform(X0)
        else:
            X = self.link_scaler.transform(X0)
        return X
            
    def build_link_mod(self):
        """
        Builds the link classifier neural network            
        """             
        
        # Compute number of inputs to the link model
        nt = len(self.rx_types)
        self.nin_link = 2*nt
        
        # Input layer
        self.link_mod = tfkm.Sequential()
        self.link_mod.add(tfkl.Input(self.nin_link, name='Input'))
        
        # Hidden layers
        for i, nh in enumerate(self.nunits_link):
            self.link_mod.add(tfkl.Dense(nh, activation='sigmoid', name='FC%d' % i))
        
        # Output softmax for classification
        self.link_mod.add(tfkl.Dense(LinkState.nlink_state,\
                                     activation='softmax', name='Output'))
              
    def add_los_zero(self,dvec,rx_type,ls):
        """
        Appends points at dvec=0 with LOS.  This is used to 
        ensure the model predicts a LOS link at zero distance.

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        rx_type : (nlink,) array of ints
            cell type. 
        ls : (nlink,) array of ints
            link types

        Returns
        -------
        dvec, rx_type, ls : as above
            Values with the zeros appended at the end

        """
        
        ns = dvec.shape[0]
        nadd = int(ns*self.add_zero_los_frac)
        if nadd <= 0:
            return dvec, rx_type, ls
        
        I = np.random.randint(ns,size=(nadd,))
        
        # Variables to append
        rx_type1 = rx_type[I]
        z = dvec[I,2]
        ls1 = np.tile(LinkState.los_link, nadd)
        dvec1 = np.zeros((nadd,3))
        dvec1[:,2] = np.maximum(z,0)
        
        # Add the points
        rx_type = np.hstack((rx_type, rx_type1))
        ls = np.hstack((ls, ls1))
        dvec = np.vstack((dvec, dvec1))
        return dvec, rx_type, ls
           
    def fit_link_mod(self, train_data, test_data, epochs=50, lr=1e-4):
        """
        Trains the link classifier model

        Parameters
        ----------
        train_data : dictionary
            training data dictionary.
        test_data : dictionary
            test data dictionary.    
        """      
                
        # Get the link state
        ytr = train_data['link_state']
        yts = test_data['link_state']        
        
        # Get the position and cell types
        dvectr = train_data['dvec']
        rx_type_tr = train_data['rx_type']
        dvects = test_data['dvec']
        rx_type_ts = test_data['rx_type']
        
        # Fit the transforms
        self.transform_link(dvectr,rx_type_tr, fit=True)

        # Append the zero points 

        dvectr, rx_type_tr, ytr = self.add_los_zero(dvectr,rx_type_tr,ytr)
        dvects, rx_type_ts, yts = self.add_los_zero(dvects,rx_type_ts,yts)
                        
        # Transform the input to the neural network
        Xtr = self.transform_link(dvectr,rx_type_tr)
        Xts = self.transform_link(dvects,rx_type_ts)
                    
        # Fit the neural network
        opt = Adam(lr=lr)
        self.link_mod.compile(opt,loss='sparse_categorical_crossentropy',\
                metrics=['accuracy'])
        
        self.link_hist = self.link_mod.fit(\
                Xtr,ytr, batch_size=100, epochs=epochs, validation_data=(Xts,yts) )            
            
    def link_predict(self,dvec,rx_type):
        """
        Predicts the link state

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        rx_type : (nlink,) array of ints
            cell type.  0 = terrestrial, 1=aerial

        Returns
        -------
        prob:  (nlink,nlink_states) array:
            probabilities of each link state

        """
        X = self.transform_link(dvec, rx_type)
        prob = self.link_mod.predict(X)
        return prob
    
    def save_link_model(self):
        """
        Saves link state predictor model data to files
     
        """
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        preproc_path = os.path.join(self.model_dir, self.link_preproc_fn)
        weigths_path = os.path.join(self.model_dir, self.link_weights_fn)

        # Serializing sklearn objects may not be valid if the de-serializing
        # program has a different sklearn version.  So, we store parameters
        # of the pre-processors instead
        link_param = preproc_to_param(self.link_scaler, 'StandardScaler')
        rx_type_param = preproc_to_param(self.rx_type_enc, 'OneHotEncoder')
        
        # Save the pre-processors
        with open(preproc_path,'wb') as fp:
            pickle.dump([self.version, link_param, self.nunits_link,\
                         rx_type_param], fp)
            
        # Save the weights
        self.link_mod.save_weights(weigths_path, save_format='h5')
        
    def load_link_model(self):
        """
        Load link state predictor model data from files
     
        """
        # Create the file paths
        preproc_path = os.path.join(self.model_dir, self.link_preproc_fn)
        weigths_path = os.path.join(self.model_dir, self.link_weights_fn)
                
        # Load the pre-processors and model config
        with open(preproc_path,'rb') as fp:
            ver, link_scaler_param, self.nunits_link, rx_type_param \
                = pickle.load(fp)
            
       # Construct the pre-processors from the saved parameters
        self.link_scaler = param_to_preproc(link_scaler_param, 'StandardScaler')
        self.rx_type_enc = param_to_preproc(rx_type_param, 'OneHotEncoder')
            
        # Build the link state predictor
        self.build_link_mod()
        
        # Load the weights
        self.link_mod.load_weights(weigths_path)
        
    def build_path_mod(self):
        """
        Builds the GAN for the NLOS paths
        """
        
        # Number of data inputs in the transformed domain
        # For each sample and each path, there is:
        # * two path loss value
        # * nangle angles
        # * one delay
        # for a total of (3+nangle)*npaths_max parameters
        self.ndat = self.npaths_max*(3+AngleFormat.nangle)
        
        # Number of condition variables
        #   * d3d
        #   * log10(d3d)
        #   * los
        #   * rx_type  one-hot encoded dropping final type
        self.ncond = 3 + len(self.rx_types)-1

        self.path_mod = CondGAN(\
            nlatent=self.nlatent, ncond=self.ncond,\
            nparams=self.nparams, npaths_max=self.npaths_max)

    
    def discriminator_loss(self, real_logits, fake_logits):
        # Define the loss functions for the discriminator,
        # which should be (fake_loss - real_loss).
        # We will add the gradient penalty later to this loss function.
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        total_loss = fake_loss - real_loss
        return total_loss

    def generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)

    def gradient_penalty(self, batch_size, real, fake, conds):
        epsilon =  tf.random.normal([batch_size,1], 0.0, 1.0)
        diff = real-fake
        interpolated = fake - epsilon*diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.path_mod.discriminator([interpolated, conds], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
            
    def transform_dly(self, dvec, nlos_dly, fit=False):
        """
        Performs the transformation on the delay data

        Parameters
        ----------
        dvec : (nlink,ndim) array, ndim=3
            Vectors from cell to UAV for each link
        nlos_dly : (nlink,npaths_max) array 
            Absolute delay of each path in each link  
        fit:  boolean
            Indicates if transform is to be fit

        Returns
        -------
        Xdly : (nlink,npaths_max)
            Tranformed delay coordinates

        """            
        
        # Compute LOS delay
        dist = np.sqrt(np.sum(dvec**2,axis=1))        
        los_dly = dist/PhyConst.light_speed
        
        # Compute delay relative to LOS delay
        dly0 = np.maximum(0, nlos_dly - los_dly[:,None])
                
        # Transform the data with the scaler
        # If fit is set, the transform is also learned
        if fit:            
            self.dly_scale = np.mean(dly0)
        Xdly = dly0 / self.dly_scale
            
        return Xdly
    
    def inv_transform_dly(self, dvec, Xdly):
        """
        Performs the inverse transformation on the delay data

        Parameters
        ----------
        dvec : (nlink,ndim) array, ndim=3
            Vectors from cell to UAV for each link
        Xdly : (nlink,npaths_max)
            Tranformed delay coordinates

        Returns
        -------            
        nlos_dly : (nlink,npaths_max) array 
            Absolute delay of each path in each link  
        """            
        
        # Compute LOS delay
        dist = np.sqrt(np.sum(dvec**2,axis=1))        
        los_dly = dist/PhyConst.light_speed

        # Inverse the transform
        dly0 = Xdly * self.dly_scale
        
        # Compute the absolute delay
        nlos_dly = dly0 + los_dly[:,None]
               
        return nlos_dly
            
    def transform_ang(self, dvec, fspl_ls, nlos_ang, nlos_pl_ls, fit=False):
        """
        Performs the transformation on the angle data

        Parameters
        ----------
        dvec : (nlink,ndim) array
            Vectors from cell to UAV for each link
        fspl1 & fspl2: (nlink,)
        nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees
        nlos_pl : (nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of 'fspl1+max_ex_pl' indicates no path
        nlos_pl2 : (nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of 'fspl2+max_ex_pl' indicates no path

        Returns
        -------
        Xang : (nlink,nangle*npaths_max)
            Tranformed angle coordinates

        """
                
        # Compute the LOS angles
        _, los_aod_phi, los_aod_theta = cart_to_sph(dvec)
        _, los_aoa_phi, los_aoa_theta = cart_to_sph(-dvec)
        
        # Get the NLOS angles
        nlos_aod_phi   = nlos_ang[:,:,AngleFormat.aod_phi_ind]
        nlos_aod_theta = nlos_ang[:,:,AngleFormat.aod_theta_ind]
        nlos_aoa_phi   = nlos_ang[:,:,AngleFormat.aoa_phi_ind]
        nlos_aoa_theta = nlos_ang[:,:,AngleFormat.aoa_theta_ind]
        
        # Rotate the NLOS angles by the LOS angles to compute
        # the relative angle        
        aod_phi_rel, aod_theta_rel = spherical_add_sub(\
            nlos_aod_phi, nlos_aod_theta,\
            los_aod_phi[:,None], los_aod_theta[:,None])
        aoa_phi_rel, aoa_theta_rel = spherical_add_sub(\
            nlos_aoa_phi, nlos_aoa_theta,\
            los_aoa_phi[:,None], los_aoa_theta[:,None])            
        
        I = (nlos_pl_ls[0,:,:]-fspl_ls[0,:,None] < self.max_ex_pl)
        # Set the relative angle on non-existent paths to zero
        for i in range(1,self.nfreq):
            I_i = (nlos_pl_ls[i,:,:]-fspl_ls[i,:,None] < self.max_ex_pl)
            I = np.logical_or(I,I_i)

        # theta: el; phi: az
        aod_phi_rel = aod_phi_rel*I
        aod_theta_rel = aod_theta_rel*I 
        aoa_phi_rel = aoa_phi_rel*I
        aoa_theta_rel = aoa_theta_rel*I

        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.aod_el_scaler = sklearn.preprocessing.StandardScaler()
            self.aoa_el_scaler = sklearn.preprocessing.StandardScaler()
            aod_theta_standard = self.aod_el_scaler.fit_transform(aod_theta_rel)
            aoa_theta_standard = self.aoa_el_scaler.fit_transform(aoa_theta_rel)
            # Stack the relative angles and scale by 180
            Xang = np.hstack(\
                (aoa_phi_rel/180, aoa_theta_standard,\
                 aod_phi_rel/180, aod_theta_standard))
        else:
            Xang = np.hstack(\
                (aoa_phi_rel/180, aoa_theta_rel/180,\
                 aod_phi_rel/180, aod_theta_rel/180))  
        
        return Xang
    
    def inv_transform_ang(self, dvec, Xang):
        """
        Performs the transformation on the angle data

        Parameters
        ----------
        dvec : (nlink,ndim) array
            Vectors from cell to UAV for each link
        Xang : (nlink,nangle*npaths_max)
            Tranformed angle coordinates            
   

        Returns
        -------
        nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees        
        """
                
        # Compute the LOS angles
        _, los_aod_phi, los_aod_theta = cart_to_sph(dvec)
        _, los_aoa_phi, los_aoa_theta = cart_to_sph(-dvec)
        
        # Get the transformed angles
        npm = self.npaths_max
        aoa_phi_rel   = Xang[:,:npm]*180
        aoa_theta_rel = self.aoa_el_scaler.inverse_transform(Xang[:,npm:2*npm])       
        aod_phi_rel   = Xang[:,2*npm:3*npm]*180
        aod_theta_rel = self.aod_el_scaler.inverse_transform(Xang[:,3*npm:])

        # Rotate the relative angles by the LOS angles to compute
        # the original NLOS angles
        nlos_aoa_phi, nlos_aoa_theta = spherical_add_sub(\
            aoa_phi_rel, aoa_theta_rel,\
            los_aoa_phi[:,None], los_aoa_theta[:,None], sub=False)
        nlos_aod_phi, nlos_aod_theta = spherical_add_sub(\
            aod_phi_rel, aod_theta_rel,\
            los_aod_phi[:,None], los_aod_theta[:,None], sub=False)
            
        # Stack the relative angles     
        nlink = nlos_aod_phi.shape[0]
        nlos_ang = np.zeros((nlink,self.npaths_max,AngleFormat.nangle))
        nlos_ang[:,:,AngleFormat.aoa_phi_ind] = nlos_aoa_phi
        nlos_ang[:,:,AngleFormat.aoa_theta_ind] = nlos_aoa_theta
        nlos_ang[:,:,AngleFormat.aod_phi_ind] = nlos_aod_phi
        nlos_ang[:,:,AngleFormat.aod_theta_ind] = nlos_aod_theta
        
        return nlos_ang
                        
    def transform_cond(self, dvec, rx_type, los, fit=False):
        """
        Pre-processing transform on the condition

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        rx_type : (nlink,) array of ints
            cell type.  One of terr_cell, aerial_cell
        los:  (nlink,) array of booleans
            indicates if link is in LOS or not
        fit : boolean
            flag indicating if the transform should be fit

        Returns
        -------
        U : (nlink,ncond) array
            Transform conditioned features
        """                     
        
        # 3D distance and vertical distance.
        # Note that vertical distance can be negative
        d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
        d3d = np.round(d3d)
        dvert = dvec[:,2]
        
        # Transform the condition variables
        nt = len(self.rx_types) 
        if (nt > 1):
            rx_one = self.rx_type_enc.transform(rx_type[:,None])
            rx_one = rx_one[:,:nt-1]            
            U0 = np.column_stack((d3d, np.log10(d3d), dvert, los, rx_one))
        else:
            U0 = np.column_stack((d3d, np.log10(d3d), los))
        self.ncond = U0.shape[1]
        
        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.cond_scaler = sklearn.preprocessing.StandardScaler()
            U = self.cond_scaler.fit_transform(U0)
        else:
            U = self.cond_scaler.transform(U0)
            
        return U
    
    def transform_pl(self, fspl_ls, nlos_pl_ls, fit=False):
        """
        Transform on the NLOS path loss

        Parameters
        ----------
        fspl_ls: (nfreq, nlink) array
        nlos_pl_ls: (nfreq, nlink, npaths_max) array  
            path losses of each NLOS path in each link for two freqs.
            A value of 'fspl+max_ex_pl' indicates no path
        fit : boolean
            flag indicating if the transform should be fit            

        Returns
        -------
        Xpl: (nlink, nfreq*npaths_max) array
            Transform data features
        """

        # Compute the path loss below the maximum path loss.
        # Hence a value of 0 corresponds to a maximum path loss value
        nlink = nlos_pl_ls.shape[1]
        ex_pl = nlos_pl_ls[0,:,:self.npaths_max]-fspl_ls[0,:,None]
        ex_pl = np.reshape(ex_pl, (nlink, self.npaths_max))
        for i in range(1,self.nfreq):
            ex_pl_i = nlos_pl_ls[i,:,:self.npaths_max]-fspl_ls[i,:,None]
            ex_pl_i = np.reshape(ex_pl_i, (nlink, self.npaths_max))
            ex_pl = np.concatenate((ex_pl, ex_pl_i), axis=1)

        # Transform the data with the scaler.
        Xpl = np.maximum(0, 1-ex_pl/self.max_ex_pl)

        return Xpl
    
    def inv_transform_pl(self, fspl_ls, Xpl):
        """
        Inverts the transform on the NLOS path loss data

        Parameters
        ----------
        fspl_ls: (nfreq, nlink) array
        Xpl: (nlink, nfreq*npaths_max)

        Returns
        -------
        nlos_pl_ls : (nfreq, nlink, npaths_max) array 
            Path losses of each NLOS path in each link.
            A value of 'fspl+max_ex_pl' indicates no path
        """
        # Invert the scaler
        nlink = Xpl.shape[0]
        Xpl = np.maximum(0,Xpl)
        Xpl = np.minimum(1,Xpl)

        ex_pl = (1-Xpl)*self.max_ex_pl
        nlos_pl_ls = np.zeros((self.nfreq, nlink, self.npaths_max))
        for i in range(self.nfreq):
            ex_pl_i = ex_pl[:, i*self.npaths_max:(i+1)*self.npaths_max]
            nlos_pl_i = ex_pl_i + fspl_ls[i, :, None]
            nlos_pl_i = np.sort(nlos_pl_i, axis = -1)
            nlos_pl_ls[i,:,:] = nlos_pl_i

        if self.nfreq == 2:
            pl_omni = -10*np.log10(np.sum(10**(-0.1*nlos_pl_ls[0,:,:]),axis=1))
            pl2_omni = -10*np.log10(np.sum(10**(-0.1*nlos_pl_ls[1,:,:]),axis=1))

            idx = np.where(pl2_omni <= (pl_omni + 20*np.log10(self.freq_set[1]/self.freq_set[0])))[0]

            for i in idx:
                valid_path_idx = np.where(nlos_pl_ls[0,i,:] != np.max(nlos_pl_ls[0,i,:]))[0]
                nlos_pl_ls[1,i,valid_path_idx] = nlos_pl_ls[1,i,valid_path_idx] + 20*np.log10(self.freq_set[1]/self.freq_set[0])
                
        return nlos_pl_ls    
        
    def transform_data(self, dvec, fspl_ls, nlos_pl_ls, nlos_ang, nlos_dly, fit=False):
        """
        Pre-processing transform on the data

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        fspl_ls: (nfreq, nlink) array 
        nlos_pl_ls:(nfreq, nlink, npaths_max) array 
            Path losses of each path in each link
            A value of 'fspl+max_ex_pl' indicates no path
        nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees           
        nlos_dly : (nlink,npaths_max) array 
            Absolute delay of each path (in seconds)
        fit : boolean
            flag indicating if the transform should be fit            

        Returns
        -------
        X : (nlink, npaths_max*(nfreq + nangle + 1)) array
            Transform data features
        """
        
        # Transform the path loss data
        Xpl = self.transform_pl(fspl_ls,nlos_pl_ls,fit) # (nlink, nfreq*npaths_max) array
        
        # Transform the angles
        Xang = self.transform_ang(dvec,fspl_ls,nlos_ang,nlos_pl_ls,fit) # (nlink,nangle*npaths_max)
        
        # Transform the delays
        Xdly = self.transform_dly(dvec, nlos_dly, fit) # (nlink,npaths_max)
        
        # Concatenate
        X = np.hstack((Xpl, Xang, Xdly)) # (nlink,npaths_max * (nfreq + nangle + 1))
        return X
    
    def inv_transform_data(self, dvec, fspl_ls, X):
        """
        Inverts the pre-processing transform on the data

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from gNB to UE
        fspl_ls: (nfreq, nlink) array             
        X : (nlink, npaths_max*(nfreq + nangle + 1)) array 
            Transform data features

        Returns
        -------
        nlos_pl_ls : (nfreq, nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of 'fspl1+max_ex_pl' indicates no path
        nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees
        nlos_dly : (nlink,npaths_max) array 
            Absolute delay of each path (in seconds)            
        """
        
        # Split

        Xpl = X[:, :self.nfreq*self.npaths_max]
        Xang = X[:,self.nfreq*self.npaths_max:self.npaths_max*(AngleFormat.nangle+self.nfreq)]
        Xdly = X[:,self.npaths_max*(AngleFormat.nangle+self.nfreq):]

        # Invert the transforms
        nlos_pl_ls = self.inv_transform_pl(fspl_ls, Xpl) # nlos_pl_ls : (nfreq, nlink, npaths_max) array 
        nlos_ang = self.inv_transform_ang(dvec, Xang)
        nlos_dly = self.inv_transform_dly(dvec, Xdly)
                
        return nlos_pl_ls, nlos_ang, nlos_dly 
    
    def get_los_path(self, dvec):
        """
        Computes LOS path loss and angles

        Parameters
        ----------
        dvec : (n,3) array            
            Vector from cell to UAV
            
        Returns
        -------
        los_pl_ls:  (nfreq, n) array
            LOS path losses computed from Friis' Law
        los_ang:  (n,AngleFormat.nangle) = (n,4) array
            LOS angles 
        los_dly:  (n,) array
            Delay of the paths computed from the speed of light
        """
        # Compute free space path loss from Friis' law
        dist = np.maximum(np.sqrt(np.sum(dvec**2,axis=1)), 1)        

        lam_ls = PhyConst.light_speed/np.array(self.freq_set)
        los_pl_ls = 20*np.log10(dist[None,:]*4*np.pi/lam_ls[:,None]) # (nfreq, n) array
        
        # Compute the LOS angles
        _, los_aod_phi, los_aod_theta = cart_to_sph(dvec)
        _, los_aoa_phi, los_aoa_theta = cart_to_sph(-dvec)
        
        # Stack the angles
        los_ang = np.stack((los_aoa_phi, los_aoa_theta,\
                            los_aod_phi, los_aod_theta), axis=-1)
            
        # Compute the delay
        los_dly = dist/PhyConst.light_speed
    
        return los_pl_ls, los_ang, los_dly
        
    def sample_path(self, dvec, fspl_ls, rx_type, link_state=None, return_dict=False):
        """
        Generates random samples of the path data using the trained model

        Parameters
        ----------
        dvec : (nlink,ndim) array
            Vector from cell to UAV
        fspl_ls: (nfreq, nlink) array 
        rx_type : (nlink,) array of ints
            Cell type.  One of terr_cell, aerial_cell
        link_state:  (nlink,) array of {no_link, los_link, nlos_link}            
            A value of `None` indicates that the link state should be
            generated randomly from the link state predictor model
        return_dict:  boolean, default False
            If set, it will return a dictionary with all the values
            Otherwise it will return a channel list
   
        Returns
        -------
        chan_list:  (nlink,) list of MPChan object
            List of random channels from the model.  Returned if
            return_dict == False
        data:  dictionary
            Dictionary in the same format as the data.
            Returned if return_dict==True
        """
        # Get dimensions
        nlink = dvec.shape[0]

        # Generate random link states if needed
        # Use the link state predictor network
        if link_state is None:
            prob = self.link_predict(dvec, rx_type) 
            cdf = np.cumsum(prob, axis=1)            
            link_state = np.zeros(nlink)
            u = np.random.uniform(0,1,nlink)
            for i in range(cdf.shape[1]-1):
                I = np.where(u>cdf[:,i])[0]
                link_state[I] = i+1
                
        # Find the indices where there are some link
        # and where there is a LOS link
        Ilink = np.where(link_state != LinkState.no_link)[0]
        Ilos  = np.where(link_state == LinkState.los_link)[0]
        los   = link_state == LinkState.los_link        
        
        # Get the condition variables and random noise
        U = self.transform_cond(dvec[Ilink], rx_type[Ilink], los[Ilink])
        nlink1 = U.shape[0]
        Z = np.random.normal(0,1,(nlink1,self.nlatent))
        
        # Run through the generator network
        X = self.path_mod.generator.predict([Z,U]) 
        
        # Compute the inverse transform to get back the path loss
        # and angle data
        # nlos_pl1, nlos_pl22, nlos_ang1 , nlos_dly1 = self.inv_transform_data(dvec[Ilink], fspl_ls[:,Ilink], X)
        nlos_pl_ls, nlos_ang1 , nlos_dly1 = self.inv_transform_data(dvec[Ilink], fspl_ls[:,Ilink], X)
        
        # Create arrays for the NLOS paths
        nlos_pl_ls_record = np.zeros((self.nfreq, nlink, self.npaths_max))
        # nlos_pl2 = np.zeros((nlink, self.npaths_max))
        for k, _ in enumerate(self.freq_set):
            for i in range(nlink):
                nlos_pl_ls_record[k,i,:] = np.tile(fspl_ls[k,i]+self.max_ex_pl, self.npaths_max).astype(np.float32)

    
        nlos_ang = np.zeros((nlink,self.npaths_max,AngleFormat.nangle), dtype=np.float32)
        nlos_dly  = np.zeros((nlink,self.npaths_max), dtype=np.float32)
        nlos_pl_ls_record[:,Ilink,:] = nlos_pl_ls
        nlos_ang[Ilink] = nlos_ang1
        nlos_dly[Ilink]  = nlos_dly1
        
        # Compute the PL and angles for the LOS paths
        los_pl_ls, los_ang, los_dly = self.get_los_path(dvec[Ilos])
        
        # Create arrays for the LOS paths
        los_pl_ls_record = np.zeros((self.nfreq, nlink), dtype=np.float32)

        los_ang_record = np.zeros((nlink,AngleFormat.nangle), dtype=np.float32)
        los_dly_record  = np.zeros((nlink,), dtype=np.float32)

        los_pl_ls_record[:,Ilos] = los_pl_ls
        los_ang_record[Ilos] = los_ang
        los_dly_record[Ilos]  = los_dly
        
        # Store in a data dictionary
        data = dict()
        data['dvec'] = dvec

        for ifreq, _ in enumerate(self.freq_set):
            if ifreq == 0:
                data['nlos_pl'] = nlos_pl_ls_record[ifreq,:,:]
                data['los_pl'] = los_pl_ls_record[ifreq,:]

            else:
                data['nlos_pl'+str(ifreq+1)] = nlos_pl_ls_record[ifreq,:,:]
                data['los_pl'+str(ifreq+1)] = los_pl_ls_record[ifreq,:]
            data['fspl'+str(ifreq+1)] = fspl_ls[ifreq,:]


        data['rx_type'] = rx_type
        data['link_state'] = link_state
        data['nlos_dly'] = nlos_dly
        data['nlos_ang'] = nlos_ang
        data['los_dly'] = los_dly_record
        data['los_ang'] = los_ang_record
        
        if return_dict:
            return data
        
        # Config
        cfg = DataConfig()
        cfg.freq_set = self.freq_set
        cfg.rx_types = self.rx_types
        cfg.nfreq = self.nfreq
        cfg.npaths_max = self.npaths_max
        cfg.max_ex_pl = self.max_ex_pl
        
        # Create list of channels
        chan_list, link_state = data_to_mpchan(data, cfg)
        
        return chan_list, link_state
            
    def save_path_preproc(self):
        """
        Saves path preprocessor
        """
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        preproc_path = os.path.join(self.model_dir, self.path_preproc_fn)
        el_path = os.path.join(self.model_dir, self.el_preproc_fn)
        
        # Serializing sklearn objects may not be valid if the de-serializing
        # program has a different sklearn version.  So, we store parameters
        # of the pre-processors instead
        cond_param = preproc_to_param(self.cond_scaler, 'StandardScaler')
        aod_el_param = preproc_to_param(self.aod_el_scaler, 'StandardScaler')
        aoa_el_param = preproc_to_param(self.aoa_el_scaler, 'StandardScaler')
        
        # Save the pre-processors
        with open(preproc_path,'wb') as fp:
            pickle.dump([self.version, cond_param, self.dly_scale,\
                         self.max_ex_pl, self.npaths_max, self.nlatent], fp)
        with open(el_path,'wb') as fp:
            pickle.dump([aod_el_param, aoa_el_param], fp)
    
    def save_path_model(self):
        """
        Saves model data to files

        """        
        
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        weigths_path = os.path.join(self.model_dir, self.path_weights_fn)

            
        # Save the GAN weights
        self.path_mod.generator.save_weights(weigths_path, save_format='h5')
        
    def load_path_model(self, ckpt=None):
        """
        Load model data from files
        
        Parameters
        ----------
        ckpt : None or int
            If integer, loads a checkpoint file with the epoch number.

        """
        # Create the file paths
        preproc_path = os.path.join(self.model_dir, self.path_preproc_fn)
        el_path = os.path.join(self.model_dir, self.el_preproc_fn)
        if ckpt is None:
            fn = self.path_weights_fn
        else:
            fn = ('path_weights.%d.h5' % ckpt)
        weights_path = os.path.join(self.model_dir, fn)
        
        # Load the pre-processors
        with open(preproc_path,'rb') as fp:
            ver, cond_param, self.dly_scale, self.max_ex_pl,\
                self.npaths_max, self.nlatent = pickle.load(fp)

        with open(el_path,'rb') as fp:
            aod_el_param, aoa_el_param = pickle.load(fp)
                
        # Re-constructor the pre-processors
        self.cond_scaler = param_to_preproc(cond_param, 'StandardScaler')       
        self.aod_el_scaler = param_to_preproc(aod_el_param, 'StandardScaler')
        self.aoa_el_scaler = param_to_preproc(aoa_el_param, 'StandardScaler')         
            
        # Build the path model
        self.build_path_mod()
            
        # Load the GAN weights
        self.path_mod.generator.load_weights(weights_path)

    def fit_path_mod(self, train_data, test_data, epochs=50, lr=1e-4,\
                     checkpoint_period = 100, save_mod=True, batch_size=512,\
                     d_steps=5, gp_weight=10):
        """
        Trains the path model
        

        Parameters
        ----------
        train_data : dictionary
            training data dictionary.
        test_data : dictionary
            test data dictionary. 
        epochs: int
            number of training epochs
        lr: scalar
            learning rate
        checkpoint_period:  int
            period in epochs for saving the model checkpoints.  
            A value of 0 indicates that checkpoints are not be saved.
        save_mod:  boolean
            Indicates if model is to be saved
        batch_size: int
        d_steps: int
            train the discriminator for d_steps more steps 
            as compared to one step of the generator
        gp_weight: int
            gradient penalty weights
        """      
        # Get the link state
        gen_loss = []
        dsc_loss = []
        ls_tr = train_data['link_state']
        los_tr = (ls_tr == LinkState.los_link)
        
        
        # Extract the links that are in LOS or NLOS
        Itr = np.where(ls_tr != LinkState.no_link)[0]
        
        # Fit and transform the condition data
        Utr = self.transform_cond(\
            train_data['dvec'][Itr], train_data['rx_type'][Itr],\
            los_tr[Itr], fit=True)        
        
        # Fit and transform the data
        train_fspl_ls = np.zeros((self.nfreq, len(Itr))) # (nfreq, nlink)
        train_nlos_pl_ls = np.zeros((self.nfreq, len(Itr), self.npaths_max)) # nlos_pl_ls:(nfreq, nlink, npaths_max) array 
        
        for ifreq in range(self.nfreq):
            train_fspl_ls[ifreq,:] = train_data['fspl'+str(ifreq+1)][Itr]
            if ifreq == 0:
                train_nlos_pl_ls[ifreq,:,:] = train_data['nlos_pl'][Itr,:self.npaths_max]
            else:
                train_nlos_pl_ls[ifreq,:,:] = train_data['nlos_pl'+str(ifreq+1)][Itr,:self.npaths_max]
            
        Xtr = self.transform_data(\
            train_data['dvec'][Itr],\
            train_fspl_ls,\
            train_nlos_pl_ls,\
            train_data['nlos_ang'][Itr,:self.npaths_max,:],\
            train_data['nlos_dly'][Itr,:self.npaths_max], fit=True)
        
        # store dvec min and max
        self.dvect_max = np.max(train_data['dvec'][Itr], axis=0)
        self.dvect_min = np.min(train_data['dvec'][Itr], axis=0)

        # Save the pre-processor
        if save_mod:
            self.save_path_preproc()
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        weigths_path = self.model_dir
        discriminator = self.path_mod.discriminator
        generator = self.path_mod.generator

        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0, beta_2=0.9)

        for epoch in range(epochs):

            I = np.random.permutation(Xtr.shape[0])

            nsteps = len(I)//batch_size

            # For each batch, we are going to perform the
            # following steps as laid out in the original paper:
            # 1. Train the generator and get the generator loss
            # 2. Train the discriminator and get the discriminator loss
            # 3. Calculate the gradient penalty
            # 4. Multiply this gradient penalty with a constant weight factor
            # 5. Add the gradient penalty to the discriminator loss
            # 6. Return the generator and discriminator losses as a loss dictionary

            # Train the discriminator first. The original paper recommends training
            # the discriminator for `x` more steps (typically 5) as compared to
            # one step of the generator.

            for i in range(nsteps):
                
                idx = I[i*batch_size:(i+1)*batch_size]
                Xtrain, labels = Xtr[idx], Utr[idx]

                for j in range(d_steps):
                    # Get the latent vector
                    z = np.random.normal(0, 1, size=(batch_size, self.nlatent))

                    with tf.GradientTape() as tape:
                        # Generate fake channels from the latent vector
                        fake_chans = generator([z, labels], training=True)
                        # Get the logits for the fake channels
                        fake_logits = discriminator([fake_chans, labels], training=True)
                        # Get the logits for the real channels
                        real_logits = discriminator([Xtrain, labels], training=True)

                        # Calculate the discriminator loss using the fake and real channel logits
                        d_cost = self.discriminator_loss(real_logits, fake_logits)
                        # Calculate the gradient penalty
                        gp = self.gradient_penalty(batch_size, Xtrain, fake_chans, labels)
                        # Add the gradient penalty to the original discriminator loss
                        d_loss = d_cost + gp * gp_weight

                    # Get the gradients w.r.t the discriminator loss
                    d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
                    # Update the weights of the discriminator using the discriminator optimizer
                    discriminator_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))

                # Train the generator
                # Get the latent vector
                z = np.random.normal(0, 1, size=(batch_size, self.nlatent))
                with tf.GradientTape() as tape:
                    # Generate fake channels using the generator
                    generated_chans = generator([z, labels], training=True)
                    # Get the discriminator logits for fake channels
                    gen_chan_logits = discriminator([generated_chans, labels], training=True)
                    # Calculate the generator loss
                    g_loss = self.generator_loss(gen_chan_logits)

                # Get the gradients w.r.t the generator loss
                gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
                # Update the weights of the generator using the generator optimizer
                generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))

                #save and print generator and discriminator loss
                gen_loss.append(g_loss.numpy())
                dsc_loss.append(d_loss.numpy())
            
            tf.print(f'Epoch:{epoch} G_loss: {g_loss} D_loss: {d_loss}')

            if epoch % checkpoint_period == 0:
                self.path_mod.generator.save(weigths_path+f'/generator-epochs-{epoch}.h5')

        # Save the weights model
        if save_mod:
            self.save_path_model()   

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        loss_hist_path = os.path.join(self.model_dir, self.loss_hist_fn)
        # Save loss
        with open(loss_hist_path,'wb') as loss_fp:
            pickle.dump(\
                [gen_loss, dsc_loss], loss_fp)

        
def set_initialization(mod, layer_names, kernel_stddev=1.0, bias_stddev=1.0):
    """
    Sets the bias and kernel initializations for a set of dense layers

    Parameters
    ----------
    mod:  Tensorflow model
        Model for which the initialization is to be applied
    layer_names : list of strings
        List of names of layers to apply the initialization
    kernel_stddev : scalar
        std deviation of the kernel in the initialization
    bias_stddev : scalar
        std deviation of the bias in the initialization            
    """
    for name in layer_names:
        layer = mod.get_layer(name)
        nin = layer.input_shape[-1]
        nout = layer.output_shape[-1]
        W = np.random.normal(0,kernel_stddev/np.sqrt(nin),\
                             (nin,nout)).astype(np.float32)
        b = np.random.normal(0,bias_stddev,\
                             (nout,)).astype(np.float32)
        layer.set_weights([W,b])

def load_model(mod_name, cfg):
    """
    Loads a pre-trained model from local directory

    Parameters
    ----------
    mod_name : string
        Model name to be downloaded. 
        
    Returns
    -------
    chan_mod:  ChanMod
        pre-trained channel model
    """    
        
    # Create the local data directory if needed    
    mod_root = os.path.join(os.path.dirname(__file__),'..','..','models')
    mod_root = os.path.abspath(mod_root)
    if not os.path.exists(mod_root):
        os.mkdir(mod_root)
        print('Creating directory %s' % mod_root)
        
    # Check if model directory exists
    mod_dir = os.path.join(mod_root, mod_name)
        
    # Check if model directory exists
    if not os.path.exists(mod_dir):
        raise ValueError('Cannot find model %s' % mod_dir)
        
    # Create the model
    chan_mod = ChanMod(model_dir=mod_dir, cfg = cfg)
    
    # Load the configuration and link classifier model
    chan_mod.load_config()
    chan_mod.load_link_model()
    chan_mod.load_path_model()        

    return chan_mod    