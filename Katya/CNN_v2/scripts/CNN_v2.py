import pandas as pd
import pdb
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense, Flatten, Reshape, merge, Highway, Activation,Dropout
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D,GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adamax, Adam, Nadam
from keras.layers.advanced_activations import ELU,PReLU,LeakyReLU
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from keras.regularizers import l2
from pylab import imshow, show
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage import measure, morphology
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
import threading
from multiprocessing import Process
from keras.regularizers import l2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def conv_block(x_input, num_filters, pool=True, norm=True, drop_rate=0.0):

    x1 = Conv3D(num_filters,3,3,3,border_mode='same',W_regularizer=l2(1e-4), data_format='channels_first')(x_input)
    
    if norm:
        x1 = BatchNormalization(axis=1)(x1)
    
    x1 = GaussianDropout(drop_rate)(x1)
    x1 = LeakyReLU(.1)(x1)
    
    if pool:
        x1 = MaxPooling3D(dim_ordering="th")(x1)
        
    x_out = x1
    
    return x_out



def dense_branch(xstart, name, outSize=1, activation='sigmoid'):
    
    xdense = Dense(32, W_regularizer=l2(1e-4))(xstart)
    xdense = BatchNormalization()(xdense)
    xdense = LeakyReLU(.1)(xdense)
    xout = Dense(outSize, activation=activation, name=name, W_regularizer=l2(1e-4))(xdense)
    
    return xout



def build_model(input_shape):

    xin = Input(input_shape)

    x1 = conv_block(xin, 8, drop_rate=0) #outputs 9 ch
    x1_ident = AveragePooling3D(dim_ordering="th")(xin)
    x1_merged = merge([x1, x1_ident], mode='concat', concat_axis=1)
    
    x2_1 = conv_block(x1_merged, 24, drop_rate=0) #outputs 16+9 ch  = 25
    x2_ident = AveragePooling3D(dim_ordering="th")(x1_ident)
    x2_merged = merge([x2_1, x2_ident], mode='concat', concat_axis=1)
    
    #by branching we reduce the #params
    x3_1 = conv_block(x2_merged,64,norm=True,drop_rate=0) #outputs 25 + 16 ch = 41
    x3_ident = AveragePooling3D(dim_ordering="th")(x2_ident)
    x3_merged = merge([x3_1,x3_ident],mode='concat', concat_axis=1)

    x4_1 = conv_block(x3_merged,72,norm=True,drop_rate=0) #outputs 25 + 16 ch = 41
    x4_ident = AveragePooling3D(dim_ordering="th")(x3_ident)
    x4_merged = merge([x4_1,x4_ident],mode='concat', concat_axis=1)
    

    x5_1 = conv_block(x4_merged, 72, norm=True, pool=False, drop_rate=0) #outputs 25 + 16 ch = 41

    xpool = GlobalMaxPooling3D()(x5_1)
    xpool_norm = BatchNormalization()(xpool)

    xout_cad_falsepositive = dense_branch(xpool_norm, name='o_fp', outSize=3, activation='softmax')
    xout_malig = dense_branch(xpool_norm,name='o_mal',outSize=1,activation='sigmoid')
    
    model = Model(input=xin, output=[xout_malig, xout_cad_falsepositive])

    
    if input_shape[1] == 32:
        lr_start = .005
    elif input_shape[1] == 64:
        lr_start = .001
    elif input_shape[1] == 128:
        lr_start = 1e-4
    elif input_shape[1] == 96:
        lr_start = 5e-4
    
    opt = Nadam(lr_start, clipvalue=1.0)
    print ('compiling model')

    model.compile(optimizer=opt,\
                  loss={'o_mal':'binary_crossentropy', 'o_fp':'categorical_crossentropy'})
    return model



def str_to_arr(arr_str,length):
    while '  ' in arr_str:
        arr_str = arr_str.replace('  ', ' ')
    result = eval(arr_str.replace('[ ', '[').replace(' ]', ']').replace(' ', ','))
    assert len(result) == length
    return np.array(result)



def get_generator_static(X1,X2,IX1,IX2,batch_size,augment=True,current_frac=.75):

    df = pd.read_csv('/home/katya/data/CSVFILES/annotations_enhanced.csv')
    
    Ycad_fp = np.zeros((df.shape[0],3)).astype('float32')
    Ycad_fp[:,0] = 1.0
    #3 categories - [nodule, cad false positive, random subvoxel]

    df['malignancy'] /= 5.

    Ymalig = df['malignancy'].values

    while True:
        
        n1 = int(current_frac * batch_size)
        n2 = batch_size - n1
        
        ixs1 = np.random.choice(range(X1.shape[0]), size=n1, replace=False)
        ixs2 = np.random.choice(range(X2.shape[0]), size=n2, replace=False)
        Xbatch1, IXbatch1 = X1[ixs1], IX1[ixs1]
        Xbatch2, IXbatch2 = X2[ixs2], IX2[ixs2]
        
        Xbatch = np.concatenate([Xbatch1, Xbatch2], axis=0)
        IXbatch = np.concatenate([IXbatch1, IXbatch2], axis=0)
        IXbatch_eq_neg2 = (IXbatch == -2)
        IXbatch[IXbatch_eq_neg2] = -1
        #this way things that are -2 (cad false positives)
        #get labeled as non-nodules for everything else.
        #but we still know where they are.
        IXbatch_eq_neg1 = (IXbatch == -1)
        
        #making sure we don't use these.
        #row 32 is a big tumor so it should be obvious
        
        #normalize
        Xbatch = np.expand_dims(Xbatch, 1)
        if augment:
            Xbatch = random_perturb(Xbatch)
        Xbatch = Xbatch.astype('float32')
        Xbatch = (Xbatch + 1000.) / (400. + 1000.)
        Xbatch = np.clip(Xbatch,0,1)
        
        Ybatch_malig = Ymalig[IXbatch]
        Ybatch_malig[IXbatch_eq_neg1] = 0.0
        
        Ybatch_cad_fp = Ycad_fp[IXbatch]
        #class 0 things = nodules are already set.
        
        #-1 = random subvoxel, class 3
        Ybatch_cad_fp[IXbatch_eq_neg1,2] = 1.0
        Ybatch_cad_fp[IXbatch_eq_neg1,0] = 0.0
        Ybatch_cad_fp[IXbatch_eq_neg1,1] = 0.0
        
        #-2 = cad false positive, class 2
        Ybatch_cad_fp[IXbatch_eq_neg2,1] = 1.0
        Ybatch_cad_fp[IXbatch_eq_neg2,0] = 0.0
        Ybatch_cad_fp[IXbatch_eq_neg2,2] = 0.0
                
        
        yield Xbatch, {'o_mal':Ybatch_malig, 'o_fp':Ybatch_cad_fp}        
        
        
        
        
def get_batch_size_for_stage(stage):

    #return np.around( 64. * (64 ** 3) / (stage ** 3))
    if stage == 32:
        batch_size=128
    if stage == 64:
        batch_size=64
    if stage == 128:
        batch_size=8
    if stage == 256:
        batch_size=2
    if stage == 72:
        batch_size=40
    if stage == 65:
        batch_size=63
    if stage == 96:
        batch_size=18
        
    return batch_size




def stage_1_lr_schedule(i):
    if i < 10:
        return np.float32(.004)
    if i < 15:
        return np.float32(.002)
    if i < 20:
        return np.float32(.0005)
    if i < 23:
        return np.float32(.0001)
    return np.float32(3e-5)

def stage_2_lr_schedule(i):
    if i == 0:
        return np.float32(.001)
    if i < 5:
        return np.float32(.0004)
    if i < 10:
        return np.float32(.0002)
    return np.float32(3e-5)








def train_model_on_stage(stage, model):
    
    split=False
    import time
    
    batch_size=get_batch_size_for_stage(stage)
    
    if split:
        Xpos = np.load(r"D:\Dlung\Xpositive_temp_v5.npy")
        IXpos = np.load(r"D:\Dlung\Ixpositive_temp_v5.npy")
        split1 = int(.75*Xpos.shape[0])
        Xtrain1 = Xpos[:split1]
        Xvalid1 = Xpos[split1:]
        IXtrain1 = IXpos[:split1]
        IXvalid1 = IXpos[split1:]
        del Xpos, IXpos
        

        Xneg = np.load(r"D:\Dlung\Xrandom_temp_v5.npy")
        IXneg = np.load(r"D:\Dlung\IXrandom_temp_v5.npy")

        split2 = int(.75*Xneg.shape[0])
        Xtrain2 = Xneg[:split2]
        Xvalid2 = Xneg[split2:]
        IXtrain2 = IXneg[:split2]
        IXvalid2 = IXneg[split2:]
        del Xneg, IXneg
        
        train_generator_75 = get_generator_static(Xtrain1,Xtrain2,IXtrain1,IXtrain2, augment=True, batch_size=batch_size,current_frac=.33)
        valid_generator_75 = get_generator_static(Xvalid1,Xvalid2,IXvalid1,IXvalid2, augment=True, batch_size=batch_size,current_frac=.33)

    else:
        Xpos = np.load('/home/katya/data/voxels_true64/subset0Xtrue.npy')
        IXpos = np.load('/home/katya/data/voxels_true64/subset0IXtrue.npy')
        Xneg = np.load('/home/katya/data/voxels_random64/subset0Xrandom.npy')
        IXneg = np.load('/home/katya/data/voxels_random64/subset0IXrandom.npy')
        train_generator_75 = get_generator_static(Xpos, Xneg, IXpos, IXneg, augment=False, batch_size=batch_size, current_frac=.33)


    name = 'model_LUNA_' + str(stage) + '_v1_{epoch:02d}.h5'
    chkp = ModelCheckpoint(filepath=name)

    if stage == 32:
        lr_schedule = LearningRateScheduler(stage_1_lr_schedule)
        nb_epoch = 25
        samples_per_epoch = 150
    else:
        lr_schedule = LearningRateScheduler(stage_2_lr_schedule)
        nb_epoch=15
        samples_per_epoch = 150
        
    for epoch in range(nb_epoch):
        model.fit_generator(train_generator_75,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[chkp,lr_schedule],
                                    initial_epoch=epoch)

        
    return model


model_64 = build_model((1,64,64,64))
model_64 = train_model_on_stage(64, model_64)
model_64.save_weights('/home/katya/LungCancer/Katya/CNN_v1/model_v1_weights_temp64.h5')
