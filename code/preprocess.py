#-*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:31:16 2020

@author: youngjae
"""

import h5py
import os
import tensorflow as tf
import numpy as np

def data_load(dpath, pts,TODO='train'):
    if TODO=='train':
        dat = h5py.File(dpath+'whole-'+str(pts)+'.hdf5', 'r')
        print(dpath+'whole-'+str(pts)+'.hdf5'+' is loaded!!')
        # matx, laby = dat['matx'].value, dat['laby'].value
        # m_s = dat['norminfo'].value
        matx, laby = dat['matx'].value, dat['laby'].value
        m_s = dat['norminfo'].value

        del dat

        #normalize
        for ch in range(matx.shape[-1]):
            matx[:,:,:,ch] = (matx[:,:,:,ch]-m_s[ch][0])/m_s[ch][1]

        #one-hot for classification
        laby = tf.keras.utils.to_categorical(laby, num_classes=3)
        return [matx, laby], m_s
    elif TODO=='inference':
        #FILL THE CODE HERE
        pass
        
def data_augmentation(x):
    augx = []
    ridx = np.random.binomial(4,0.5,len(x))
    for bt in range(len(ridx)):
        if ridx[bt] == 0:
            augx.append(np.fliplr(x[bt]))
        elif ridx[bt] == 1:
            augx.append(np.flipud(x[bt]))
        elif ridx[bt] == 2:
            augx.append(np.rot90(x[bt]))
        elif ridx[bt] == 3:
            augx.append(np.rot90(x[bt], 2))
        elif ridx[bt] == 4:
            augx.append(np.rot90(x[bt], 3))

    return np.array(augx)
