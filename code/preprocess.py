#-*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:31:16 2020

@author: youngjae
"""

import h5py
import os
import tensorflow as tf
import numpy as np

def data_load(dpath, pts,TODO='train_CLS'):
    #data_load for classification train dataset
    if TODO == 'train_CLS':
        dat = h5py.File(dpath+'whole-'+str(pts)+'.hdf5', 'r')
        print(dpath+'whole-'+str(pts)+'.hdf5'+' is loaded!!')
        matx, laby = dat['matx'].value, dat['laby'].value
        m_s = dat['norminfo'].value

        del dat

        #normalize
        for ch in range(matx.shape[-1]):
            matx[:,:,:,ch] = (matx[:,:,:,ch]-m_s[ch][0])/m_s[ch][1]

        #one-hot for classification
        laby = tf.keras.utils.to_categorical(laby, num_classes=3)
        return [matx, laby], m_s
    
    #data_load for segmentation train dataset
    elif TODO == 'train_SEG':
        dat = h5py.File(dpath+'whole-'+str(pts)+'.hdf5', 'r')
        print(dpath+'whole-'+str(pts)+'.hdf5'+' is loaded!!')
        matx, maty = dat['matx'][()], dat['maty'][()]
        coord = dat['coord'][()]
        m_s = dat['norminfo'][()]
        
        maty = maty.astype(np.float32)
        del dat

        #normalize
        for ch in range(matx.shape[-1]):
            matx[:,:,:,ch] = (matx[:,:,:,ch]-m_s[ch][0])/m_s[ch][1]
        
        maty = np.expand_dims(maty, axis=-1)
        return [matx, maty, coord], m_s

    elif TODO == 'inference':
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

def makeup_img(mat, coord, mode='overwrite'):
    true_mat = np.zeros((np.max(coord[:,0])+256, np.max(coord[:,1])+256), dtype=np.float32)

    for p in range(len(coord)):
        sw, sh = coord[p]
        ew, eh = sw+256, sh+256

        org = true_mat[sw:ew, sh:eh]
        new = mat[p]
        if mode == 'overwrite':
            mask = org == 0
            org[mask] = new[mask]
        elif mode == 'max_prob':
            max_ = np.maximum(org, new)
            true_mat[sw:ew, sh:eh] = max_
    
    return true_mat.round()
