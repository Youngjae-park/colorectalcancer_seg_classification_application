#-*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:31:16 2020

@author: youngjae
"""

import numpy as np
import preprocess
from models import CLS_MODEL
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score

dpath = '/home/dhodwo/venv/dataset/whole/'
spath = '/home/dhodwo/venv/results/'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Hyperparameters
lr = 5e-5
epochs = 30
batch = 16
H_info = [lr, epochs, batch]


if __name__ == '__main__':
    if len(sys.argv) == 2:
        NET_NAME = sys.argv[1]
        NN = CLS_MODEL.Network(sys.argv[1], lr)
    else:
        NN = CLS_MODEL.Network('inceptionv3', lr)
    pts = len(os.listdir(dpath))-1 #except norm-info.npz
    NN.placehold(256,256)
    NN.classification_model()
    NN.session_init() #global vars init
    
    ###
    trloss, tracc_cls, trf1 = [], [], []
    #infcla, inference = [], []
    ###
    

    if not os.path.exist(spath+NET_NAME+'/'):
        os.makedirs(spath+NET_NAME+'/', exist_ok=True)

    for ep in range(epochs):
        trloss_, tracccls_, trf1_ = [], [], []
        for trs in range(pts):
            trains, ninfo = preprocess.data_load(dpath, trs, TODO='train')
            ntr = len(trains[0])
            ytr = np.argmax(trains[1], axis=1)
            ypred = []
            iters = int(np.ceil(ntr/batch))
            for it in range(iters):
                b_x, b_yl = trains[0][batch*it:batch*(it+1)], trains[1][batch*it:batch*(it+1)]
                
                flag_aug = np.random.binomial(1, 0.3)
                if flag_aug:
                    b_x = preprocess.data_augmentation(b_x)
                
                #print(b_x.shape, b_yl.shape)
                NN.train(b_x, b_yl)
                loss, predict_label, acc = NN.result_info(b_x, b_yl)

                trloss_.append(loss)
                ypred.append(predict_label)
                tracccls_.append(acc)
            ypred = np.concatenate(ypred)
            trf1_.append(f1_score(ytr, ypred, average='macro'))
        trloss.append(np.average(trloss_))
        tracc_cls.append(np.average(traccls_))
        trf1.append(np.average(trf1_))

    #Save the result
    np.savez(spath+NET_NAME+'/tr_result.npz',loss = trloss, acc = tracc_cls, f1 = trf1, Hyper=H_info)
