#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020. 10. 26. (MON) 21:39:12 KST

@author: jinhee
"""

import numpy as np
import preprocess
from models import SEG_MODEL
import os
import sys
from sklearn.metrics import jaccard_score


dpath = '/home/dhodwo/venv/dataset/whole/'
spath = '/home/dhodwo/venv/results/'
w_path = '/home/dhodwo/venv/weights/'
result_path = spath+'SEG/'
if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Hyperparameteres
lr = 5e-5
epochs = 20
batch = 16
H_info = [lr, epochs, batch]


if __name__ == '__main__':
    if len(sys.argv) == 2:
        NET_NAME = sys.argv[1]
        NN = SEG_MODEL.Network(sys.argv[1], lr)
    else:
        NET_NAME = 'vgg16'
        NN = SEG_MODEL.Network(NET_NAME, lr)
    
    # Set the path where weights will be saved
    w_path = w_path+NET_NAME+'/SEG/'
    if not os.path.exists(w_path):
        os.makedirs(w_path, exist_ok=True)

    pts = len(os.listdir(dpath)) - 1   #excep norm-info.npz
    
    NN.placehold_getNN(256,256)
    NN.session_init()
    NN.saver_init(epochs)
    
    trloss, tracc_seg = [], []
    infseg, pinfseg = [], []
    
    print("Training Started")
    for ep in range(epochs):
        trloss_, traccseg_ = [], []
        for trs in range(pts):
            trains, ninfo = preprocess.data_load(dpath, trs, TODO='train_SEG')
            ntr = len(trains[0])
            iters = int(np.ceil(ntr/batch))
            for it in range(iters):
                b_x, b_ym = trains[0][batch*it:batch*(it+1)], trains[1][batch*it:batch*(it+1)]
                ### Data augmentation ###

                #########################
                NN.train(b_x, b_ym)
                loss, b_ymat, pmat = NN.result_info(b_x, b_ym)
                
                trloss_.append(loss)
                tjaccard = jaccard_score(b_ymat, pmat, average='samples')
                traccseg_.append(tjaccard)
        
        #tr
        trloss.append(np.average(trloss_))
        tracc_seg.append(np.average(traccseg_))

        NN.save(w_path+'epoch', ep) #NN.save => global_step = ep+1
        print('trloss :', trloss[-1], ', tracc_seg:', tracc_seg[-1])
    
    ### Save results ###
    if not os.path.exists(result_path+NET_NAME+'/'):
        os.makedirs(result_path+NET_NAME+'/', exist_ok=True)
    np.savez(result_path+NET_NAME+'/tr_result.npz', loss = trloss, acc = tracc_seg)

