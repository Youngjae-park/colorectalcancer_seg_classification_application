#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020. 10. 26. (MON) 21:39:12 KST

@author: youngjae
"""

import tensorflow as tf
import segmentation_models as SMT

def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")
    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true*pred_prob) + ((1-y_true)*(1-pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=tf.keras.backend.floatx())
        alpha_factor = y_true*alpha + (1-y_true)*(1-alpha)
    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.keras.backend.floatx())
        modulating_factor = tf.pow((1.0-p_t), gamma)

    out = tf.reduce_sum(alpha_factor*modulating_factor*ce, axis=-1)

    # compute the final loss and return
    return tf.reduce_mean(out)

class Network:
    def __init__(self, model_, lr_):
        self.model_name = model_
        self.lr = lr_
        self.wpath = '/home/dhodwo/venv/weights/'+self.model_name+'/SEG/'

    def placehold_getNN(self, x_shape_, y_shape_, channel=3, cls=1):
        self.x_shape = x_shape_
        self.y_shape = y_shape_
        self.cls = cls
        
        ### get model
        self.model = SMT.Unet(self.model_name, input_shape=(x_shape_, y_shape_, channel), classes=cls, activation='sigmoid', encoder_weights='imagenet')
        ### input and output placeholder
        self.datx = tf.placeholder(tf.float32, (None, self.x_shape, self.x_shape, channel), name = 'datx')
        self.daty_seg = tf.placeholder(tf.float32, (None, self.y_shape, self.y_shape, cls), name = 'daty_seg')
        
        ### connect input placeholder to model
        self.logits = self.model(self.datx)
        
        ### Cost, Optimizer, Train_op
        self.cost = sigmoid_focal_crossentropy(self.daty_seg, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train_op = self.optimizer.minimize(self.cost)
        
    def session_init(self, init='global_init'):
        self.sess = tf.InteractiveSession()
        if init=='global_init':
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, pl_in, pl_out):
        self.sess.run(self.train_op, {self.datx:pl_in, self.daty_seg:pl_out})

    def result_info(self, pl_in, pl_out):
        loss, pm = self.sess.run([self.cost, self.logits], {self.datx:pl_in, self.daty_seg:pl_out})
        pm = pm.round()
        b_ym, pm = pl_out.reshape((len(pl_out), -1)), pm.reshape((len(pm), -1))
        return loss, b_ym, pm

    def saver_init(self, epochs):
        self.saver = tf.train.Saver(max_to_keep=epochs)

    def save(self, path=self.wpath, ep):
        self.saver.save(self.sess, path, global_step = ep+1)

    def get_weights(self, ep):
        fwpath = self.wpath + 'epoch-'+str(ep)
        self.saver.restore(self.sess, fwpath)
