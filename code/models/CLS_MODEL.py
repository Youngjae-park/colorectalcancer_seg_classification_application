#-*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:31:16 2020

@author: youngjae
"""

import tensorflow as tf
from classification_models.tfkeras import Classifiers


class Network:
    def __init__(self, model_, lr_):
        self.model_name = model_
        self.model, _ = Classifiers.get(model_)
        self.lr = lr_
    
        self.wpath = '/home/dhodwo/venv/weights/'+self.model_name+'/CLS/'
    
    def placehold(self, x_shape_, y_shape_, channel=3, cls=3):
        self.x_shape = x_shape_
        self.y_shape = y_shape_
        self.channel = channel
        self.cls = cls
        self.datx = tf.placeholder(tf.float32, (None, self.x_shape, self.x_shape, channel), name='datx')
        self.daty_cls = tf.placeholder(tf.float32, (None, channel), name='daty_cls')
    
    def classification_model(self):
        self.model = self.model(include_top=False, input_shape=(self.x_shape, self.x_shape, self.channel), classes=self.cls, weights='imagenet')
        
        logits = self.model(self.datx)

        with tf.variable_scope('custom'):
            logits = tf.contrib.layers.flatten(logits)
            logits = tf.expand_dims(logits, axis=2)
            logits = tf.layers.conv1d(logits, 3, logits.shape[1].value)
            self.flogits = tf.squeeze(logits, 1)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.flogits, labels=self.daty_cls))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.cost)
        self.predict = tf.argmax(self.flogits, 1, name='predict')
        self.correct_prediction = tf.equal(tf.argmax(self.flogits, 1), tf.argmax(self.daty_cls, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
    def session_init(self, init='global_init'):
        self.sess = tf.InteractiveSession()
        if init=='global_init':
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, pl_in, pl_out):
        self.sess.run(self.train_op, {self.datx:pl_in, self.daty_cls:pl_out})
    
    def result_info(self, pl_in, pl_out):
        l, pl, a = self.sess.run([self.cost, self.predict, self.accuracy], {self.datx:pl_in, self.daty_cls:pl_out})
        return l, pl, a

    def saver_init(self, epochs):
        self.saver = tf.train.Saver(max_to_keep=epochs)

    def save(self, ep, path):
        self.saver.save(self.sess, path, global_step = ep+1)

    def get_weights(self, ep):
        fwpath = self.wpath + 'epoch-'+str(ep)
        self.saver.restore(self.sess, fwpath)
