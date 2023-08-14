# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:43:47 2022

@author: dell
"""

import tensorflow as tf

a = tf.constant([1,2,3,4],dtype=tf.int32)
b = tf.random.normal([3,4],5,5)
c = tf.cast(b,tf.int32)
print(c)
print(tf.reduce_sum(c,axis=1))
# print(b)