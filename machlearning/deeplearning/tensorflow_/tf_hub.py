# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:43:47 2022

@author: dell
"""
import tensorflow as tf
import tensorflow_hub as hub

# 指定模型的 URL
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130/classification/4"

# 创建一个 TensorFlow Hub 模块
module = hub.Module(module_url)

# 创建输入占位符
input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))

# 调用 TensorFlow Hub 模块进行推理
logits = module(input_tensor)

# 加载预训练权重（如果有的话）
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# 创建一个 TensorFlow Hub 模块（带有可训练参数）
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130/classification/4"
module = hub.Module(module_url, trainable=True)

module1 = hub.load(module_url)
# 创建输入占位符
input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))

# 调用 TensorFlow Hub 模块进行推理
logits = module(input_tensor)

# 加载预训练权重
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# 构造输入数据
your_input_data = ...

# 使用模型进行推理
predictions = sess.run(logits, feed_dict={input_tensor: your_input_data})

# 处理模型的输出（例如，获取分类结果）
print(predictions)


