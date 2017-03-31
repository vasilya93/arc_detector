#!/usr/bin/python
import tensorflow as tf

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def constructMlp(mlpInput, inputSize, layerSizes, outSize, keepProb = None):
    if keepProb is None:
        keepProb = tf.placeholder(tf.float32)

    layersNum = len(layerSizes)

    weightLayer = weightVariable([inputSize, layerSizes[0]])
    biasLayer = biasVariable([layerSizes[0]])
    layerOut = tf.nn.relu(tf.matmul(mlpInput, weightLayer) + biasLayer)
    layerOutDrop = tf.nn.dropout(layerOut, keepProb)

    for i in range(1, len(layerSizes)):
        weightLayer = weightVariable([layerSizes[i - 1], layerSizes[i]])
        biasLayer = biasVariable([layerSizes[i]])
        layerOut = tf.nn.relu(tf.matmul(layerOutDrop, weightLayer) + biasLayer)
        layerOutDrop = tf.nn.dropout(layerOut, keepProb)

    weightOut = weightVariable([layerSizes[layersNum - 1], outSize])
    biasOut = biasVariable([outSize])

    out = tf.matmul(layerOutDrop, weightOut) + biasOut
    return (out, keepProb)

