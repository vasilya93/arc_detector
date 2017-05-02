#!/usr/bin/python
import tensorflow as tf
import numpy as np

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Characteristics of convolutional layer:
# - window size (5x5)
# - number of output channels
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Characteristics of pooling layer:
# - window size
# - stride
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

    for i in range(1, layersNum):
        weightLayer = weightVariable([layerSizes[i - 1], layerSizes[i]])
        biasLayer = biasVariable([layerSizes[i]])
        layerOut = tf.nn.relu(tf.matmul(layerOutDrop, weightLayer) + biasLayer)
        layerOutDrop = tf.nn.dropout(layerOut, keepProb)

    weightOut = weightVariable([layerSizes[layersNum - 1], outSize])
    biasOut = biasVariable([outSize])

    out = tf.matmul(layerOutDrop, weightOut) + biasOut
    return (out, keepProb)


def constructCnn(nssInput, channelsInp, layerSizes, convWindowSize = None):
    if convWindowSize is None or len(layerSizes) != len(convWindowSize):
        convWindowSize = [5] * len(layerSizes)

    layersNum = len(layerSizes)

    weightConv = weightVariable([convWindowSize[0], convWindowSize[0], channelsInp, layerSizes[0]])
    biasConv = biasVariable([layerSizes[0]])
    outConv = tf.nn.relu(conv2d(nssInput, weightConv) + biasConv)
    outPool = maxPool2x2(outConv)

    for i in range(1, layersNum):
        weightConv = weightVariable([convWindowSize[i], convWindowSize[i], layerSizes[i - 1], layerSizes[i]])
        biasConv = biasVariable([layerSizes[i]])
        outConv = tf.nn.relu(conv2d(outPool, weightConv) + biasConv)
        outPool = maxPool2x2(outConv)

    [outHeight, outWidth, outChannels] = outPool.get_shape()[1:4]
    convOutputSize = np.int(outHeight * outWidth * outChannels)
    outFlat = tf.reshape(outPool, [-1, convOutputSize])

    return (outFlat, outPool)
