#!/usr/bin/python
from tensorflow.examples.tutorials.mnist import input_data
from prepare_dataset import DataSet
from subprocess import call
from time import strftime
import tensorflow as tf
import os.path
import os

import numpy as np

from construct_network import weightVariable, biasVariable, \
        conv2d, maxPool2x2, constructMlp, constructCnn

# TODO: save size of the image in the dir with the
# model

DATASET_DIR = "dataset"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
DO_USE_PREV_MODEL = False 

def divideByTwo(number):
    if (number % 2 == 0):
        return number / 2
    else:
        return number / 2 + 1

dataSet = DataSet()
isSuccess = dataSet.prepareDataset(DATASET_DIR)
if not isSuccess:
    print("Error: could not load dataset. Exiting...")
    exit(1)

heightInp, widthInp, channelsInp = dataSet.getInputDimensions()

sizeOut = dataSet.getOutSizeTotal()
sizeOutObject = dataSet.getOutSizeObject()
numObjects = sizeOut / sizeOutObject

# TensorFlow model description

sess = tf.InteractiveSession()

x_image = tf.placeholder(tf.float32, shape = (None, heightInp, widthInp, channelsInp))
y_ = tf.placeholder(tf.float32, shape = (None, sizeOut))

cnnOut = constructCnn(x_image, channelsInp, [32, 64])
cnnOutSize = np.int(cnnOut.get_shape()[1])

mlpLayersSize = [256]
(yConvCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, mlpLayersSize, sizeOutObject)
yConvList = [yConvCurrent]

for i in range(1, numObjects):
    (yConvCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, mlpLayersSize, sizeOutObject, keepProb)
    yConvList.append(yConvCurrent)

yConv = tf.concat(yConvList, 1)

averageAbsDelta = tf.abs(yConv - y_) / sizeOut
absLoss = tf.reduce_sum(averageAbsDelta)

squared_deltas = tf.square(yConv - y_)
loss = tf.reduce_sum(squared_deltas)

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

saver = tf.train.Saver()
currentDir = os.getcwd()
pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME
if os.path.exists(pathCurrent) and DO_USE_PREV_MODEL:
    modelFilePath = pathCurrent + "/" + MODEL_FILENAME
    saver.restore(sess, modelFilePath)
    dataSet.setRandomTrainingBeginning()
else:
    init = tf.global_variables_initializer()
    sess.run(init)

trainBatchSize = 100
testBatchSize = 1000
for i in range(21):
    if i % 10 == 0:
        batchInput, batchOutput = dataSet.getTestingBatch(testBatchSize)
        absLossCurr = sess.run(absLoss, {x_image: batchInput, y_: batchOutput, keepProb: 1.0})
        absLossCurr = absLossCurr / testBatchSize
        print("%d: average error is %f" %(i, absLossCurr))

    batchInput, batchOutput = dataSet.getTrainingBatch(trainBatchSize)
    stepCurr, lossCurr = sess.run([train_step, loss], {x_image: batchInput, y_: batchOutput, keepProb: 0.5})
    print("%d: loss is %f" % (i, lossCurr))

stringDateTime = strftime("%y%m%d_%H%M%S")
modelDirPath = currentDir + "/" + MODEL_DIR + "/" + stringDateTime
call(["mkdir", "-p", modelDirPath])
modelPath = modelDirPath + "/" + MODEL_FILENAME

saver.save(sess, modelPath)

if os.path.exists(pathCurrent):
    call(["rm", pathCurrent])

call(["ln", "-s", modelDirPath, pathCurrent])
