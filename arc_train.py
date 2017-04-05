#!/usr/bin/python

from subprocess import call
from time import strftime
import tensorflow as tf
import os.path
import os

import numpy as np

from construct_network import weightVariable, biasVariable, \
        conv2d, maxPool2x2, constructMlp, constructCnn

from config_file import writeConfigFile
from nn_config import NnConfig
from prepare_dataset import DataSet

#def writeConfigFile(configDir, dictionary, configFilename = None):

# TODO: save size of the image in the dir with the
# model

DATASET_DIR = "dataset"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
DO_USE_PREV_MODEL = False 

def trainNn(nnConfig, phInput, phOutput, session, dataSet, doSaveModel):
    cnnOut = constructCnn(phInput, nnConfig.channelsInp, \
            nnConfig.cnnLayersSize, \
            nnConfig.convWindowSize)
    cnnOutSize = np.int(cnnOut.get_shape()[1])

    (yConvCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject)
    yConvList = [yConvCurrent]

    for i in range(1, nnConfig.numObjects):
        (yConvCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject, keepProb)
        yConvList.append(yConvCurrent)

    yConv = tf.concat(yConvList, 1)

    # End of network construction

    averageAbsDelta = tf.abs(yConv - phOutput) / nnConfig.sizeOut
    absLoss = tf.reduce_sum(averageAbsDelta)

    squared_deltas = tf.square(yConv - phOutput)
    loss = tf.reduce_sum(squared_deltas)

    train_step = tf.train.AdamOptimizer(nnConfig.optimizationStep).minimize(loss)

    initOp = tf.global_variables_initializer()
    sess.run(initOp)

    # Training the network

    testBatchSize = 1000
    for i in range(nnConfig.optimizationIterationsNum):
        if i % 25 == 0:
            batchInput, batchOutput = dataSet.getTestingBatch(testBatchSize)
            absLossCurr = sess.run(absLoss, {phInput: batchInput, phOutput: batchOutput, keepProb: 1.0})
            absLossCurr = absLossCurr / testBatchSize
            print("%d: average error is %f" % (i, absLossCurr))

        batchInput, batchOutput = dataSet.getTrainingBatch(nnConfig.batchSize)
        stepCurr, lossCurr = sess.run([train_step, loss], {phInput: batchInput, phOutput: batchOutput, keepProb: 0.5})
        print("%d: loss is %f" % (i, lossCurr))

    validationSetSize = dataSet.getValidationSetSize()
    batchInput, batchOutput = dataSet.getValidationBatch(validationSetSize)
    absLossCurr = sess.run(absLoss, {phInput: batchInput, phOutput: batchOutput, keepProb: 1.0})
    absLossCurr = absLossCurr / validationSetSize
    print absLossCurr
    nnConfig.error = absLossCurr

    # Saving the network

    currentDir = os.getcwd()
    allModelsPath = currentDir + "/" + MODEL_DIR
    stringDateTime = strftime("%y%m%d_%H%M%S")
    modelDirPath = allModelsPath + "/" + stringDateTime
    call(["mkdir", "-p", modelDirPath])
    modelPath = modelDirPath + "/" + MODEL_FILENAME
    pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME

    if doSaveModel:
        saver = tf.train.Saver()
        saver.save(sess, modelPath)

    nnConfig.saveToFile(modelDirPath, stringDateTime)
    nnConfig.markCommonRecord(allModelsPath, stringDateTime)
    if os.path.exists(pathCurrent):
        call(["rm", pathCurrent])

    call(["ln", "-s", modelDirPath, pathCurrent])


dataSet = DataSet()
isSuccess = dataSet.prepareDataset(DATASET_DIR)
if not isSuccess:
    print("Error: could not load dataset. Exiting...")
    exit(1)

# creating description of network configuration
nnConfig = NnConfig()

nnConfig.heightInp, nnConfig.widthInp, nnConfig.channelsInp = dataSet.getInputDimensions()

nnConfig.sizeOut = dataSet.getOutSizeTotal()
nnConfig.sizeOutObject = dataSet.getOutSizeObject()
nnConfig.numObjects = nnConfig.sizeOut / nnConfig.sizeOutObject

# TensorFlow model description

sess = tf.InteractiveSession()

phInput = tf.placeholder(tf.float32, shape = (None, nnConfig.heightInp, \
    nnConfig.widthInp, \
    nnConfig.channelsInp))
phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

# Beginning of network construction

nnConfig.cnnLayersSize = [32, 64]
nnConfig.convWindowSize = 3

nnConfig.optimizationIterationsNum = 301
nnConfig.optimizationStep = 1e-3
nnConfig.batchSize = 100

nnConfig.mlpLayersSize = [512]
trainNn(nnConfig, phInput, phOutput, sess, dataSet, doSaveModel = True)

###

# Restoring previously saved model
#saver = tf.train.Saver()
#pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME
#if os.path.exists(pathCurrent) and DO_USE_PREV_MODEL:
#    modelFilePath = pathCurrent + "/" + MODEL_FILENAME
#    saver.restore(sess, modelFilePath)
#    dataSet.setRandomTrainingBeginning()
#else:
#    init = tf.global_variables_initializer()
#    sess.run(init)

###

