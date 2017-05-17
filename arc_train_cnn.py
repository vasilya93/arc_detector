#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from copy import deepcopy

from nn_config import NnConfig
from prepare_dataset import DataSet
from construct_network import constructCnn

DATASET_DIR = "dataset"

def constructCnnMarkup(phInput, nnConfig):
    cnnLayersSize = deepcopy(nnConfig.cnnLayersSize)
    cnnLayersSize.append(nnConfig.numObjects)

    convWindowSize = deepcopy(nnConfig.convWindowSize)
    convWindowSize.append(convWindowSize[-1])

    cnnOutFlat, cnnOutPool, cnnWeights = constructCnn(phInput, nnConfig.channelsInp, \
            cnnLayersSize, \
            convWindowSize)
    cnnOutSize = cnnOutPool.get_shape()
 
    phOutput = tf.placeholder(tf.float32, cnnOutSize)

    #height = cnnOutSize[1]
    #width = cnnOutSize[2]
    #numObjects = cnnOutSize[3]
 
    sqDelta = tf.square(cnnOutPool - phOutput)
    loss = tf.reduce_sum(sqDelta)
    #sumPixel = tf.reduce_sum(cnnOutPool)# / numObjects
    #sumPixelRef = tf.reduce_sum(phOutput)#

    return loss, phOutput

def trainNn(nnConfig, \
        phInput, \
        session, \
        dataSet, \
        doSaveModel, \
        doCheckImprovement = None, \
        doRestoreModel = None):

    if doCheckImprovement is None:
        doCheckImprovement = False

    if doRestoreModel is None:
        doRestoreModel = False

    #phOutput = tf.placeholder(tf.float32, \
    #    shape = (None, 13, 19, 1))

    constructCnnMarkup(phInput, nnConfig)
    loss, phOutput = constructCnnMarkup(phInput, phOutput, nnConfig)
    #height = phOutput.get_shape()[1]
    #width = phOutput.get_shape()[2]
    #print "height %d; width %d" % (height, width)
    train_step = tf.train.AdamOptimizer(nnConfig.optimizationStep).minimize(loss)

    initOp = tf.global_variables_initializer()
    session.run(initOp)

    iterationCounter = 0
    noImproveCounter = 0
    minAbsLoss = float("inf")
    while True:
        if iterationCounter >= 100:
            break

        batchInput, batchOutput = dataSet.getTrainingBatchMarkup(nnConfig.batchSize, height, width)
        stepCurr, lossCurr, sumPixCurr, sumPixelRefCurr = session.run([train_step, loss, sumPixel, sumPixelRef], \
                {phInput: batchInput, phOutput: batchOutput})
        pixCurrMean = np.float(sumPixCurr) / nnConfig.batchSize
        pixCurrRefMean = np.float(sumPixelRefCurr) / nnConfig.batchSize
        print("%d: loss is %f, pixCurrMean is %f, pixCurrRefMean is %f" % \
                (iterationCounter, lossCurr, pixCurrMean, pixCurrRefMean))

        cnnWeightCurrent = session.run(cnnWeights[4])
        print("cnn weight is %s\r\n" % str(cnnWeightCurrent))

        iterationCounter += 1

    return

def trainSingle(nnConfig, dataSet):
    sess = tf.InteractiveSession()

    phInput = tf.placeholder(tf.float32, \
            shape = (None, \
            nnConfig.heightInp, \
            nnConfig.widthInp, \
            nnConfig.channelsInp))
    phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

    trainNn(nnConfig, phInput, phOutput, sess, dataSet, doSaveModel = True, \
            doCheckImprovement = True, doRestoreModel = True)

dataSet = DataSet()
isSuccess = dataSet.prepareDataset(DATASET_DIR)
if not isSuccess:
    print("Error: could not load dataset. Exiting...")
    exit(1)

# creating description of network configuration
nnConfig = NnConfig()

nnConfig.objectNames = dataSet.getObjectNames()

nnConfig.heightInp, \
nnConfig.widthInp, \
nnConfig.channelsInp = dataSet.getInputDimensions()

nnConfig.sizeOut = dataSet.getOutSizeTotal()
nnConfig.sizeOutObject = dataSet.getOutSizeObject()
nnConfig.numObjects = nnConfig.sizeOut / nnConfig.sizeOutObject

# beginning of network construction
nnConfig.optimizationIterationsNum = 3001
nnConfig.optimizationStep = 1e-3
nnConfig.batchSize = 1000
nnConfig.mlpLayersSize = [256]
nnConfig.cnnLayersSize = [8, 16, 32, 48, 64]
nnConfig.convWindowSize = [3, 3, 3, 3, 3]

trainSingle(nnConfig, dataSet)
