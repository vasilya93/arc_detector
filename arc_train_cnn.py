#!/usr/bin/env python

import os

import tensorflow as tf
import numpy as np

from copy import deepcopy
from time import strftime
from subprocess import call

from nn_config import NnConfig
from prepare_dataset import DataSet
from construct_network import constructCnn

DATASET_DIR = "dataset"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
IMPROVE_COUNTER_LIMIT = 10

def constructCnnMarkup(phInput, nnConfig):
    cnnLayersSize = deepcopy(nnConfig.cnnLayersSize)
    cnnLayersSize.append(nnConfig.numObjects)

    convWindowSize = deepcopy(nnConfig.convWindowSize)
    convWindowSize.append(convWindowSize[-1])

    cnnOutFlat, cnnOutPool = constructCnn(phInput, nnConfig.channelsInp, \
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

    constructCnnMarkup(phInput, nnConfig)
    loss, phOutput = constructCnnMarkup(phInput, nnConfig)
    nnConfig.heightOut = phOutput.get_shape()[1]
    nnConfig.widthOut = phOutput.get_shape()[2]
    print "height %d; width %d" % (nnConfig.heightOut, nnConfig.widthOut)
    train_step = tf.train.AdamOptimizer(nnConfig.optimizationStep).minimize(loss)

    currentDir = os.getcwd()
    pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME

    if doRestoreModel:
        print("model is restored")
        saver = tf.train.Saver()
        modelFilePath = pathCurrent + "/" + MODEL_FILENAME
        saver.restore(session, modelFilePath)
        dataSet.setRandomTrainingBeginning()
    else:
        initOp = tf.global_variables_initializer()
        session.run(initOp)

    iterationCounter = 0
    noImproveCounter = 0
    minLoss = float("inf")
    while True:
        batchInput, batchOutput = dataSet.getTrainingBatchMarkup(nnConfig.batchSize, nnConfig.heightOut, nnConfig.widthOut)
        stepCurr, lossCurr = session.run([train_step, loss], \
                {phInput: batchInput, phOutput: batchOutput})
        print("%d: loss is %f" % (iterationCounter, lossCurr))

        #cnnWeightCurrent = session.run(cnnWeights[4])
        #print("cnn weight is %s\r\n" % str(cnnWeightCurrent))
        if iterationCounter % 10 == 0:
            batchInput, batchOutput = dataSet.getTestingBatchMarkup(100, nnConfig.heightOut, nnConfig.widthOut)
            lossCurr = session.run(loss, {phInput: batchInput, \
                    phOutput: batchOutput})
            print("%d: error is %f" % (iterationCounter, lossCurr))

            if lossCurr < minLoss:
                noImproveCounter = 0
                minLoss = lossCurr
            else:
                noImproveCounter += 1
                print("no improvement in error for %d iterations" % \
                        noImproveCounter)
                if noImproveCounter > IMPROVE_COUNTER_LIMIT:
                    break

        iterationCounter += 1

    
    allModelsPath = currentDir + "/" + MODEL_DIR
    stringDateTime = strftime("%y%m%d_%H%M%S")
    modelDirPath = allModelsPath + "/" + stringDateTime
    call(["mkdir", "-p", modelDirPath])
    modelPath = modelDirPath + "/" + MODEL_FILENAME

    if doSaveModel:
        saver = tf.train.Saver()
        saver.save(session, modelPath)
    print("Info: model of the network was saved")

    nnConfig.saveToFile(modelDirPath, stringDateTime)
    print("Info: configuration of the network was saved")
    if os.path.exists(pathCurrent):
        call(["rm", pathCurrent])

    call(["ln", "-s", modelDirPath, pathCurrent])

    return

def trainSingle(nnConfig, dataSet):
    sess = tf.InteractiveSession()

    phInput = tf.placeholder(tf.float32, \
            shape = (None, \
            nnConfig.heightInp, \
            nnConfig.widthInp, \
            nnConfig.channelsInp))
    phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

    trainNn(nnConfig, phInput, sess, dataSet, doSaveModel = True, \
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
nnConfig.batchSize = 50
nnConfig.mlpLayersSize = [256]
nnConfig.cnnLayersSize = [8]
nnConfig.convWindowSize = [3]

trainSingle(nnConfig, dataSet)
