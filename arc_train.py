#!/usr/bin/python

from subprocess import call
from time import strftime
from time import sleep
import tensorflow as tf
import os.path
import os

import numpy as np

from construct_network import weightVariable, biasVariable, \
        conv2d, maxPool2x2, constructMlp, constructCnn
from construct_network import constructCnnMlpPresenceIndication

from config_file import writeConfigFile
from nn_config import NnConfig
from prepare_dataset import DataSet

import threading

TRAINING_MODE_TRY_ARCHITECTURES = 1
TRAINING_MODE_DEEP = 2
TRAINING_MODE_SINGLE = 3

DATASET_DIR = "dataset/images"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
DO_USE_PREV_MODEL = False 
IMPROVE_COUNTER_LIMIT = 10

trainingMode = TRAINING_MODE_SINGLE

doesUserAskQuit = False

def threadKeyReading():
    global doesUserAskQuit
    lock = threading.Lock()
    while True:
        with lock:
            userInput = raw_input()
            if userInput == "quit":
                doesUserAskQuit = True

def trainNn(nnConfig, \
        phInput, \
        phOutput, \
        session, \
        dataSet, \
        doSaveModel, \
        doCheckImprovement = None, \
        doRestoreModel = None):

    if doCheckImprovement is None:
        doCheckImprovement = False

    if doRestoreModel is None:
        doRestoreModel = False

    # Constructing network with three outputs for each object: x, y, is_present
    errorSum, errorSumAbs, keepProb = constructCnnMlpPresenceIndication(phInput, phOutput, nnConfig)

    # Restoring previous network
    currentDir = os.getcwd()
    pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME

    #averageAbsDelta = tf.abs(yConv - phOutput) / nnConfig.sizeOut
    averageErrorAbs = errorSumAbs / nnConfig.numObjects / 2
    absLoss = tf.reduce_sum(averageErrorAbs)

    #squared_deltas = tf.square(yConv - phOutput)
    loss = tf.reduce_sum(errorSum)

    train_step = tf.train.AdamOptimizer(nnConfig.optimizationStep).minimize(loss)

    if doRestoreModel:
        print("model is restored")
        saver = tf.train.Saver()
        modelFilePath = pathCurrent + "/" + MODEL_FILENAME
        saver.restore(session, modelFilePath)
        dataSet.setRandomTrainingBeginning()
    else:
        initOp = tf.global_variables_initializer()
        session.run(initOp)

    # Training the network
    testBatchSize = 1000
    iterationCounter = 0
    noImproveCounter = 0
    minAbsLoss = float("inf")
    while True:
        if ((not doCheckImprovement) and \
                (iterationCounter >= nnConfig.optimizationIterationsNum)):
            break

        if doesUserAskQuit:
            print("Info: exiting the training loop on user request")
            break

        if iterationCounter % 10 == 0:
            batchInput, batchOutput = dataSet.getTestingBatch(testBatchSize)
            absLossCurr = session.run(absLoss, {phInput: batchInput, \
                    phOutput: batchOutput, keepProb: 1.0})
            absLossCurr = absLossCurr / testBatchSize
            print("%d: average error is %f" % (iterationCounter, absLossCurr))
            if absLossCurr < minAbsLoss:
                noImproveCounter = 0
                minAbsLoss = absLossCurr
            else:
                noImproveCounter += 1
                print("no improvement in error for %d iterations" % \
                        noImproveCounter)
                if noImproveCounter > IMPROVE_COUNTER_LIMIT:
                    break

        batchInput, batchOutput = dataSet.getTrainingBatch(nnConfig.batchSize)
        stepCurr, lossCurr = session.run([train_step, loss], \
                {phInput: batchInput, phOutput: batchOutput, keepProb: 0.5})
        print("%d: loss is %f" % (iterationCounter, lossCurr))

        iterationCounter += 1

    validationSetSize = dataSet.getValidationSetSize()
    batchInput, batchOutput = dataSet.getValidationBatch(validationSetSize)
    absLossCurr = session.run(absLoss, {phInput: batchInput, \
            phOutput: batchOutput, keepProb: 1.0})
    absLossCurr = absLossCurr / validationSetSize
    print("Info: average error for validation network is %f" % absLossCurr)
    nnConfig.error = absLossCurr

    # Saving the network
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
    nnConfig.markCommonRecord(allModelsPath, stringDateTime)
    if os.path.exists(pathCurrent):
        call(["rm", pathCurrent])

    call(["ln", "-s", modelDirPath, pathCurrent])

def tryArchitectures(nnConfig, dataSet):
    phInput = tf.placeholder(tf.float32, shape = (None, nnConfig.heightInp, \
        nnConfig.widthInp, \
        nnConfig.channelsInp))
    phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

    cnnLayerSizes = [[8, 16, 32, 64], [8, 16, 32, 48, 64], [8, 16, 32, 48, 64], \
            [8, 16, 32, 64, 128]]
    convWindowSizes = [[3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 5, 5], [3, 3, 3, 3, 3]]
    layerSizes = [128, 256, 512, 768]
    for layerSize in layerSizes:
        nnConfig.mlpLayersSize = [layerSize]
        for j in range(len(convWindowSizes)):
            nnConfig.cnnLayersSize = cnnLayerSizes[j]
            nnConfig.convWindowSize = convWindowSizes[j]

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            phInput = tf.placeholder(tf.float32, shape = (None, nnConfig.heightInp, \
                nnConfig.widthInp, \
                nnConfig.channelsInp))
            phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

            trainNn(nnConfig, phInput, phOutput, sess, dataSet, \
            doSaveModel = False, doCheckImprovement = True, doRestoreModel = False)

def trainDeep(nnConfig, dataSet):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    batchSizes = [100, 300, 600, 1000]
    optimizationSteps = [[1e-3], [1e-3], [1e-3], [1e-3]]
    for i in range(len(batchSizes)):
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        phInput = tf.placeholder(tf.float32, \
                shape = (None, \
                nnConfig.heightInp, \
                nnConfig.widthInp, \
                nnConfig.channelsInp))
        phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

        nnConfig.batchSize = batchSizes[i]
        if i == 0:
            trainNn(nnConfig, phInput, phOutput, sess, dataSet, doSaveModel = True, \
                    doCheckImprovement = True, doRestoreModel = False)
        else:
            for optimizationStep in optimizationSteps[i]:
                nnConfig.optimizationStep = optimizationStep
                trainNn(nnConfig, phInput, phOutput, sess, dataSet, doSaveModel = True, \
                        doCheckImprovement = True, doRestoreModel = True)
                if doesUserAskQuit == True:
                    break

        if doesUserAskQuit == True:
            print("Info: exiting deep training of the network")
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            break

        sleep(10)

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

# Beginning of network construction
nnConfig.optimizationIterationsNum = 3001
nnConfig.optimizationStep = 1e-3
nnConfig.batchSize = 600
nnConfig.mlpLayersSize = [256]
nnConfig.cnnLayersSize = [8, 16, 32, 48, 64]
nnConfig.convWindowSize = [3, 3, 3, 3, 3]

threading.Thread(target = threadKeyReading).start()

if trainingMode == TRAINING_MODE_TRY_ARCHITECTURES:
    tryArchitectures(nnConfig, dataSet)
elif trainingMode == TRAINING_MODE_DEEP:
    trainDeep(nnConfig, dataSet)
elif trainingMode == TRAINING_MODE_SINGLE:
    trainSingle(nnConfig, dataSet)

print("Info: exiting the program")
