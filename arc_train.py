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
IMPROVE_COUNTER_LIMIT = 10

def trainNn(nnConfig, phInput, phOutput, session, dataSet, doSaveModel, doCheckImprovement = None, doRestoreModel = None):
    if doCheckImprovement is None:
        doCheckImprovement = False

    if doRestoreModel is None:
        doRestoreModel = False

    cnnOut = constructCnn(phInput, nnConfig.channelsInp, \
            nnConfig.cnnLayersSize, \
            nnConfig.convWindowSize)
    cnnOutSize = np.int(cnnOut.get_shape()[1])

    (outCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject)
    outList = [outCurrent]

    for i in range(1, nnConfig.numObjects):
        (outCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject, keepProb)
        outList.append(outCurrent)

    out = tf.concat(outList, 1)

    isPresentReal = tf.slice(phOutput, [0, 0], [-1, 1])
    isPresentPredicted = tf.slice(out, [0, 0], [-1, 1])
    isPresentSqDelta = tf.square(isPresentReal - isPresentPredicted)
    isPresentAbsDelta = tf.abs(isPresentReal - isPresentPredicted)

    xReal = tf.slice(phOutput, [0, 1], [-1, 1])
    xPredicted = tf.slice(out, [0, 1], [-1, 1])
    xSqDelta = isPresentReal * tf.square(xReal - xPredicted)
    xAbsDelta = isPresentReal * tf.abs(xReal - xPredicted)

    yReal = tf.slice(phOutput, [0, 2], [-1, 1])
    yPredicted = tf.slice(out, [0, 2], [-1, 1])
    ySqDelta = isPresentReal * tf.square(yReal - yPredicted)
    yAbsDelta = isPresentReal * tf.abs(yReal - yPredicted)

    errorSum = isPresentSqDelta + xSqDelta + ySqDelta
    errorSumAbs = xAbsDelta + yAbsDelta

    for i in range(1, nnConfig.numObjects):
        isPresentReal = tf.slice(phOutput, [0, i * 3], [-1, 1])
        isPresentPredicted = tf.slice(out, [0, i * 3], [-1, 1])
        isPresentSqDelta = tf.square(isPresentReal - isPresentPredicted)
        isPresentAbsDelta = tf.abs(isPresentReal - isPresentPredicted)

        xReal = tf.slice(phOutput, [0, i * 3 + 1], [-1, 1])
        xPredicted = tf.slice(out, [0, i * 3 + 1], [-1, 1])
        xSqDelta = isPresentReal * tf.square(xReal - xPredicted)
        xAbsDelta = isPresentReal * tf.abs(xReal - xPredicted)

        yReal = tf.slice(phOutput, [0, i * 3 + 2], [-1, 1])
        yPredicted = tf.slice(out, [0, i * 3 + 2], [-1, 1])
        ySqDelta = isPresentReal * tf.square(yReal - yPredicted)
        yAbsDelta = isPresentReal * tf.abs(yReal - yPredicted)

        errorSum += isPresentSqDelta + xSqDelta + ySqDelta
        errorSumAbs += xAbsDelta + yAbsDelta

    # End of network construction

    # (ispresent-_ispresent)^2 + (_ispresent*(y-_y)^2+_ispresent*(x-_x)^2)

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
        sess.run(initOp)

    # Training the network

    testBatchSize = 1000
    iterationCounter = 0
    noImproveCounter = 0
    minAbsLoss = float("inf")
    while True:
        if (not doCheckImprovement) and (iterationCounter >= nnConfig.optimizationIterationsNum):
            break

        if iterationCounter % 10 == 0:
            batchInput, batchOutput = dataSet.getTestingBatch(testBatchSize)
            absLossCurr = sess.run(absLoss, {phInput: batchInput, phOutput: batchOutput, keepProb: 1.0})
            absLossCurr = absLossCurr / testBatchSize
            print("%d: average error is %f" % (iterationCounter, absLossCurr))
            if absLossCurr < minAbsLoss:
                noImproveCounter = 0
                minAbsLoss = absLossCurr
            else:
                noImproveCounter += 1
                print("no improvement in error for %d iterations" % noImproveCounter)
                if noImproveCounter > IMPROVE_COUNTER_LIMIT:
                    break

        batchInput, batchOutput = dataSet.getTrainingBatch(nnConfig.batchSize)
        stepCurr, lossCurr = sess.run([train_step, loss], {phInput: batchInput, phOutput: batchOutput, keepProb: 0.5})
        print("%d: loss is %f" % (iterationCounter, lossCurr))

        iterationCounter += 1

    validationSetSize = dataSet.getValidationSetSize()
    batchInput, batchOutput = dataSet.getValidationBatch(validationSetSize)
    absLossCurr = sess.run(absLoss, {phInput: batchInput, phOutput: batchOutput, keepProb: 1.0})
    absLossCurr = absLossCurr / validationSetSize
    print absLossCurr
    nnConfig.error = absLossCurr

    # Saving the network

    allModelsPath = currentDir + "/" + MODEL_DIR
    stringDateTime = strftime("%y%m%d_%H%M%S")
    modelDirPath = allModelsPath + "/" + stringDateTime
    call(["mkdir", "-p", modelDirPath])
    modelPath = modelDirPath + "/" + MODEL_FILENAME

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

nnConfig.cnnLayersSize = [8, 16, 32, 48, 64]
nnConfig.convWindowSize = 3

nnConfig.optimizationIterationsNum = 3001

nnConfig.optimizationStep = 1e-5
nnConfig.batchSize = 100

nnConfig.mlpLayersSize = [128]
trainNn(nnConfig, phInput, phOutput, sess, dataSet, doSaveModel = True, doCheckImprovement = True, doRestoreModel = True)
