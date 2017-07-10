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
IMPROVE_COUNTER_LIMIT = 5
TESTSET_DIR = "testset"

currentDir = os.getcwd()
pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME
if not os.path.exists(pathCurrent):
    print("Error: could not find current model directory. Exiting...")
    exit(1)
nnConfig = NnConfig()
nnConfig.loadFromFile(pathCurrent)

# TensorFlow model description

sess = tf.InteractiveSession()

phInput = tf.placeholder(tf.float32, shape = (None, nnConfig.heightInp, nnConfig.widthInp, nnConfig.channelsInp))
#phOutput = tf.placeholder(tf.float32, shape = (None, nnConfig.heightOut, nnConfig.widthOut, nnConfig.objectNames))

cnnLayersSize = deepcopy(nnConfig.cnnLayersSize)
cnnLayersSize.append(nnConfig.numObjects)

convWindowSize = deepcopy(nnConfig.convWindowSize)
convWindowSize.append(convWindowSize[-1])

cnnOutFlat, cnnOutPool = constructCnn(phInput, nnConfig.channelsInp, \
    cnnLayersSize, \
    convWindowSize)

saver = tf.train.Saver()

modelFilePath = pathCurrent + "/" + MODEL_FILENAME
saver.restore(sess, modelFilePath)

testImageNames = getImageNames(TESTSET_DIR)

