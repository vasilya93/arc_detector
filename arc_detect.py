#!/usr/bin/python
from tensorflow.examples.tutorials.mnist import input_data
from image_check import getImageNames
from subprocess import call
from time import strftime
import time
import tensorflow as tf
import os.path
import os
import cv2
import math

import numpy as np

from construct_network import weightVariable, biasVariable, \
        conv2d, maxPool2x2, constructMlp, constructCnn

from nn_config import NnConfig

# TODO: cosider possibility that the object searched is not in the
# image, so there should be an output indicating whether the object
# is present in the frame

# TODO: transport the code into c++

# TODO: consider possibility of variations in the object (different
# colors of toothbrush packages, different patterns on a notebook)


DATASET_DIR = "dataset"
TESTSET_DIR = "testset"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
DO_CROP = True

top = 295
left = 721
right = 1148
bottom = 513

currentDir = os.getcwd()
pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME
if not os.path.exists(pathCurrent):
    print("Error: could not find current model directory. Exiting...")
    exit(1)
nnConfig = NnConfig()
nnConfig.loadFromFile(pathCurrent)

# TensorFlow model description

sess = tf.InteractiveSession()

x_image = tf.placeholder(tf.float32, shape = (None, nnConfig.heightInp, nnConfig.widthInp, nnConfig.channelsInp))
y_ = tf.placeholder(tf.float32, shape = (None, nnConfig.sizeOut))

# Beginning of network construction

cnnOut = constructCnn(x_image, nnConfig.channelsInp, nnConfig.cnnLayersSize, nnConfig.convWindowSize)
cnnOutSize = np.int(cnnOut.get_shape()[1])

(yConvCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject)
yConvList = [yConvCurrent]

for i in range(1, nnConfig.numObjects):
    (yConvCurrent, keepProb) = constructMlp(cnnOut, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject, keepProb)
    yConvList.append(yConvCurrent)

yConv = tf.concat(yConvList, 1)

# End of network construction

saver = tf.train.Saver()

modelFilePath = pathCurrent + "/" + MODEL_FILENAME
saver.restore(sess, modelFilePath)

testImageNames = getImageNames(TESTSET_DIR)

inputData = np.zeros((1, nnConfig.heightInp, nnConfig.widthInp, nnConfig.channelsInp), np.float32)
for imageName in testImageNames:
    imagePath = TESTSET_DIR + "/" + imageName
    image = cv2.imread(imagePath)

    if DO_CROP:
        image = image[top:bottom, left:right]

    currentHeight, currentWidth, currentChannels = image.shape
    currentHeightHalf = currentHeight / 2
    currentWidthHalf = currentWidth / 2
    imageResized = np.float32(cv2.resize(image, (nnConfig.widthInp, nnConfig.heightInp)))
    imageResized /= 255.0
    inputData[0, :, :, :] = imageResized[:, :, :]

    start_time = time.time()
    yCurr = sess.run(yConv, {x_image: inputData, keepProb: 1.0})

    print("object 1: %f \t %f \t %f" % (yCurr[0][0], yCurr[0][1], yCurr[0][2]))
    
    objectX1 = np.int(yCurr[0][1] * currentWidthHalf + currentWidthHalf)
    objectY1 = np.int(yCurr[0][2] * currentHeightHalf + currentHeightHalf)
    if yCurr[0][0] > 0.5:
        cv2.circle(image, (objectX1, objectY1), 10, (255, 0, 255), 2)

    if sizeOut >= 6:
        print("object 2: %f \t %f \t %f" % (yCurr[0][3], yCurr[0][4], yCurr[0][5]))
        objectX1 = np.int(yCurr[0][4] * currentWidthHalf + currentWidthHalf)
        objectY1 = np.int(yCurr[0][5] * currentHeightHalf + currentHeightHalf)
        if yCurr[0][3] > 0.5:
            cv2.circle(image, (objectX1, objectY1), 10, (255, 0, 0), 2)

    if sizeOut >= 9:
        print("object 3: %f \t %f \t %f" % (yCurr[0][6], yCurr[0][7], yCurr[0][8]))
        objectX1 = np.int(yCurr[0][7] * currentWidthHalf + currentWidthHalf)
        objectY1 = np.int(yCurr[0][8] * currentHeightHalf + currentHeightHalf)
        if yCurr[0][6] > 0.5:
            cv2.circle(image, (objectX1, objectY1), 10, (0, 0, 255), 2)

    if sizeOut >= 12:
        print("object 4: %f \t %f \t %f" % (yCurr[0][9], yCurr[0][10], yCurr[0][11]))
        objectX1 = np.int(yCurr[0][10] * currentWidthHalf + currentWidthHalf)
        objectY1 = np.int(yCurr[0][11] * currentHeightHalf + currentHeightHalf)
        if yCurr[0][9] > 0.5:
            cv2.circle(image, (objectX1, objectY1), 10, (0, 255, 0), 2)

    print ""

    cv2.imshow("image", image)
    cv2.waitKey(0)
