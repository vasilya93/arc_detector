#!/usr/bin/python
from image_check import getImageNames
from subprocess import call
from time import strftime
from copy import deepcopy
from random import randint

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
from visualize_cnn_output import visualizeCnnOutput
from divide_image import divide_image

# TODO: consider possibility of variations in the object (different
# colors of toothbrush packages, different patterns on a notebook)

DATASET_DIR = "dataset"
TESTSET_DIR = "testset"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
DO_CROP = False
DO_DIVIDE = True

top = 441
left = 376
right = 979
bottom = 844

def decipher_network_output(nn_output, nn_config, image):
    yTextCurrent = 30
    for i in range(nnConfig.numObjects):
        probability = yCurr[0][i * nnConfig.sizeOutObject + 0]
        if nnConfig.sizeOutObject >= 3:
            xRel = yCurr[0][i * nnConfig.sizeOutObject + 1]
            yRel = yCurr[0][i * nnConfig.sizeOutObject + 2]
            objectX1 = np.int(xRel * currentWidthHalf + currentWidthHalf)
            objectY1 = np.int(yRel * currentHeightHalf + currentHeightHalf)
        if probability > 0.5:
            print("%s:\t\t %f" % (nnConfig.objectNames[i], probability))
            if probability > 1.0:
                probability = 1.0
            if nnConfig.sizeOutObject >= 3:
                markerColor = (randint(0, 255), randint(0, 255), randint(0, 255))
                imageText = "%s" % (nnConfig.objectNames[i])
                cv2.putText(image, imageText, (10, yTextCurrent), cv2.FONT_HERSHEY_TRIPLEX, .7, markerColor)
                yTextCurrent += 30
                cv2.circle(image, (objectX1, objectY1), 10, markerColor, 2)

    print "\r\n"

def network_output_to_object_list(nn_output, nn_config):
    object_list = []
    for i in range(nnConfig.numObjects):
        probability = yCurr[0][i * nnConfig.sizeOutObject + 0]
        if probability > 0.5:
            object_list.append(nnConfig.objectNames[i])
    return object_list


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

cnnOutFlat, cnnOut = constructCnn(x_image, nnConfig.channelsInp, nnConfig.cnnLayersSize, nnConfig.convWindowSize)
cnnOutSize = np.int(cnnOutFlat.get_shape()[1])

(yConvCurrent, keepProb) = constructMlp(cnnOutFlat, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject)
yConvList = [yConvCurrent]

for i in range(1, nnConfig.numObjects):
    (yConvCurrent, keepProb) = constructMlp(cnnOutFlat, cnnOutSize, nnConfig.mlpLayersSize, nnConfig.sizeOutObject, keepProb)
    yConvList.append(yConvCurrent)

yConv = tf.concat(yConvList, 1)

# End of network construction

saver = tf.train.Saver()

modelFilePath = pathCurrent + "/" + MODEL_FILENAME
saver.restore(sess, modelFilePath)

testImageNames = getImageNames(TESTSET_DIR)

counter = 0
inputData = np.zeros((1, nnConfig.heightInp, nnConfig.widthInp, nnConfig.channelsInp), np.float32)
for imageName in testImageNames:
    imagePath = TESTSET_DIR + "/" + imageName
    image = cv2.imread(imagePath)

    if DO_CROP:
        image = image[top:bottom, left:right]

    currentHeight, currentWidth, currentChannels = image.shape

    # preparing matrix to be fed to the neural network
    currentHeightHalf = currentHeight / 2
    currentWidthHalf = currentWidth / 2
    if not DO_DIVIDE:
        imageResized = np.float32(cv2.resize(image, (nnConfig.widthInp, nnConfig.heightInp)))
        imageResized /= 255.0
        inputData[0, :, :, :] = imageResized[:, :, :]
        yCurr, cnnOutCurr = sess.run([yConv, cnnOut], {x_image: inputData, keepProb: 1.0})
        #cnnOutImage = visualizeCnnOutput(cnnOutCurr, cnnOutRescale = 10.0, doAddBorder = True)
        decipher_network_output(yCurr, nnConfig, imageResized)
    else:
        image_parts = divide_image(image, nnConfig.heightInp, nnConfig.widthInp)
        object_list = []
        for i, image_part in enumerate(image_parts):
            image_part = np.float32(image_part)
            image_part /= 255.0
            print i
            inputData[0, :, :, :] = image_part[:, :, :]
            yCurr = sess.run(yConv, {x_image: inputData, keepProb: 1.0})
            object_list_current = network_output_to_object_list(yCurr, nnConfig)
            for object_name in object_list_current:
                if not object_name in object_list:
                    object_list.append(object_name)
        print object_list
        print ""

    cv2.imshow("image", image)
    #cv2.imshow("cnn out", cnnOutImage)
    cv2.waitKey(0)
