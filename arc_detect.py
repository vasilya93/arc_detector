#!/usr/bin/python
from tensorflow.examples.tutorials.mnist import input_data
from prepare_dataset import DataSet
from image_check import getImageNames
from subprocess import call
from time import strftime
import time
import tensorflow as tf
import os.path
import os
import cv2

import numpy as np

from construct_network import weightVariable, biasVariable, \
        conv2d, maxPool2x2, constructMlp, constructCnn

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
DO_CROP = False

top = 221
left = 635
right = 1134
bottom = 479

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

saver = tf.train.Saver()
currentDir = os.getcwd()
pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME
if not os.path.exists(pathCurrent):
    print("Error: could not find current model directory. Exiting...")

modelFilePath = pathCurrent + "/" + MODEL_FILENAME
saver.restore(sess, modelFilePath)

testImageNames = getImageNames(TESTSET_DIR)

inputData = np.zeros((1, heightInp, widthInp, channelsInp), np.float32)
for imageName in testImageNames:
    imagePath = TESTSET_DIR + "/" + imageName
    image = cv2.imread(imagePath)

    if DO_CROP:
        image = image[top:bottom, left:right]

    currentHeight, currentWidth, currentChannels = image.shape
    currentHeightHalf = currentHeight / 2
    currentWidthHalf = currentWidth / 2
    imageResized = np.float32(cv2.resize(image, (widthInp, heightInp)))
    imageResized /= 255.0
    inputData[0, :, :, :] = imageResized[:, :, :]

    start_time = time.time()
    yCurr = sess.run(yConv, {x_image: inputData, keepProb: 1.0})

    objectX1 = np.int(yCurr[0][0] * currentWidthHalf + currentWidthHalf)
    objectY1 = np.int(yCurr[0][1] * currentHeightHalf + currentHeightHalf)
    cv2.circle(image, (objectX1, objectY1), 10, (255, 255, 255), 2)

    if numObjects > 1:
        objectX2 = np.int(yCurr[0][2] * currentWidthHalf + currentWidthHalf)
        objectY2 = np.int(yCurr[0][3] * currentHeightHalf + currentHeightHalf)
        cv2.circle(image, (objectX2, objectY2), 10, (0, 255, 255), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
