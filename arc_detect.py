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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def constructMlp(input, inputSize, layerSize, outSize, keepProb = None):
    if keepProb is None:
        keepProb = tf.placeholder(tf.float32)

    W_fc1 = weight_variable([inputSize, layerSize])
    b_fc1 = bias_variable([layerSize])

    h_fc1 = tf.nn.relu(tf.matmul(input, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keepProb)

    W_fc2 = weight_variable([layerSize, outSize])
    b_fc2 = bias_variable([outSize])

    output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return (output, keepProb)

dataSet = DataSet()
isSuccess = dataSet.prepareDataset(DATASET_DIR)
if not isSuccess:
    print("Error: could not load dataset. Exiting...")
    exit(1)

heightInp, widthInp, channelsInp = dataSet.getInputDimensions()
heightInpHalf = divideByTwo(heightInp)
heightInpQuarter = divideByTwo(heightInpHalf)

widthInpHalf = divideByTwo(widthInp)
widthInpQuarter = divideByTwo(widthInpHalf)

sizeOut = dataSet.getOutputSize()

# TensorFlow model description

sess = tf.InteractiveSession()

convFeaturesNum1 = 48
W_conv1 = weight_variable([5, 5, channelsInp, convFeaturesNum1])
b_conv1 = bias_variable([convFeaturesNum1])

x_image = tf.placeholder(tf.float32, shape = (None, heightInp, widthInp, channelsInp))
y_ = tf.placeholder(tf.float32, shape = (None, sizeOut))

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

convFeaturesNum2 = convFeaturesNum1 * 2
W_conv2 = weight_variable([5, 5, convFeaturesNum1, convFeaturesNum2])
b_conv2 = bias_variable([convFeaturesNum2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

convOutputSize = heightInpQuarter * widthInpQuarter * convFeaturesNum2
h_pool2_flat = tf.reshape(h_pool2, [-1, convOutputSize])

mlpLayerSize = 1024
(yConv1, keepProb) = constructMlp(h_pool2_flat, convOutputSize, mlpLayerSize, 2)
(yConv2, keepProb) = constructMlp(h_pool2_flat, convOutputSize, mlpLayerSize, 2, keepProb)
yConv = tf.concat([yConv1, yConv2], 1)

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

    objectX2 = np.int(yCurr[0][2] * currentWidthHalf + currentWidthHalf)
    objectY2 = np.int(yCurr[0][3] * currentHeightHalf + currentHeightHalf)
    cv2.circle(image, (objectX2, objectY2), 10, (0, 255, 255), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
