#!/usr/bin/python
from tensorflow.examples.tutorials.mnist import input_data
from prepare_dataset import DataSet
from subprocess import call
from time import strftime
import tensorflow as tf
import os.path
import os

# TODO: save size of the image in the dir with the
# model

DATASET_DIR = "dataset"
MODEL_DIR = "model"
CURRENT_MODEL_NAME = "current"
MODEL_FILENAME = "model.ckpt"
DO_USE_PREV_MODEL = False


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

W_conv1 = weight_variable([5, 5, channelsInp, 32])
b_conv1 = bias_variable([32])

x_image = tf.placeholder(tf.float32, shape = (None, heightInp, widthInp, channelsInp))
y_ = tf.placeholder(tf.float32, shape = (None, sizeOut))

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

convOutputSize = heightInpQuarter * widthInpQuarter * 64
mlpOutSize1 = 1024
W_fc1 = weight_variable([convOutputSize, mlpOutSize1])
b_fc1 = bias_variable([mlpOutSize1])

h_pool2_flat = tf.reshape(h_pool2, [-1, convOutputSize])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

mlpInSize2 = mlpOutSize1
W_fc2 = weight_variable([mlpInSize2, sizeOut])
b_fc2 = bias_variable([sizeOut])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
squared_deltas = tf.square(y_conv - y_)
loss = tf.reduce_sum(squared_deltas)

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

saver = tf.train.Saver()
currentDir = os.getcwd()
pathCurrent = currentDir + "/" + MODEL_DIR + "/" + CURRENT_MODEL_NAME
if os.path.exists(pathCurrent) and DO_USE_PREV_MODEL:
    modelFilePath = pathCurrent + "/" + MODEL_FILENAME
    saver.restore(sess, modelFilePath)
    dataSet.setRandomTrainingBeginning()
else:
    init = tf.global_variables_initializer()
    sess.run(init)

for i in range(2000):
    batchInput, batchOutput = dataSet.getTrainingBatch(100)
    stepCurr, lossCurr = sess.run([train_step, loss], {x_image: batchInput, y_: batchOutput, keep_prob: 0.5})
    print("%d: loss is %f" % (i, lossCurr))

stringDateTime = strftime("%y%m%d_%H%M%S")
modelDirPath = currentDir + "/" + MODEL_DIR + "/" + stringDateTime
call(["mkdir", "-p", modelDirPath])
modelPath = modelDirPath + "/" + MODEL_FILENAME

saver.save(sess, modelPath)

if os.path.exists(pathCurrent):
    call(["rm", pathCurrent])

call(["ln", "-s", modelDirPath, pathCurrent])
