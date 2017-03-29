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
print("sizeOut is %d" % sizeOut)

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

averageAbsDelta = tf.abs(yConv - y_) / sizeOut
absLoss = tf.reduce_sum(averageAbsDelta)

squared_deltas = tf.square(yConv - y_)
loss = tf.reduce_sum(squared_deltas)

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

#print "shape of squared_deltas is " + str(squared_deltas.get_shape())
#print "shape of loss is " + str(loss.get_shape())
#print "shape of train_step is " + str(train_step.get_shape())

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

trainBatchSize = 100
testBatchSize = 1000
for i in range(100):
    if i % 10 == 0:
        batchInput, batchOutput = dataSet.getTestingBatch(testBatchSize)
        absLossCurr = sess.run(absLoss, {x_image: batchInput, y_: batchOutput, keepProb: 1.0})
        absLossCurr = absLossCurr / testBatchSize
        print("%d: average error is %f" %(i, absLossCurr))

    batchInput, batchOutput = dataSet.getTrainingBatch(trainBatchSize)
    stepCurr, lossCurr = sess.run([train_step, loss], {x_image: batchInput, y_: batchOutput, keepProb: 0.5})
    print("%d: loss is %f" % (i, lossCurr))

stringDateTime = strftime("%y%m%d_%H%M%S")
modelDirPath = currentDir + "/" + MODEL_DIR + "/" + stringDateTime
call(["mkdir", "-p", modelDirPath])
modelPath = modelDirPath + "/" + MODEL_FILENAME

saver.save(sess, modelPath)

if os.path.exists(pathCurrent):
    call(["rm", pathCurrent])

call(["ln", "-s", modelDirPath, pathCurrent])
