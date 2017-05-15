#!/usr/bin/python
import cv2
import os
import numpy as np
from random import randint

from config_file import readConfigFile
from ast import literal_eval

IMAGES_DIR = "images"
MARKUP_DIR = "markup"

# there must be a functions which accepts just filenames, and on the basis of
# the filenames it prepares all the data and images

class DataSet:
    def __init__(self):
        self.trainingProportion_ = 0.85
        self.testingProportion_ = 0.10
        self.validationProportion_ = 1 - self.trainingProportion_ - \
            self.testingProportion_

        self.trainingBatchBeg_ = 0
        self.testingBatchBeg_ = 0
        self.validationBatchBeg_ = 0

        self.imageNamesTraining_ = []
        self.imageNamesTesting_ = []
        self.imageNamesValidation_ = []

        self.markupNamesTraining_ = []
        self.markupNamesTesting_ = []
        self.markupNamesValidation_ = []

        self.outputTraining_ = None
        self.outputTesting_ = None
        self.outputValidation_ = None

        self.propertyNames_ = ["is_present", "x_rel", "y_rel"]
        self.objectNames_ = []

        #self.propertyNames_ = ["x_left_top", "y_left_top", \
        #        "x_right_top", "y_right_top", \
        #        "x_left_bottom", "y_left_bottom", \
        #        "x_right_bottom", "y_right_bottom"]

        #self.propertyNames_ = ["x_rel", "y_rel"]

    def prepareDataset(self, parentDir):
        imagesDir = parentDir + "/" + IMAGES_DIR
        markupDir = parentDir + "/" + MARKUP_DIR

        dictFiles = readConfigFile(imagesDir)
        outputData = None
        imageNames = []
        markupNames = []
        for imageName in dictFiles:
            imageNames.append(imagesDir + "/" + imageName)
            markupNames.append(markupDir + "/" + imageName)
            if len(self.objectNames_) <= 0:
                for objName in dictFiles[imageName]:
                    self.objectNames_.append(os.path.splitext(objName)[0])
            newData = []
            for objName in dictFiles[imageName]:
                objectData = literal_eval(dictFiles[imageName][objName])
                addedData = []
                for propertyName in self.propertyNames_:
                    addedData.append(objectData[propertyName])
                newData.extend(addedData)
            if outputData is None:
                outputData = newData
            else:
                outputData = np.vstack([outputData, newData])

        dataSize = len(imageNames)
        if dataSize == 0:
            return False

        trainingSize = np.int(self.trainingProportion_ * dataSize)
        testingSize = np.int(self.testingProportion_ * dataSize)
        validationSize = np.int(dataSize - trainingSize - testingSize)

        self.imageNamesTraining_ = imageNames[0:trainingSize]
        self.markupNamesTraining_ = markupNames[0:trainingSize]
        self.outputTraining_ = outputData[0:trainingSize, :]

        testingSetEnd = trainingSize + testingSize
        self.imageNamesTesting_ = imageNames[trainingSize:testingSetEnd]
        self.markupNamesTesting_ = markupNames[trainingSize:testingSetEnd]
        self.outputTesting_ = outputData[trainingSize:testingSetEnd, :]

        self.imageNamesValidation_ = imageNames[testingSetEnd:dataSize]
        self.markupNamesValidation_ = markupNames[testingSetEnd:dataSize]
        self.outputValidation_ = outputData[testingSetEnd:dataSize, :]

        self.trainingBatchBeg_ = 0
        self.testingBatchBeg_ = 0
        self.validationBatchBeg_ = 0

        return True

    def getObjectNames(self):
        return self.objectNames_

    def getTrainingSetSize(self):
        return len(self.imageNamesTraining_)

    def getTestingSetSize(self):
        return len(self.imageNamesTesting_)

    def getValidationSetSize(self):
        return len(self.imageNamesValidation_)

    def getValidationSetSize(self):
        return len(self.imageNamesValidation_)

    def getOutSizeObject(self):
        return len(self.propertyNames_)

    def getOutSizeTotal(self):
        if self.outputTraining_ is None:
            return 0
        else:
            return self.outputTraining_.shape[1]


    def getInputSize(self):
        height, width, channels = self.getInputDimensions()
        return height * width * channels


    def getInputDimensions(self):
        if len(self.imageNamesTesting_) == 0:
            return (0, 0, 0)
        else:
            image = cv2.imread(self.imageNamesTesting_[0])
            if image is None:
                return (0, 0, 0)
            else:
                return image.shape

    def setRandomTrainingBeginning(self):
        setSize = len(self.imageNamesTraining_)
        self.trainingBatchBeg_ = randint(0, setSize - 1)

    def getTrainingBatchMarkup(self, batchSize, heightOut, widthOut):
        return self._getBatchMarkup(batchSize, \
                self.imageNamesTraining_, \
                self.trainingBatchBeg_, \
                self.markupNamesTraining_, \
                heightOut, \
                widthOut)

    def getTestingBatchMarkup(self, batchSize, heightOut, widthOut):
        return self._getBatchMarkup(batchSize, \
                self.imageNamesTesting_, \
                self.testingBatchBeg_, \
                self.markupNamesTesting_, \
                heightOut, \
                widthOut)

    def getTestingBatchMarkup(self, batchSize, heightOut, widthOut):
        return self._getBatchMarkup(batchSize, \
                self.imageNamesValidation_, \
                self.validationBatchBeg_, \
                self.markupNamesValidation_, \
                heightOut, \
                widthOut)

    def getTrainingBatch(self, batchSize):
        return self._getBatch(batchSize, \
                self.imageNamesTraining_, \
                self.trainingBatchBeg_, \
                self.outputTraining_)

    def getTestingBatch(self, batchSize):
        return self._getBatch(batchSize, \
                self.imageNamesTesting_, \
                self.testingBatchBeg_, \
                self.outputTesting_)

    def getValidationBatch(self, batchSize):
        return self._getBatch(batchSize, \
                self.imageNamesValidation_, \
                self.validationBatchBeg_, \
                self.outputValidation_)

    def _getBatch(self, batchSize, imageNames, batchBeg, output):
        indeces = self._prepareTrainingIndeces(batchSize, imageNames, batchBeg)
        if indeces == None:
            return (None, None)

        batchImageNames = [imageNames[i] for i in indeces]
        batchInput = self._prepareInputMatrices(batchImageNames)

        batchOutput = [output[i] for i in indeces]

        return (batchInput, batchOutput)

    def _getBatchMarkup(self, batchSize, imageNames, batchBeg, markupNames, heightOut, widthOut):
        indeces = self._prepareTrainingIndeces(batchSize, imageNames, batchBeg)
        if indeces == None:
            return (None, None)

        batchImageNames = [imageNames[i] for i in indeces]
        batchMarkupNames = [markupNames[i] for i in indeces]

        batchInput = self._prepareInputMatrices(batchImageNames)
        batchOutput = self._prepareOutputMatrices(batchMarkupNames, heightOut, widthOut)

        return (batchInput, batchOutput)

    def _prepareTrainingIndeces(self, batchSize, imageNames, batchBeg):
        setSize = len(imageNames)
        if batchSize > setSize:
            print("Error: requested batch size (%d) is more than there are elements in the set (%d)" % \
                    (batchSize, setSize))
            return None

        leftToEnd = setSize - batchBeg
        if leftToEnd >= batchSize:
            batchBegNew = batchBeg + batchSize
            indeces = range(batchBeg, batchBegNew)
            batchBeg = batchBegNew
        else:
            indeces = range(batchBeg, setSize)
            leftToCopy = batchSize - leftToEnd
            indeces.extend(range(0, leftToCopy))
            batchBeg = leftToCopy

	return indeces

    def _prepareInputImages(self, batchImageNames):
        inputData = None
        for imageName in batchImageNames:
            imageCurrent = np.float64(cv2.imread(imageName))
            imageCurrent /= 255.0
            height, width, channels = imageCurrent.shape
            if height == 0 or width == 0:
                continue
            dimension = height * width * channels
            imageCurrent = imageCurrent.reshape(1, dimension)
            if inputData == None:
                inputData = imageCurrent
            else:
                inputData = np.vstack([inputData, imageCurrent])
        return inputData

    def _prepareInputMatrices(self, batchImageNames):
        heightGen, widthGen, channelsGen = self.getInputDimensions()
        dataSize = len(batchImageNames)
        inputData = np.zeros((dataSize, heightGen, widthGen, channelsGen), np.float32)
        for i in range(len(batchImageNames)):
            imageName = batchImageNames[i]
            imageCurrent = np.float32(cv2.imread(imageName))
            imageCurrent /= 255.0
            height, width, channels = imageCurrent.shape
            if height != heightGen or width != widthGen or channels != channelsGen:
                continue
    	    inputData[i, :, :, :] = imageCurrent[:, :, :]
        return inputData

    def _prepareOutputMatrices(self, batchMarkupNames, height, width):
        objectsNum = len(self.objectNames_)
        dataSize = len(batchMarkupNames)
        outputData = np.zeros((dataSize, height, width, objectsNum), np.float32)
        for i in range(len(batchMarkupNames)):
            markupName = batchMarkupNames[i]
            imageMarkup = cv2.imread(markupName, 0)
            imageMarkup = cv2.resize(imageMarkup, (width, height))
            imageCurrent = np.zeros((height, width, objectsNum), dtype = np.float32)
            for i in range(1, objectsNum + 1):
                currentObjectLayer = imageCurrent[:, :, i - 1]
                currentObjectLayer[imageMarkup == i] = 1.0
 
            outputData[i, :, :, :] = imageCurrent[:, :, :]

        return outputData
