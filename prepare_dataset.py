#!/usr/bin/python
import cv2
import numpy as np
from random import randint

from config_file import readConfigFile
from ast import literal_eval

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

        self.outputTraining_ = None
        self.outputTesting_ = None
        self.outputValidation_ = None

        #self.propertyNames_ = ["x_left_top", "y_left_top", \
        #        "x_right_top", "y_right_top", \
        #        "x_left_bottom", "y_left_bottom", \
        #        "x_right_bottom", "y_right_bottom"]

        self.propertyNames_ = ["x_rel", "y_rel"]

    def prepareDataset(self, parentDir):
        dictFiles = readConfigFile(parentDir)
        outputData = None
        imageNames = []
        for imageName in dictFiles:
            imageNames.append(parentDir + "/" + imageName)
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
        self.outputTraining_ = outputData[0:trainingSize, :]

        testingSetEnd = trainingSize + testingSize
        self.imageNamesTesting_ = imageNames[trainingSize:testingSetEnd]
        self.outputTesting_ = outputData[trainingSize:testingSetEnd, :]

        self.imageNamesValidation_ = imageNames[testingSetEnd:dataSize]
        self.outputValidation_ = outputData[testingSetEnd:dataSize, :]

        self.trainingBatchBeg_ = 0
        self.testingBatchBeg_ = 0
        self.validationBatchBeg_ = 0

        return True

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

    def _prepareTrainingIndeces(self, batchSize, imageNames, batchBeg):
        setSize = len(imageNames)
        if batchSize > setSize:
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
