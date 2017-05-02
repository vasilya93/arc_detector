#!/usr/bin/python
from config_file import writeConfigFile
from config_file import readConfigFile

class NnConfig:
    def __init__(self):
        self.heightInp = 0
        self.widthInp = 0
        self.channelsInp = 0
        self.sizeOut = 0
        self.sizeOutObject = 0
        self.numObjects = 0
        self.cnnLayersSize = [0]
        self.convWindowSize = [0]
        self.mlpLayersSize = [0]

        # Optimization details
        self.optimizationIterationsNum = 0
        self.optimizationStep = 0.0
        self.batchSize = 0
        self.error = 0
        self.objectNames = []

    def saveToFile(self, parentDir, stringDateTime):
        dictConfig = {}
        dictConfig[stringDateTime] = {}
        dictConfig[stringDateTime]["height_inp"] = self.heightInp
        dictConfig[stringDateTime]["width_inp"] = self.widthInp
        dictConfig[stringDateTime]["channels_inp"] = self.channelsInp
        dictConfig[stringDateTime]["size_out"] = self.sizeOut
        dictConfig[stringDateTime]["size_out_object"] = self.sizeOutObject
        dictConfig[stringDateTime]["num_objects"] = self.numObjects
        dictConfig[stringDateTime]["cnn_layers_size"] = self.cnnLayersSize
        dictConfig[stringDateTime]["conv_window_size"] = self.convWindowSize
        dictConfig[stringDateTime]["mlp_layer_size"] = self.mlpLayersSize
        dictConfig[stringDateTime]["optimization_iterations_num"] = self.optimizationIterationsNum
        dictConfig[stringDateTime]["optimization_step"] = self.optimizationStep
        dictConfig[stringDateTime]["batch_size"] = self.batchSize
        dictConfig[stringDateTime]["error"] = self.error
        dictConfig[stringDateTime]["object_names"] = self.objectNames
        writeConfigFile(parentDir, dictConfig)

    def loadFromFile(self, parentDir):
        dictConfig = readConfigFile(parentDir)
        for key in dictConfig:
            self.heightInp = eval(dictConfig[key]["height_inp"])
            self.widthInp = eval(dictConfig[key]["width_inp"])
            self.channelsInp = eval(dictConfig[key]["channels_inp"])
            self.sizeOut = eval(dictConfig[key]["size_out"])
            self.sizeOutObject = eval(dictConfig[key]["size_out_object"])
            self.numObjects = eval(dictConfig[key]["num_objects"])
            self.cnnLayersSize = eval(dictConfig[key]["cnn_layers_size"])
            self.convWindowSize = eval(dictConfig[key]["conv_window_size"])
            self.mlpLayersSize = eval(dictConfig[key]["mlp_layer_size"])
            self.optimizationIterationsNum = eval(dictConfig[key]["optimization_iterations_num"])
            self.optimizationStep = eval(dictConfig[key]["optimization_step"])
            self.batchSize = eval(dictConfig[key]["batch_size"])
            self.error = eval(dictConfig[key]["error"])
            self.objectNames = eval(dictConfig[key]["object_names"])
            break

    def printData(self):
        print "heightInp " + self.heightInp
        print "widthInp " + self.widthInp
        print "channelsInp " + self.channelsInp
        print "sizeOut " + self.sizeOut
        print "sizeOutObject" + self.sizeOutObject
        print "numObjects" + self.numObjects
        print "cnnLayersSize" + self.cnnLayersSize
        print "convWindowSize" + self.convWindowSize
        print "mlpLayersSize" + self.mlpLayersSize


    def markCommonRecord(self, allModelsPath, stringDateTime):
        writtenString = "%s: %s\t %s\t %s\t %f\n" % (stringDateTime, \
                str(self.cnnLayersSize), \
                str(self.convWindowSize), \
                str(self.mlpLayersSize), \
                self.error)

        with open(allModelsPath + "/record.txt", "a") as recordFile:
            recordFile.write(writtenString)
