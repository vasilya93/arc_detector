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
        writeConfigFile(parentDir, dictConfig)

    def loadFromFile(self, parentDir):
        dictConfig = readConfigFile(parentDir)
        for key in dictConfig:
            self.heightInp = dictConfig[key]["height_inp"]
            self.widthInp = dictConfig[key]["width_inp"]
            self.channelsInp = dictConfig[key]["channels_inp"]
            self.sizeOut = dictConfig[key]["size_out"]
            self.sizeOutObject = dictConfig[key]["size_out_object"]
            self.numObjects = dictConfig[key]["num_objects"]
            self.cnnLayersSize = dictConfig[key]["cnn_layers_size"]
            self.convWindowSize = dictConfig[key]["conv_window_size"]
            self.mlpLayersSize = dictConfig[key]["mlp_layer_size"]
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

