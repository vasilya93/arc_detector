#!/usr/bin/python

import cv2
import numpy as np

from copy import deepcopy

def visualizeCnnOutput(cnnOut, cnnOutRescale, doAddBorder):
    outHeight = cnnOut.shape[1]
    outWidth = cnnOut.shape[2]
    outChannels = cnnOut.shape[3]
    rowSize = np.int(np.ceil(np.sqrt(outChannels)))
    cnnRawWidth = np.int(outWidth * rowSize)
    cnnRawHeight = np.int(outHeight * rowSize)

    if doAddBorder:
        borderComplement = (rowSize + 1) * 3
        cnnOutImage = np.zeros([cnnRawHeight + borderComplement, \
                cnnRawWidth + borderComplement], np.uint8)
    else:
        cnnOutImage = np.zeros([cnnRawHeight, \
                cnnRawWidth], np.uint8)

    channelCounter = 0
    for i in range(rowSize):
        yBeg = 3 + i * (outHeight + 3) if doAddBorder else i * outHeight
        yEnd = yBeg + outHeight
        if doAddBorder: # Draw regular horizontal line
            cnnOutImage[yBeg - 2, :] = 255
        for j in range(rowSize):
            xBeg = 3 + j * (outWidth + 3) if doAddBorder else j * outWidth
            xEnd = xBeg + outWidth
            currentChannelImage = deepcopy(cnnOut[0, :, :, channelCounter])
            currentChannelImage *= 255
            currentChannelImage = np.uint8(currentChannelImage)
            cnnOutImage[yBeg:yEnd, xBeg:xEnd] = currentChannelImage[:, :]
            if doAddBorder: # Draw regular vertical line
                cnnOutImage[yBeg - 1: yEnd + 1, xBeg - 2] = 255
            channelCounter += 1
            if channelCounter >= outChannels:
                break
        if doAddBorder: # Draw final vertical line
            cnnOutImage[yBeg - 1: yEnd + 1, xEnd + 1] = 255

        if channelCounter >= outChannels:
            if doAddBorder: # Draw downmost horizontal line
                cnnOutImage[yEnd + 1, :] = 255
            break

    if cnnOutRescale != 1.0:
        cnnOutImage = cv2.resize(cnnOutImage, (0, 0), fx = cnnOutRescale, fy = cnnOutRescale, interpolation = cv2.INTER_NEAREST)

    return cnnOutImage

