from numpy.core.fromnumeric import size
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import random
import cv2
import mahotas as mh
import imutils

# directory of training data
train_dir = './rock-paper-scissor/rps/rps'
test_dir = './rock-paper-scissor/rps-test-set/rps-test-set'

def loadImg(i, dir):
    img = cv2.imread(dir)
    img = imutils.resize(img, width = 64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 20, 80)
    data = np.reshape(edged, (64*64))  
    return data

def main():
    # 840 images per train set, 2520 total
    train_rock = os.listdir(train_dir + '/rock')
    train_paper = os.listdir(train_dir + '/paper')
    train_scissors = os.listdir(train_dir + '/scissors')

    # 124 images per test set, 372 total
    test_rock = os.listdir(test_dir + '/rock')
    test_paper = os.listdir(test_dir + '/paper')
    test_scissors = os.listdir(test_dir + '/scissors')

    # # show random sample data
    # x = random.randint(0, len(train_rock))
    # cv2.imshow('rock', cv2.imread(train_dir + '/rock/' + train_rock[x]))
    # cv2.imshow('paper', cv2.imread(train_dir + '/paper/' + train_paper[x]))
    # cv2.imshow('scissors', cv2.imread(train_dir + '/scissors/' + train_scissors[x]))
    # cv2.waitKey(0)

    # paper 0 rock 1 scissors 2

    # importing the images
    trainX = np.ndarray(shape = (840*3,64*64), dtype=np.uint8)
    trainY = np.ndarray(shape = (840*3, 1), dtype=np.uint8)
    
    # processing image
    for i in range(len(train_paper)):
        dir = train_dir + '/paper/' + train_paper[i]
        trainX[i] = loadImg(i, dir)
        trainY[i] = 0
    
    for i in range(len(train_rock)):
        dir = train_dir + '/rock/' + train_rock[i]
        trainX[i] = loadImg(i, dir)
        trainY[i] = 1

    for i in range(len(train_rock)):
        dir = train_dir + '/scissors/' + train_scissors[i]
        trainX[i] = loadImg(i, dir)
        trainY[i] = 2

    trainY = np.reshape(trainY, (840*3))
    cv2.imshow(str(trainY[837]), np.reshape(trainX[837], (64, 64)))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()