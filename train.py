from numpy.core.fromnumeric import size
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from matplotlib import pyplot
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
    img = imutils.resize(img, width = 32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    T= mh.thresholding.otsu(blurred)
    gray[gray>T] = 255
    gray[gray<255] = 0
    data = np.reshape(gray, (32*32))  
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
    trainX = np.ndarray(shape = (2520,32*32), dtype=np.uint8)
    trainY = np.ndarray(shape = (2520, 1), dtype=np.uint8)
    
    # processing image
    cur_index = 0
    for i in range(len(train_paper)):
        dir = train_dir + '/paper/' + train_paper[i]
        trainX[cur_index] = loadImg(i, dir)
        trainY[cur_index] = 0
        cur_index += 1
        print('\rProcessed %s%% of the image.' % int((cur_index/2520)*100), end='')
    
    for i in range(len(train_rock)):
        dir = train_dir + '/rock/' + train_rock[i]
        trainX[cur_index] = loadImg(i, dir)
        trainY[cur_index] = 1
        cur_index += 1
        print('\rProcessed %s%% of the image.' % int((cur_index/2520)*100), end='')

    for i in range(len(train_rock)):
        dir = train_dir + '/scissors/' + train_scissors[i]
        trainX[cur_index] = loadImg(i, dir)
        trainY[cur_index] = 2
        cur_index += 1
        print('\rProcessed %s%% of the image.' % int((cur_index/2520)*100), end='')

    trainY = np.reshape(trainY, (2520))
    # cv2.imshow(str(trainY[837]), np.reshape(trainX[837], (32, 32)))

    # cv2.waitKey(0)
    # test data
    # importing the images
    testX = np.ndarray(shape = (372,32*32), dtype=np.uint8)
    testY = np.ndarray(shape = (372, 1), dtype=np.uint8)
    
    cur_index = 0
    # processing image
    for i in range(len(test_paper)):
        dir = test_dir + '/paper/' + test_paper[i]
        testX[cur_index] = loadImg(i, dir)
        testY[cur_index] = 0
        cur_index += 1
        print('\r%s' % cur_index, end='')
    
    for i in range(len(test_rock)):
        dir = test_dir + '/rock/' + test_rock[i]
        testX[cur_index] = loadImg(i, dir)
        testY[cur_index] = 1
        cur_index += 1
        print('\r%s' % cur_index, end='')

    for i in range(len(test_rock)):
        dir = test_dir + '/scissors/' + test_scissors[i]
        testX[cur_index] = loadImg(i, dir)
        testY[cur_index] = 2
        cur_index += 1
        print('\r%s' % cur_index, end='')

    testY = np.reshape(testY, (372))
    # cv2.destroyAllWindows()
    # cv2.imshow(str(testY[370]), np.reshape(testX[370], (32, 32)))

    # cv2.waitKey(0)

    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(trainX, trainY)
    preds = classifier.predict(testX)

    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, testY):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f'Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}')

    plot_confusion_matrix(classifier, testX, testY)
    pyplot.show()


if __name__ == '__main__':
    main()